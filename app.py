import streamlit as st
from pymongo import MongoClient
from pydantic import BaseModel, ValidationError
from typing import List
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pdfminer_high_level import extract_text as _extract_text  # fallback alias if needed
from pdfminer.high_level import extract_text
import json
import re

st.set_page_config(layout="wide", page_title="금천구 감사결과 PDF 파싱 서비스")

# secrets에서 키/URI 불러오기
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
client_db = MongoClient(st.secrets["MONGO_URI"])

db = client_db["json_db"]
counter_collection = db["Yangsan_Audit"]

# 최신 모델 (빠름/저비용)
MODEL_NAME = "gemini-2.5-flash"
model = genai.GenerativeModel(model_name=MODEL_NAME)

# -------- Pydantic 스키마 --------
class AuditResult(BaseModel):
    건명: str
    처분: str
    관련규정: str
    지적사항: str

class ResearchPaperExtraction(BaseModel):
    감사연도: str
    피감기관: str
    감사결과: List[AuditResult]

# -------- 유틸: PDF 추출 / JSON 보정 / 복구 / 안전 레닥션 --------
def extract_text_from_doc(file):
    try:
        return extract_text(file)
    except Exception:
        # 일부 환경에서 모듈 경로 차이 대응
        return _extract_text(file)

def coerce_json_from_text(raw: str) -> str:
    """코드펜스/잡텍스트 제거하고 최외곽 JSON 블록만 추출"""
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return s

REDACT_TOKEN = "【비공개】"

def redact_for_safety(text: str) -> str:
    s = text
    # 이메일, URL
    s = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", REDACT_TOKEN, s)
    s = re.sub(r"https?://\S+", REDACT_TOKEN, s)
    # 전화번호(국내)
    s = re.sub(r"\b(?:0\d{1,2}-\d{3,4}-\d{4})\b", REDACT_TOKEN, s)
    s = re.sub(r"\b\d{3}-\d{4}-\d{4}\b", REDACT_TOKEN, s)
    # 사건/문서번호류
    s = re.sub(r"(?:\d{4}[^\s]{0,4}\d{3,8})", REDACT_TOKEN, s)
    s = re.sub(r"\b\d{4}-\d{3,6}\b", REDACT_TOKEN, s)
    # 큰 숫자(금액 등) 과도 차단 완화용
    s = re.sub(r"\b\d{7,}\b", REDACT_TOKEN, s)
    # 사람 이름+직함(보수적 처리)
    s = re.sub(r"([가-힣]{2,4})\s?(과장|팀장|위원장|위원|계장|주무관)", REDACT_TOKEN, s)
    # 대괄호식 내부 식별 흔적
    s = re.sub(r"\[[^\]]{1,30}\]", REDACT_TOKEN, s)
    return s

def filter_relevant(text: str) -> str:
    keys = ("시정","주의","기타","회수","추징","추급","환급","징계","훈계","관련규정")
    lines = text.splitlines()
    picked = [ln for ln in lines if any(k in ln for k in keys)]
    return "\n".join(picked[:4000])

# Google SDK에서 지원하는 4개 카테고리만 설정 (dict)
SAFETY_RELAXED = {
    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_SEXUAL: HarmBlockThreshold.BLOCK_NONE,
    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
}

# -------- 세션 상태 --------
if "structured_json" not in st.session_state:
    st.session_state["structured_json"] = None
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = None

st.title("금천구 감사결과 PDF 파일 파싱 서비스")

col1, col2 = st.columns(2)

# 좌측: 업로드
with col1:
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_doc(uploaded_file)
        st.session_state["extracted_text"] = text
        with st.expander("PDF에서 추출된 텍스트 확인하기"):
            st.write(st.session_state["extracted_text"])

# 우측: 구조화/저장
with col2:
    if st.session_state.get("extracted_text"):
        st.subheader("RAG_Parse_PDF")

        if st.button("AI로 구조화 분석하기"):
            with st.spinner("Structured Outputs..."):
                try:
                    system_msg = (
                        "You are an expert at structured data extraction. "
                        "You will be given unstructured text from a research paper and should convert it into the given structure."
                    )
                    # 입력 레닥션 및 핵심 추출
                    raw_text = st.session_state["extracted_text"]
                    focused = filter_relevant(raw_text)
                    redacted = redact_for_safety(focused)
                    user_msg = (
                        f"{redacted} 내용 중 '시정','주의','기타','회수(추징)','추급(환급)','징계','훈계(경고)' 등 "
                        "처분결과 기준으로 자료를 모두 만들고, 관련규정은 요약하지 말고 모두 입력해 주세요. "
                        "민감정보(개인명·전화·사건번호·이메일·URL 등)는 절대 포함하지 마세요."
                    )

                    # 1차 호출
                    resp = model.generate_content(
                        [system_msg, user_msg],
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=ResearchPaperExtraction,
                            temperature=0,
                            max_output_tokens=8192,
                        ),
                    )

                    def is_safety_blocked(r) -> bool:
                        return bool(getattr(r, "candidates", None)) and r.candidates[0].finish_reason == 2

                    if is_safety_blocked(resp):
                        # 2차: 안전성 완화 설정으로 재시도 (동일 입력)
                        resp = model.generate_content(
                            [system_msg, user_msg],
                            generation_config=genai.GenerationConfig(
                                response_mime_type="application/json",
                                response_schema=ResearchPaperExtraction,
                                temperature=0,
                                max_output_tokens=8192,
                            ),
                            safety_settings=SAFETY_RELAXED,
                        )

                    if is_safety_blocked(resp):
                        st.error("안전성 필터에 의해 응답이 차단되었습니다. 텍스트 범위를 줄이거나 표/개인정보를 제외하고 다시 시도해 주세요.")
                        if getattr(resp, "prompt_feedback", None):
                            st.caption(f"prompt_feedback: {resp.prompt_feedback}")
                        st.stop()

                    # 1차 파싱
                    try:
                        structured = ResearchPaperExtraction.model_validate_json(resp.text)
                    except (ValidationError, json.JSONDecodeError, Exception):
                        # 2차: 보정(coerce) 후 재파싱
                        cleaned = coerce_json_from_text(getattr(resp, "text", "") or "")
                        try:
                            data = json.loads(cleaned)
                        except json.JSONDecodeError:
                            # 3차: 복구(repair) 재요청 (설명/코드펜스 금지, JSON만)
                            repair_prompt = f"""
다음 응답은 JSON 문법 오류가 있습니다. 아래 스키마에 맞게 유효한 JSON만 출력하세요. 설명/코드펜스 금지.

SCHEMA:
- 감사연도: string
- 피감기관: string
- 감사결과: list of objects with fields ["건명","처분","관련규정","지적사항"]

BROKEN:
{getattr(resp, 'text', '')}
"""
                            repair = model.generate_content(
                                [repair_prompt],
                                generation_config=genai.GenerationConfig(
                                    response_mime_type="application/json",
                                    response_schema=ResearchPaperExtraction,
                                    temperature=0,
                                    max_output_tokens=4096,
                                ),
                                safety_settings=SAFETY_RELAXED,
                            )
                            if is_safety_blocked(repair):
                                st.error("안전성 필터로 인해 JSON 복구가 차단되었습니다. 입력을 더 축약하거나 민감정보를 제거해 주세요.")
                                st.stop()
                            structured = ResearchPaperExtraction.model_validate_json(repair.text)
                        else:
                            # 상단 키 누락 시 최소 보정
                            if "감사연도" not in data or "피감기관" not in data:
                                data = {
                                    "감사연도": data.get("감사연도", ""),
                                    "피감기관": data.get("피감기관", ""),
                                    "감사결과": data.get("감사결과", []),
                                }
                            structured = ResearchPaperExtraction(**data)

                    st.session_state["structured_json"] = structured

                    st.write("구조화된 JSON 데이터:")
                    with st.expander("구조화된 JSON 데이터:"):
                        st.json(structured.model_dump())

                except Exception as e:
                    # SAFETY 차단 시에는 resp.text가 없을 수 있으므로 접근하지 않음
                    st.error(f"Gemini API 호출 또는 응답 처리 중 오류 발생: {e}")

        if st.session_state.get("structured_json") and st.button("MongoDB 저장"):
            with st.spinner("MongoDB Save..."):
                try:
                    counter_collection.insert_one(st.session_state["structured_json"].model_dump())
                    st.success("MongoDB에 데이터 저장 완료!")
                except Exception as e:
                    st.error(f"데이터 저장 중 오류 발생: {e}")
    else:
        st.markdown(
            """본 서비스는 문서 기반 RAG 시스템 개발을 지원하기 위해 설계되었습니다.

1) PDF에서 텍스트 추출  
2) AI로 구조화(JSON)  
3) MongoDB에 저장 및 검색
"""
        )

st.markdown("---")

# 검색
search_query = st.text_input("검색할 단어 또는 문장을 입력하세요:")

if search_query:
    try:
        query = {
            "감사결과": {
                "$elemMatch": {
                    "$or": [
                        {"건명": {"$regex": search_query, "$options": "i"}},
                        {"처분": {"$regex": search_query, "$options": "i"}},
                        {"관련규정": {"$regex": search_query, "$options": "i"}},
                        {"지적사항": {"$regex": search_query, "$options": "i"}},
                    ]
                }
            }
        }
        result_list = list(counter_collection.find(query))

        if result_list:
            for idx, doc in enumerate(result_list, start=1):
                st.markdown(f"### 결과 {idx}")
                st.write(f"**감사연도:** {doc.get('감사연도')}")
                st.write(f"**피감기관:** {doc.get('피감기관')}")
                for audit in doc.get("감사결과", []):
                    blob = (
                        audit.get("건명", "")
                        + audit.get("처분", "")
                        + audit.get("관련규정", "")
                        + audit.get("지적사항", "")
                    )
                    if search_query.lower() in blob.lower():
                        st.write(f"**건명:** {audit.get('건명')}")
                        st.write(f"**처분:** {audit.get('처분')}")
                        st.write(f"**관련규정:** {audit.get('관련규정')}")
                        st.write(f"**지적사항:** {audit.get('지적사항')}")
                        st.markdown("---")
        else:
            st.info("검색 결과가 없습니다.")
    except Exception as e:
        st.error(f"검색 중 오류 발생: {e}")
else:
    st.warning("검색어를 입력해주세요.")