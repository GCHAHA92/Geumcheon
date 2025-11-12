import streamlit as st
from pymongo import MongoClient
from pydantic import BaseModel, ValidationError
from typing import List
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
from pdfminer.high_level import extract_text
import json
import re

st.set_page_config(layout="wide", page_title="금천구 감사결과 PDF 파싱 서비스")

# --- Secrets ---
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
client_db = MongoClient(st.secrets["MONGO_URI"])

db = client_db["json_db"]
counter_collection = db["Yangsan_Audit"]

# --- Model (무료 티어/안정) ---
MODEL_NAME = "gemini-2.5-flash"  # 무료 티어/속도 우선, 스키마 강제 지원
model = genai.GenerativeModel(model_name=MODEL_NAME)

# --- Pydantic Schemas ---
class AuditResult(BaseModel):
    건명: str
    처분: str
    관련규정: str
    지적사항: str

class ResearchPaperExtraction(BaseModel):
    감사연도: str
    피감기관: str
    감사결과: List[AuditResult]

class ChunkExtraction(BaseModel):
    감사결과: List[AuditResult]

# --- Utils: text, chunk, json repair ---
def extract_text_from_doc(file):
    return extract_text(file)

def coerce_json_from_text(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    start, end = s.find("{"), s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start:end + 1]
    return s

def filter_relevant(text: str) -> str:
    keys = ("시정","주의","기타","회수","추징","추급","환급","징계","훈계","관련규정")
    lines = text.splitlines()
    picked = [ln for ln in lines if any(k in ln for k in keys)]
    picked = [ln for ln in picked if not re.search(r"\d{2,}/\d{2,}|-{5,}|={5,}", ln)]
    return "\n".join(picked[:6000])

def chunk_text(s: str, size: int = 3000, overlap: int = 150):
    i, n = 0, len(s)
    while i < n:
        yield s[i:i+size]
        i += size - overlap

# --- Safety enums (버전 호환) ---
def _cat(cls, *names):
    for n in names:
        if hasattr(cls, n):
            return getattr(cls, n)
    raise AttributeError(f"{cls.__name__} has none of: {names}")

SEXUAL_CAT = _cat(HarmCategory,
    "HARM_CATEGORY_SEXUALLY_EXPLICIT",
    "HARM_CATEGORY_SEXUAL_CONTENT",
    "HARM_CATEGORY_SEXUAL",
)

SAFETY_RELAXED = {
    _cat(HarmCategory, "HARM_CATEGORY_HATE_SPEECH"):        HarmBlockThreshold.BLOCK_NONE,
    _cat(HarmCategory, "HARM_CATEGORY_HARASSMENT"):         HarmBlockThreshold.BLOCK_NONE,
    SEXUAL_CAT:                                              HarmBlockThreshold.BLOCK_NONE,
    _cat(HarmCategory, "HARM_CATEGORY_DANGEROUS_CONTENT"):  HarmBlockThreshold.BLOCK_NONE,
}

# --- Helpers to call model safely ---
def is_safety_blocked(resp) -> bool:
    return bool(getattr(resp, "candidates", None)) and resp.candidates[0].finish_reason == 2

def call_with_optional_safety(messages, schema_model):
    gen_config = genai.GenerationConfig(
        response_mime_type="application/json",
        response_schema=schema_model,
        temperature=0,
        max_output_tokens=8192,
    )
    try:
        resp = model.generate_content(messages, generation_config=gen_config)
    except Exception as e:
        msg = str(e)
        if "Quota exceeded" in msg or "rate limit" in msg.lower():
            st.error("무료 쿼터를 초과했습니다. 모델을 'gemini-2.5-flash'로 바꾸거나, 입력 길이를 줄여 다시 시도하세요.")
            st.stop()
        raise
    if is_safety_blocked(resp):
        try:
            resp = model.generate_content(messages, generation_config=gen_config, safety_settings=SAFETY_RELAXED)
        except Exception:
            resp = model.generate_content(messages, generation_config=gen_config)
    return resp

# --- Session State ---
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

        if st.button("AI로 구조화 분석하기 (단일·스키마 강제)"):
            with st.spinner("Structured Outputs (single call)..."):
                try:
                    system_msg = (
                        "You are an expert at structured data extraction. "
                        "You will be given unstructured text from a research paper and should convert it into the given structure."
                    )
                    raw_text = st.session_state["extracted_text"]
                    focused = filter_relevant(raw_text)

                    user_msg = (
                        f"{focused} 내용 중 '시정','주의','기타','회수(추징)','추급(환급)','징계','훈계(경고)' 등 "
                        "처분결과 기준으로 자료를 모두 만들고, 관련규정은 요약하지 말고 모두 입력해 주세요."
                    )

                    resp = call_with_optional_safety([system_msg, user_msg], ResearchPaperExtraction)

                    if is_safety_blocked(resp):
                        st.error("안전성 필터에 의해 응답이 차단되었습니다. 텍스트 범위를 줄이거나 표/개인정보를 제외하고 다시 시도해 주세요.")
                        st.stop()

                    parsed_obj = getattr(resp, "parsed", None)
                    if parsed_obj is not None:
                        structured = parsed_obj
                    else:
                        try:
                            structured = ResearchPaperExtraction.model_validate_json(resp.text)
                        except Exception:
                            cleaned = coerce_json_from_text(getattr(resp, "text", "") or "")
                            try:
                                data = json.loads(cleaned)
                            except json.JSONDecodeError:
                                repair_prompt = f"""
다음 응답은 JSON 문법 오류가 있습니다. 아래 스키마에 맞게 유효한 JSON만 출력하세요. 설명/코드펜스 금지.

SCHEMA:
- 감사연도: string
- 피감기관: string
- 감사결과: list of objects with fields ["건명","처분","관련규정","지적사항"]

BROKEN:
{getattr(resp, 'text', '')}
"""
                                repair = call_with_optional_safety([repair_prompt], ResearchPaperExtraction)
                                if is_safety_blocked(repair):
                                    st.error("안전성 필터로 인해 JSON 복구가 차단되었습니다. 입력을 더 축약하거나 민감정보를 제거해 주세요.")
                                    st.stop()
                                structured = ResearchPaperExtraction.model_validate_json(repair.text)
                            else:
                                if "감사연도" not in data or "피감기관" not in data:
                                    data = {"감사연도": data.get("감사연도", ""),
                                            "피감기관": data.get("피감기관", ""),
                                            "감사결과": data.get("감사결과", [])}
                                structured = ResearchPaperExtraction(**data)

                    st.session_state["structured_json"] = structured

                    st.write("구조화된 JSON 데이터:")
                    with st.expander("구조화된 JSON 데이터:"):
                        st.json(structured.model_dump())

                except Exception as e:
                    st.error(f"Gemini API 호출 또는 응답 처리 중 오류 발생: {e}")

        if st.button("AI로 구조화 분석하기 (조각 처리·병합)"):
            with st.spinner("Structured Outputs (chunking)..."):
                try:
                    system_msg = (
                        "You are an expert at structured data extraction. "
                        "You will be given unstructured text from a research paper and should convert it into the given structure."
                    )
                    raw_text = st.session_state["extracted_text"]
                    focused = filter_relevant(raw_text)

                    parts = list(chunk_text(focused, size=3000, overlap=150))
                    all_items: List[AuditResult] = []

                    for idx, part in enumerate(parts, start=1):
                        user_msg = (
                            f"PART {idx}/{len(parts)}\n\n" +
                            f"{part}\n\n" +
                            "위 텍스트에서 '건명','처분','관련규정','지적사항'만 추출하여 JSON으로 반환하세요. "
                            "상위 키는 '감사결과' 하나만 포함합니다. 처분은 '시정','주의','기타','회수(추징)','추급(환급)','징계','훈계(경고)'만 사용. "
                            "관련규정은 요약 금지(원문 그대로). 민감정보(개인명·전화·사건번호·이메일·URL)는 절대 포함하지 마세요."
                        )

                        resp = call_with_optional_safety([system_msg, user_msg], ChunkExtraction)

                        if is_safety_blocked(resp):
                            st.warning(f"PART {idx}: 안전성 필터 차단으로 건너뜀")
                            continue

                        try:
                            chunk_struct = ChunkExtraction.model_validate_json(resp.text)
                        except Exception:
                            cleaned = coerce_json_from_text(getattr(resp, "text", "") or "")
                            try:
                                data = json.loads(cleaned)
                                if "감사결과" not in data:
                                    data = {"감사결과": []}
                                chunk_struct = ChunkExtraction(**data)
                            except json.JSONDecodeError:
                                repair_prompt = f"""
다음 응답은 JSON 문법 오류가 있습니다. 아래 스키마에 맞게 유효한 JSON만 출력하세요. 설명/코드펜스 금지.

SCHEMA:
- 감사결과: list of objects with fields ["건명","처분","관련규정","지적사항"]

BROKEN:
{getattr(resp, 'text', '')}
"""
                                repair = call_with_optional_safety([repair_prompt], ChunkExtraction)
                                if is_safety_blocked(repair):
                                    st.warning(f"PART {idx}: 복구도 차단되어 스킵")
                                    continue
                                chunk_struct = ChunkExtraction.model_validate_json(repair.text)

                        all_items.extend(chunk_struct.감사결과)

                    if not all_items:
                        st.error("추출된 항목이 없습니다. 텍스트 범위를 줄이거나 민감정보를 더 제거해 다시 시도해 주세요.")
                        st.stop()

                    final_struct = ResearchPaperExtraction(
                        감사연도="",
                        피감기관="",
                        감사결과=all_items,
                    )

                    st.session_state["structured_json"] = final_struct

                    st.write("구조화된 JSON 데이터:")
                    with st.expander("구조화된 JSON 데이터:"):
                        st.json(final_struct.model_dump())

                except Exception as e:
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