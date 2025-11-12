import streamlit as st
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List
import google.generativeai as genai
from pdfminer.high_level import extract_text
import json

st.set_page_config(layout="wide", page_title="금천구 감사결과 PDF 파싱 서비스")

# 🔐 secrets.toml에서 키/URI 불러오기
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
client_db = MongoClient(st.secrets["MONGO_URI"])

db = client_db["json_db"]
counter_collection = db["Yangsan_Audit"]

# ✅ 최신 모델로 교체 (flash=빠름/저비용, pro=정밀)
MODEL_NAME = "gemini-1.5-flash"
model = genai.GenerativeModel(model_name=MODEL_NAME)

def extract_text_from_doc(file):
    return extract_text(file)

# ---- Pydantic 스키마 ----
class AuditResult(BaseModel):
    건명: str
    처분: str
    관련규정: str
    지적사항: str

class ResearchPaperExtraction(BaseModel):
    감사연도: str
    피감기관: str
    감사결과: List[AuditResult]

# ---- 세션 상태 ----
if "structured_json" not in st.session_state:
    st.session_state["structured_json"] = None
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = None

st.title("금천구 감사결과 PDF 파일 파싱 서비스")

col1, col2 = st.columns(2)

# -------- 좌측: 업로드 --------
with col1:
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_doc(uploaded_file)
        st.session_state["extracted_text"] = text
        with st.expander("PDF에서 추출된 텍스트 확인하기"):
            st.write(st.session_state["extracted_text"])

# -------- 우측: 구조화/저장 --------
with col2:
    if st.session_state.get("extracted_text"):
        st.subheader("RAG_Parse_PDF")

        # OpenAI의 response_format과 유사: schema + JSON 강제
        system_msg = (
            "You are an expert at structured data extraction. "
            "You will be given unstructured text from a research paper and should convert it into the given structure."
        )
        user_msg = (
            f"{st.session_state['extracted_text']} 내용 중 '시정','주의','기타','회수(추징)',"
            "'추급(환급)','징계','훈계(경고)' 등 처분결과 기준으로 자료를 모두 만들고, "
            "관련규정은 요약하지 말고 모두 입력해 주세요."
        )

        if st.button("AI로 구조화 분석하기"):
            with st.spinner("Structured Outputs..."):
                try:
                    resp = model.generate_content(
                        [system_msg, user_msg],
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=ResearchPaperExtraction,  # ← Pydantic 스키마 그대로!
                            temperature=0,
                        ),
                    )
                    # Gemini는 위 설정이면 JSON만 반환
                    structured = ResearchPaperExtraction.model_validate_json(resp.text)
                    st.session_state["structured_json"] = structured

                    st.write("구조화된 JSON 데이터:")
                    with st.expander("구조화된 JSON 데이터:"):
                        st.json(structured.model_dump())

                except Exception as e:
                    st.error(f"Gemini API 호출 또는 응답 처리 중 오류 발생: {e}")
                    if "resp" in locals():
                        st.caption("Raw Gemini response:")
                        st.write(getattr(resp, "text", ""))

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

# -------- 검색 --------
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