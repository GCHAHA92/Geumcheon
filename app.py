import streamlit as st
from pymongo import MongoClient
from pydantic import BaseModel
import google.generativeai as genai
from pdfminer.high_level import extract_text
import json
import re
from typing import List

# ---------- Version check (optional) ----------
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # pragma: no cover
    from importlib_metadata import version, PackageNotFoundError

try:
    st.caption(f"google-generativeai: {version('google-generativeai')}")
except PackageNotFoundError:
    st.error("google-generativeai 미설치. requirements.txt에 'google-generativeai>=0.8.0' 추가 후 Reboot 해주세요.")

st.set_page_config(layout="wide", page_title="금천구 감사결과 PDF 파싱")
st.title("금천구 감사결과 PDF 파일 파싱 서비스")

# ---------- Secrets / Clients ----------
@st.cache_resource
def get_mongo_client() -> MongoClient:
    return MongoClient(st.secrets["MONGO_URI"])  # secrets.toml 또는 Cloud Edit secrets 필수

try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])  # ✅ 절대 하드코딩 금지
except Exception as e:
    st.stop()

client_db = get_mongo_client()
db = client_db["json_db"]
collection = db["Yangsan_Audit"]

# ✅ 최신 v1 API 모델 (구버전 금지: gemini-pro / 1.0-pro-latest 등)
MODEL_NAME = "gemini-1.5-flash"  # 빠르고 무료티어 넉넉 / 정밀은 "gemini-1.5-pro"
st.caption(f"Using model: {MODEL_NAME}")
model = genai.GenerativeModel(model_name=MODEL_NAME)

# ---------- Pydantic schema ----------
class AuditResult(BaseModel):
    건명: str
    처분: str
    관련규정: str
    지적사항: str

class ResearchPaperExtraction(BaseModel):
    감사연도: str
    피감기관: str
    감사결과: List[AuditResult]

# ---------- Helpers ----------
def extract_text_from_pdf(file) -> str:
    return extract_text(file)

def coerce_json_from_text(raw: str) -> str:
    """Clean markdown fences and try to extract the largest JSON object."""
    s = raw.strip()
    # remove code fences
    s = re.sub(r"^```(?:json)?", "", s).strip()
    s = re.sub(r"```$", "", s).strip()
    # try biggest {...} block
    start = s.find("{")
    end = s.rfind("}")
    if start != -1 and end != -1 and end > start:
        return s[start : end + 1]
    return s

# ---------- Session state ----------
if "structured_json" not in st.session_state:
    st.session_state["structured_json"] = None
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = None

# ---------- UI: Left (Upload) ----------
col1, col2 = st.columns(2)
with col1:
    uploaded = st.file_uploader("PDF 파일을 업로드하세요", type=["pdf"])
    if uploaded:
        with st.spinner("PDF 텍스트 추출 중..."):
            text = extract_text_from_pdf(uploaded)
        st.session_state["extracted_text"] = text
        with st.expander("PDF에서 추출된 텍스트"):
            st.text_area("원문", text[:8000], height=240)

# ---------- UI: Right (AI parse + Save) ----------
with col2:
    if st.session_state.get("extracted_text"):
        st.subheader("RAG_Parse_PDF")
        prompt = f"""
You are an expert at structured data extraction. Convert the following text into this JSON schema:

{{
  "감사연도": "string",
  "피감기관": "string",
  "감사결과": [
    {{
      "건명": "string",
      "처분": "string",
      "관련규정": "string",
      "지적사항": "string"
    }}
  ]
}}

Use '시정','주의','기타','회수(추징)','추급(환급)','징계','훈계(경고)' as disposition categories.
Do not summarize '관련규정' — include all as-is.

TEXT:
{st.session_state['extracted_text']}
"""
        if st.button("AI로 구조화 분석하기", use_container_width=True):
            with st.spinner("Gemini 분석 중..."):
                try:
                    response = model.generate_content(prompt)
                    raw = (response.text or "").strip()
                    cleaned = coerce_json_from_text(raw)
                    data = json.loads(cleaned)
                    st.session_state["structured_json"] = ResearchPaperExtraction(**data)
                    with st.expander("구조화된 JSON 데이터"):
                        st.json(st.session_state["structured_json"].dict(ensure_ascii=False))
                except json.JSONDecodeError as e:
                    st.error(f"JSON 파싱 오류: {e}")
                    st.caption("Raw Gemini response:")
                    st.write(response.text if 'response' in locals() else "")
                except Exception as e:
                    st.error(f"Gemini API 호출 또는 응답 처리 중 오류: {e}")
                    if 'response' in locals():
                        st.caption("Raw Gemini response:")
                        st.write(response.text)

        if st.session_state.get("structured_json") and st.button("MongoDB 저장", use_container_width=True):
            with st.spinner("MongoDB 저장 중..."):
                try:
                    collection.insert_one(st.session_state["structured_json"].dict())
                    st.success("MongoDB에 데이터 저장 완료!")
                except Exception as e:
                    st.error(f"데이터 저장 중 오류: {e}")
    else:
        st.info("PDF를 업로드하면 구조화 버튼이 활성화됩니다.")

st.markdown("---")

# ---------- 검색 ----------
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
                        {"지적사항": {"$regex": search_query, "$options": "i"}}
                    ]
                }
            }
        }
        results = list(collection.find(query))
        if results:
            for i, doc in enumerate(results,  start=1):
                st.markdown(f"### 결과 {i}")
                st.write(f"**감사연도:** {doc.get('감사연도','')}")
                st.write(f"**피감기관:** {doc.get('피감기관','')}")
                for audit in doc.get("감사결과", []):
                    st.write(f"**건명:** {audit.get('건명','')}")
                    st.write(f"**처분:** {audit.get('처분','')}")
                    st.write(f"**관련규정:** {audit.get('관련규정','')}")
                    st.write(f"**지적사항:** {audit.get('지적사항','')}")
                    st.markdown("---")
        else:
            st.info("검색 결과가 없습니다.")
    except Exception as e:
        st.error(f"검색 중 오류: {e}")
else:
    st.warning("검색어를 입력해주세요.")