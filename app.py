import streamlit as st
from pymongo import MongoClient
from pydantic import BaseModel
import google.generativeai as genai
from pdfminer.high_level import extract_text
import json

# Python 3.8+ 표준
try:
    from importlib.metadata import version, PackageNotFoundError
except ImportError:  # 아주 구버전 대비 (백포트)
    from importlib_metadata import version, PackageNotFoundError  # pip install importlib-metadata

try:
    st.write("google-generativeai version:", version("google-generativeai"))
except PackageNotFoundError:
    st.error("google-generativeai 미설치. requirements.txt에 'google-generativeai>=0.8.0' 추가 후 Reboot 해주세요.")

st.set_page_config(layout="wide", page_title="테스트")

# 🔐 secrets.toml에서 키/URI 불러오기 (하드코딩 금지)
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
client_db = MongoClient(st.secrets["MONGO_URI"])

db = client_db['json_db']
counter_collection = db['Yangsan_Audit']

# ✅ 최신 모델명으로 1회만 생성 (flash가 빠르고 무료 티어 넉넉)
model = genai.GenerativeModel(model_name="gemini-1.0-pro-latest")

def extract_text_from_doc(file):
    return extract_text(file)

class AuditResult(BaseModel):
    건명: str
    처분: str
    관련규정: str
    지적사항: str

class ResearchPaperExtraction(BaseModel):
    감사연도: str
    피감기관: str
    감사결과: list[AuditResult]

if "structured_json" not in st.session_state:
    st.session_state["structured_json"] = None

if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = None

st.title("금천구 감사결과 PDF 파일 파싱 서비스")

col1, col2 = st.columns(2)

with col1:
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if uploaded_file is not None:
        text = extract_text_from_doc(uploaded_file)
        st.session_state['extracted_text'] = text
        with st.expander("PDF에서 추출된 텍스트 확인하기"):
            st.write(st.session_state['extracted_text'])

with col2:
    if st.session_state.get('extracted_text'):
        st.subheader("RAG_Parse_PDF")
        with st.spinner('Structured Outputs...'):
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

            try:
                response = model.generate_content(prompt)
                raw = response.text or ""  # 방어적 처리

                # ```json ... ``` 방지용 트리밍
                s = raw.strip()
                if s.startswith("```json"):
                    s = s[len("```json"):].strip()
                if s.endswith("```"):
                    s = s[:-3].strip()

                data = json.loads(s)  # JSON 파싱
                st.session_state['structured_json'] = ResearchPaperExtraction(**data)

                with st.expander("구조화된 JSON 데이터:"):
                    st.json(st.session_state['structured_json'].dict(ensure_ascii=False))

            except json.JSONDecodeError as e:
                st.error(f"JSON 파싱 오류: {e}")
                st.caption("Raw Gemini response:")
                st.write(response.text if 'response' in locals() else "")
            except Exception as e:
                st.error(f"Gemini API 호출 또는 응답 처리 중 오류 발생: {e}")
                if 'response' in locals():
                    st.caption("Raw Gemini response:")
                    st.write(response.text)

        if st.button("MongoDB 저장"):
            with st.spinner('MongoDB Save...'):
                try:
                    if st.session_state['structured_json']:
                        counter_collection.insert_one(st.session_state['structured_json'].dict())
                        st.success("MongoDB에 데이터 저장 완료!")
                    else:
                        st.error("저장할 구조화된 JSON 데이터가 없습니다.")
                except Exception as e:
                    st.error(f"데이터 저장 중 오류 발생: {e}")
    else:
        st.markdown("""본 서비스는 문서 기반 RAG 시스템 개발을 지원하기 위해 설계되었습니다.

1) PDF에서 텍스트 추출  
2) AI로 구조화(JSON)  
3) MongoDB에 저장 및 검색
""")

st.markdown("---")

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
        result_list = list(counter_collection.find(query))

        if result_list:
            for idx, doc in enumerate(result_list, start=1):
                st.markdown(f"### 결과 {idx}")
                st.write(f"**감사연도:** {doc.get('감사연도')}")
                st.write(f"**피감기관:** {doc.get('피감기관')}")
                for audit in doc.get('감사결과', []):
                    blob = (audit.get('건명','') + audit.get('처분','') +
                            audit.get('관련규정','') + audit.get('지적사항',''))
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