import streamlit as st
from pymongo import MongoClient
from pydantic import BaseModel
from openai import OpenAI
from pdfminer.high_level import extract_text
import json
import io
import re

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(layout="wide", page_title="감사결과 PDF 파일 파싱 서비스")
st.title("감사결과 PDF 자동 구조화 시스템")

# -----------------------------
# 시크릿 로딩
# -----------------------------
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
MONGO_URI = st.secrets.get("MONGO_URI")

if not OPENAI_API_KEY or not MONGO_URI:
    st.error("OPENAI_API_KEY 또는 MONGO_URI가 설정되지 않았습니다. .streamlit/secrets.toml을 확인하세요.")
    st.stop()

# -----------------------------
# 클라이언트 설정
# -----------------------------
client = OpenAI(api_key=OPENAI_API_KEY)
mongo_client = MongoClient(MONGO_URI)
db = mongo_client["json_db"]
collection = db["Yangsan_Audit"]

# -----------------------------
# Pydantic 모델
# -----------------------------
class AuditResult(BaseModel):
    건명: str
    처분: str
    관련규정: str
    지적사항: str

class ResearchPaperExtraction(BaseModel):
    감사연도: str
    피감기관: str
    감사결과: list[AuditResult]

# -----------------------------
# PDF 텍스트 추출 함수
# -----------------------------
def extract_text_from_pdf(file):
    if hasattr(file, "read"):
        data = file.read()
        file.seek(0)
        return extract_text(io.BytesIO(data))
    return extract_text(file)

# -----------------------------
# 텍스트 정제 함수 (표·연번 제거)
# -----------------------------
def clean_text_for_ai(text: str) -> str:
    lines = text.splitlines()
    cleaned = []

    for line in lines:
        # 표 구조나 구분선 제거
        if re.search(r"[│┃┏┓┗┛━═\-]{3,}", line):  # 긴 구분선
            continue
        if re.search(r"^\s*\d{1,2}\s*[.|)]", line):  # 연번 (1. / 2) / 3)
            continue
        if "표 " in line or "표-" in line or "table" in line.lower():
            continue
        if len(line.strip()) == 0:
            continue

        # 금액이나 총건수는 유지 (예: 27,000원 / 총 14건)
        cleaned.append(line)

    return "\n".join(cleaned)

# -----------------------------
# 세션 상태
# -----------------------------
if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = None
if "structured_json" not in st.session_state:
    st.session_state["structured_json"] = None

# -----------------------------
# 레이아웃
# -----------------------------
col1, col2 = st.columns(2)

# ----------- (1) 파일 업로드 -----------
with col1:
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if uploaded_file:
        extracted_text = extract_text_from_pdf(uploaded_file)
        st.session_state["extracted_text"] = extracted_text

        st.subheader("📄 PDF 원문 미리보기")
        st.text_area("추출된 텍스트", extracted_text[:8000], height=400)

# ----------- (2) AI 분석 -----------
with col2:
    if st.session_state.get("extracted_text"):
        cleaned_text = clean_text_for_ai(st.session_state["extracted_text"])

        if st.button("AI로 구조화(JSON) 변환"):
            with st.spinner("AI가 문서를 분석 중입니다..."):
                try:
                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert in audit report parsing. "
                                    "You must convert unstructured text into structured JSON according to the schema."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"{cleaned_text}\n\n"
                                    "다음 조건을 지켜 감사결과를 JSON으로 구조화하세요:\n"
                                    "- '시정','주의','기타','회수(환수)','추급(환급)','징계','훈계(경징계/중징계)' 처분결과를 모두 포함합니다.\n"
                                    "- 표, 연번, 목록형 데이터(1. 2. 3. …)는 제거합니다.\n"
                                    "- 금액(예: 27,000원), 총 건수(예: 총 14건)는 유지합니다.\n"
                                    "- 관련규정은 요약하지 말고 법령 원문 전체를 그대로 포함합니다.\n"
                                    "- 조치할 사항은 반드시 포함합니다.\n"
                                    "- JSON 형식은 다음과 같습니다:\n"
                                    "{ '감사연도': str, '피감기관': str, '감사결과': [ {'건명': str, '처분': str, '관련규정': str, '지적사항': str} ] }"
                                ),
                            },
                        ],
                        response_format=ResearchPaperExtraction,
                        temperature=0,
                    )

                    structured = completion.choices[0].message.parsed
                    st.session_state["structured_json"] = structured

                    st.success("✅ AI 구조화 완료!")
                    st.json(structured.model_dump())

                except Exception as e:
                    st.error(f"AI 처리 중 오류 발생: {e}")

        if st.session_state.get("structured_json"):
            if st.button("MongoDB 저장"):
                doc = st.session_state["structured_json"].model_dump()
                collection.insert_one(doc)
                st.success("✅ MongoDB에 저장 완료!")

# ----------- (3) 검색 -----------
st.markdown("---")
st.subheader("MongoDB 검색")

search_query = st.text_input("검색어를 입력하세요:")
if search_query:
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

    results = list(collection.find(query))
    if results:
        st.success(f"총 {len(results)}건의 결과가 검색되었습니다.")
        for idx, doc in enumerate(results, start=1):
            st.markdown(f"### {idx}. {doc.get('피감기관')} ({doc.get('감사연도')})")
            for r in doc.get("감사결과", []):
                st.markdown(f"**건명:** {r.get('건명')}  \n**처분:** {r.get('처분')}  \n**관련규정:** {r.get('관련규정')}  \n**지적사항:** {r.get('지적사항')}")
                st.markdown("---")
    else:
        st.info("검색 결과가 없습니다.")
