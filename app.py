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
    분야: str | None = None   # 새 필드
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
        raw = line
        line = line.rstrip("\n")

        # 1) 완전한 구분선(테이블 테두리 등) 제거
        #    ─, │, ┃, ┏, ┓, ┗, ┛, =, - 등으로만 이루어진 줄
        if re.match(r"^[\s│┃┏┓┗┛━═\-_=]+$", line):
            continue

        stripped = line.strip()

        # 2) 완전히 비어 있는 줄은 건너뛰기
        if not stripped:
            continue

        # 3) 페이지 번호 형식 제거 (예: "- 15 -", "15 / 32" 등)
        if re.match(r"^[\-–—\s]*\d+\s*/\s*\d+[\-–—\s]*$", stripped):
            continue
        if re.match(r"^[\-–—\s]*\d+[\-–—\s]*$", stripped) and len(stripped) <= 8:
            # 짧은 페이지 번호 형태(예: "- 15 -", "15")만 제거
            continue

        # 4) 표 캡션 제거 (예: "표 1", "표 2-1", "Table 1" 등)
        if re.match(r"^표\s*\d+([\--–]\d+)?", stripped):
            continue
        if "table" in stripped.lower():
            continue

        # 5) 리스트 번호 같은 "1.", "2)", "3. 가)" 형태는 제거하되
        #    실제 제목/건명 줄은 절대 삭제하지 않기
        #
        #   - 예) "1." / "2)" / "3. 가)" 처럼 숫자+기호만 있고 내용이 거의 없는 경우만 제거
        #
        if re.match(r"^\d{1,2}\s*[.)]\s*$", stripped):
            # 내용 없는 순번만 있는 줄 (예: "1." / "2)")
            continue
        if re.match(r"^\d{1,2}\s*[.)]\s*[가-힣]\s*$", stripped):
            # 예: "1. 가" "2) 나" 같은 순번+한 글자만 있는 줄
            continue

        # ⛔ 여기서부터는 "5. 건강관리 분야", "15 ○○센터 비품관리대장…" 같은
        #    실제 제목/건명 줄은 그대로 유지됨

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
        st.text_area("추출된 텍스트", extracted_text[:800000], height=400)

# ----------- (2) AI 분석 -----------
with col2:
    if st.session_state.get("extracted_text"):
        cleaned_text = clean_text_for_ai(st.session_state["extracted_text"])

        if st.button("AI로 구조화(JSON) 변환"):
            with st.spinner("AI가 문서를 분석 중입니다..."):
                try:
                    completion = client.beta.chat.completions.parse(
    model="gpt-5-mini",
    messages=[
        {
            "role": "system",
            "content": (
                "You are an expert in Korean audit report parsing. "
                "You must convert unstructured text into structured JSON according to the schema."
            ),
        },
        {
            "role": "user",
            "content": (
                f"{cleaned_text}\n\n"
                "다음 조건을 지켜 감사결과를 JSON으로 구조화하세요:\n"
                "1) 상위 제목과 세부 제목을 구분하세요.\n"
                "   - '○○ 분야', '건강관리 분야', '예산·회계 분야'처럼 '분야'로 끝나는 것은 **분야**입니다.\n"
                "   - '15 ○○○○센터 비품관리대장 관리 소홀 [시정]'처럼 번호 + 제목 + [처분] 형태는\n"
                "     번호를 제외한 부분을 **건명**으로 사용합니다.\n"
                "2) JSON 필드는 다음과 같습니다.\n"
                "   - '분야': '예산·회계', '건강관리', '보건위생' 등 상위 분야 이름(예: '건강관리 분야' → '건강관리').\n"
                "   - '건명': 각 지적사항의 구체적인 제목\n"
                "       예) '특별휴가 사용 관리 소홀', '○○센터 비품관리대장 관리 소홀' 등.\n"
                "       '예산·회계 분야', '건강관리 분야'처럼 상위 제목은 건명에 절대 넣지 마세요.\n"
                "   - '처분': '시정', '주의', '통보', '시정/주의/통보' 등.\n"
                "   - '관련규정': 해당 지적사항 아래 '관련규정' 항목 전체 (요약 금지).\n"
                "   - '지적사항': 해당 지적사항 아래 '지적사항' 및 '조치할 사항' 내용을 자연스럽게 연결한 문단.\n"
                "3) JSON 전체 구조는 다음과 같습니다.\n"
                "{ '감사연도': str,\n"
                "  '피감기관': str,\n"
                "  '감사결과': [\n"
                "    { '분야': str, '건명': str, '처분': str, '관련규정': str, '지적사항': str }, ...\n"
                "  ]\n"
                "}\n"
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
    regex = re.compile(search_query, re.IGNORECASE)

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

    # 🔹 문서 안에서 다시 항목별 필터링
    total_matched = 0
    display_blocks = []

    for doc in results:
        matched_items = []
        for r in doc.get("감사결과", []):
            text_fields = [
                r.get("건명", ""),
                r.get("처분", ""),
                r.get("관련규정", ""),
                r.get("지적사항", ""),
            ]
            if any(regex.search(str(t)) for t in text_fields):
                matched_items.append(r)

        if matched_items:
            total_matched += len(matched_items)
            display_blocks.append((doc, matched_items))

    if total_matched > 0:
        st.success(f"총 {total_matched}건의 결과가 검색되었습니다.")
        for idx, (doc, items) in enumerate(display_blocks, start=1):
            st.markdown(f"### {idx}. {doc.get('피감기관')} ({doc.get('감사연도')})")
            for r in items:
                st.markdown(
                    f"**건명:** {r.get('건명')}  \n"
                    f"**처분:** {r.get('처분')}  \n"
                    f"**관련규정:** {r.get('관련규정')}  \n"
                    f"**지적사항:** {r.get('지적사항')}"
                )
                st.markdown("---")
    else:
        st.info("검색 결과가 없습니다.")