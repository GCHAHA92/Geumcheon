import streamlit as st
from pymongo import MongoClient
from pydantic import BaseModel
from openai import OpenAI
from pdfminer.high_level import extract_text
import json
import io
import traceback

# -----------------------------
# 기본 설정
# -----------------------------
st.set_page_config(layout="wide", page_title="양산시 인공지능 자동화 서비스")
st.title("양산시 감사결과 PDF 파일 파싱 서비스")

# -----------------------------
# 시크릿 로딩 (필수 값 점검)
# -----------------------------
def load_secrets():
    openai_key = st.secrets.get("OPENAI_API_KEY")
    mongo_uri = st.secrets.get("MONGO_URI")
    missing = []
    if not openai_key:
        missing.append("OPENAI_API_KEY")
    if not mongo_uri:
        missing.append("MONGO_URI")
    return openai_key, mongo_uri, missing

OPENAI_API_KEY, MONGO_URI, MISSING = load_secrets()

if MISSING:
    st.error(
        "다음 시크릿 항목이 누락되어 있어요: " + ", ".join(MISSING) +
        "\n\n`.streamlit/secrets.toml` 파일을 확인해 주세요."
    )
    st.stop()

# -----------------------------
# 외부 클라이언트 초기화
# -----------------------------
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    st.error(f"OpenAI 클라이언트 초기화 실패: {e}")
    st.stop()

try:
    client_db = MongoClient(MONGO_URI)
    db = client_db["json_db"]
    counter_collection = db["Yangsan_Audit"]
except Exception as e:
    st.error(f"MongoDB 연결 실패: {e}")
    st.stop()

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
# 세션 상태
# -----------------------------
if "structured_json" not in st.session_state:
    st.session_state["structured_json"] = None

if "extracted_text" not in st.session_state:
    st.session_state["extracted_text"] = None

# -----------------------------
# 유틸: Pydantic → dict (v1/v2 호환)
# -----------------------------
def pydantic_to_dict(model_obj):
    try:
        if hasattr(model_obj, "model_dump"):
            return model_obj.model_dump()
        if hasattr(model_obj, "dict"):
            return model_obj.dict()
        return json.loads(model_obj.json())
    except Exception:
        # 어떻게든 직렬화
        return json.loads(json.dumps(model_obj, default=lambda o: getattr(o, "__dict__", str(o))))

# -----------------------------
# PDF 텍스트 추출
# -----------------------------
def extract_text_from_doc(file_like):
    # Streamlit은 UploadedFile(바이너리) → pdfminer는 file-like 지원
    # 업로드 객체를 BytesIO로 감싸 안정적 처리
    if hasattr(file_like, "read"):
        data = file_like.read()
        file_like.seek(0)
        bio = io.BytesIO(data)
        return extract_text(bio)
    # 이미 경로나 바이너리라면 그대로 시도
    return extract_text(file_like)

# -----------------------------
# 레이아웃
# -----------------------------
col1, col2 = st.columns(2)

# ----------- 업로드 & 미리보기 -----------
with col1:
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")

    if uploaded_file is not None:
        try:
            extracted_text = extract_text_from_doc(uploaded_file)
            st.session_state["extracted_text"] = extracted_text
            with st.expander("PDF에서 추출된 텍스트 확인하기", expanded=False):
                st.write(st.session_state["extracted_text"][:5000] + ("..." if len(st.session_state["extracted_text"]) > 5000 else ""))
        except Exception as e:
            st.error(f"PDF 텍스트 추출 중 오류: {e}")
            st.caption(traceback.format_exc())

# ----------- 파싱 & Mongo 저장 -----------
with col2:
    if uploaded_file is not None and st.session_state.get("extracted_text"):
        st.subheader("RAG_Parse_PDF")
        if st.button("AI로 구조화 파싱 실행"):
            with st.spinner("Structured Outputs..."):
                try:
                    # OpenAI Structured Outputs (Pydantic) 사용
                    completion = client.beta.chat.completions.parse(
                        model="gpt-4o-mini",
                        messages=[
                            {
                                "role": "system",
                                "content": (
                                    "You are an expert at structured data extraction. "
                                    "You will be given unstructured text from an audit report "
                                    "and should convert it into the given structure."
                                ),
                            },
                            {
                                "role": "user",
                                "content": (
                                    f"{st.session_state['extracted_text']}\n\n"
                                    "내용 중 '시정','주의','기타','회수(추징)','추급(환급)','징계','훈계(경고)' 처분결과 기준으로 "
                                    "자료를 모두 만들고, 관련규정은 요약하지 말고 모두 입력해 주세요."
                                ),
                            },
                        ],
                        response_format=ResearchPaperExtraction,
                        temperature=0,
                    )

                    structured_response = completion.choices[0].message.parsed
                    st.session_state["structured_json"] = structured_response

                    with st.expander("구조화된 JSON 데이터 보기", expanded=True):
                        st.json(pydantic_to_dict(structured_response))

                except Exception as e:
                    st.error(f"파싱 중 오류: {e}")
                    st.caption(traceback.format_exc())

        if st.session_state.get("structured_json") is not None:
            if st.button("MongoDB 저장"):
                with st.spinner("MongoDB Save..."):
                    try:
                        doc = pydantic_to_dict(st.session_state["structured_json"])
                        # 기본 보정: 필수 키 유효성 체크
                        for k in ["감사연도", "피감기관", "감사결과"]:
                            if k not in doc:
                                raise ValueError(f"'{k}' 필드가 없습니다.")
                        counter_collection.insert_one(doc)
                        st.success("MongoDB에 데이터 저장 완료!")
                    except Exception as e:
                        st.error(f"데이터 저장 중 오류 발생: {e}")
                        st.caption(traceback.format_exc())

    else:
        st.markdown(
            """
**본 서비스는 문서 기반 RAG(Retrieval-Augmented Generation) 시스템 개발을 지원하기 위해 설계되었습니다.**

1. **공개된 감사결과 PDF의 텍스트를 추출합니다.**  
   - 업로드한 PDF에서 텍스트를 안정적으로 추출합니다.  
   - 참고: [양산시 감사결과 공개 사이트](https://www.yangsan.go.kr/portal/board/post/list.do?bcIdx=13&mid=0407020000)

2. **추출된 텍스트를 인공지능으로 구조화(JSON)합니다.**  
   - 감사결과의 ‘시정, 주의, 기타, 회수(추징), 추급(환급), 징계, 훈계(경고)’ 등 처분 기준으로 파싱합니다.  
   - 관련 규정은 **요약 없이 원문 전체**를 담습니다.

3. **구조화된 결과를 MongoDB에 저장하고 검색합니다.**
            """
        )

st.markdown("---")

# -----------------------------
# 검색 UI (MongoDB)
# -----------------------------
st.subheader("MongoDB 검색")
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
                        # 사용하신 스키마에 '규정주요내용'이 없다면 자동으로 건너뜁니다.
                        {"규정주요내용": {"$regex": search_query, "$options": "i"}},
                    ]
                }
            }
        }

        results = counter_collection.find(query).limit(100)
        result_list = list(results)

        if result_list:
            for idx, doc in enumerate(result_list, start=1):
                st.markdown(f"### 결과 {idx}")
                st.write(f"**감사연도:** {doc.get('감사연도', '')}")
                st.write(f"**피감기관:** {doc.get('피감기관', '')}")
                audits = doc.get("감사결과", [])
                if not audits:
                    st.caption("감사결과 항목이 없습니다.")
                else:
                    for audit in audits:
                        text_bucket = " ".join([
                            str(audit.get("건명", "")),
                            str(audit.get("처분", "")),
                            str(audit.get("관련규정", "")),
                            str(audit.get("지적사항", "")),
                            str(audit.get("규정주요내용", "")),
                        ]).lower()
                        if search_query.lower() in text_bucket:
                            st.write(f"- **건명:** {audit.get('건명', '')}")
                            st.write(f"  **처분:** {audit.get('처분', '')}")
                            st.write(f"  **관련규정:** {audit.get('관련규정', '')}")
                            st.write(f"  **지적사항:** {audit.get('지적사항', '')}")
                            st.markdown("---")
        else:
            st.info("검색 결과가 없습니다.")
    except Exception as e:
        st.error(f"검색 중 오류 발생: {e}")
        st.caption(traceback.format_exc())
else:
    st.caption("검색어를 입력하면 MongoDB에서 관련 결과를 찾아 보여드립니다.")
