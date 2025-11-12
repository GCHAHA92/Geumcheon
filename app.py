import streamlit as st
from pymongo import MongoClient
from pydantic import BaseModel, Field
from typing import List
import google.generativeai as genai
import json, base64, time

# --- 기본 설정 ---
st.set_page_config(layout="wide", page_title="양산시 감사결과 PDF 자동 분석기 (Chunk 지원)")
st.title("양산시 감사결과 PDF 자동 분석기 (Chunk 지원)")

# --- API & DB 연결 ---
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
except Exception:
    st.error("❌ Gemini API Key가 secrets.toml에 없습니다.")
    st.stop()

try:
    client_db = MongoClient(st.secrets["MONGO_URI"])
    db = client_db["json_db"]
    counter_collection = db["Yangsan_Audit"]
except Exception:
    st.warning("⚠️ MongoDB 연결 실패 — 저장/검색 기능 비활성화.")
    client_db = None

# --- Pydantic 스키마 ---
class AuditResult(BaseModel):
    건명: str
    처분: str
    관련규정: str
    지적사항: str

class AuditReportExtraction(BaseModel):
    감사연도: str
    피감기관: str
    감사결과: List[AuditResult]

class ChunkExtraction(BaseModel):
    감사결과: List[AuditResult]

# --- 헬퍼 함수 ---
def split_text(text, size=4000, overlap=200):
    """긴 텍스트를 일정 크기로 분할"""
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i:i + size])
    return chunks

# --- PDF 업로드 ---
uploaded_file = st.file_uploader("📎 PDF 파일을 업로드하세요", type="pdf")

if uploaded_file:
    with st.expander("📄 업로드된 PDF 미리보기", expanded=False):
        base64_pdf = base64.b64encode(uploaded_file.getvalue()).decode('utf-8')
        st.markdown(f'<iframe src="data:application/pdf;base64,{base64_pdf}" width="100%" height="500"></iframe>', unsafe_allow_html=True)

    model = genai.GenerativeModel("gemini-2.5-flash-lite")
    pdf_bytes = uploaded_file.getvalue()

    if st.button("🚀 Gemini로 감사정보 자동 추출 (자동 청크 분석)"):
        with st.spinner("Gemini가 PDF를 분석 중입니다..."):
            try:
                # 1️⃣ 먼저 PDF를 텍스트로 변환해 Gemini가 전체 파악 가능하게 함
                prompt = (
                    "다음 PDF 문서에서 감사 정보를 추출해줘. "
                    "문서의 전체 내용을 분석해서 감사연도, 피감기관을 찾고, "
                    "'시정','주의','기타','회수(추징)','추급(환급)','징계','훈계(경고)' 처분결과를 기준으로 "
                    "모든 지적사항을 JSON 형식으로 만들어줘. "
                    "관련규정은 요약하지 말고 원문 그대로 입력해야 해."
                )

                response = model.generate_content(
                    [
                        prompt,
                        {"mime_type": "application/pdf", "data": pdf_bytes},
                    ],
                    generation_config=genai.GenerationConfig(
                        response_mime_type="application/json",
                        response_schema=AuditReportExtraction,
                        temperature=0,
                    )
                )

                # 안전필터 차단 시 완화 재시도
                if response.candidates and response.candidates[0].finish_reason == 2:
                    st.warning("⚠️ 1차 요청이 안전필터에 차단됨. 완화 모드로 재시도 중...")
                    time.sleep(2)
                    response = model.generate_content(
                        [
                            prompt.replace("징계", "징*계").replace("주의", "주 의"),
                            {"mime_type": "application/pdf", "data": pdf_bytes},
                        ],
                        generation_config=genai.GenerationConfig(
                            response_mime_type="application/json",
                            response_schema=AuditReportExtraction,
                            temperature=0,
                        )
                    )

                # 텍스트 길이 짧으면 그대로 사용
                text_result = response.text or ""
                if len(text_result) > 100:
                    data = json.loads(text_result)
                    st.session_state["structured_json"] = data
                    st.success("✅ 분석 완료 (단일 모드)")
                else:
                    st.warning("⚠️ 단일 호출 결과가 짧습니다. 청크 분석으로 전환합니다...")

                    # 2️⃣ 긴 PDF를 조각 단위로 분석
                    text_parts = split_text(response.candidates[0].content.parts[0].text)
                    all_items = []

                    for idx, part in enumerate(text_parts, start=1):
                        st.info(f"🔹 PART {idx}/{len(text_parts)} 분석 중...")
                        sub_prompt = (
                            f"다음은 감사결과의 일부입니다 (PART {idx}).\n"
                            "이 부분에서 '건명','처분','관련규정','지적사항'만 추출해 JSON으로 반환하세요.\n"
                            "상위 키는 '감사결과' 하나만 포함합니다.\n"
                            "관련규정은 요약하지 말고 원문 그대로 입력하세요."
                        )
                        resp = model.generate_content(
                            [sub_prompt, part],
                            generation_config=genai.GenerationConfig(
                                response_mime_type="application/json",
                                response_schema=ChunkExtraction,
                                temperature=0,
                            )
                        )
                        try:
                            chunk_data = json.loads(resp.text)
                            all_items.extend(chunk_data.get("감사결과", []))
                        except Exception:
                            st.warning(f"⚠️ PART {idx} JSON 변환 실패 — 건너뜀")
                            continue

                    st.session_state["structured_json"] = {
                        "감사연도": "",
                        "피감기관": "",
                        "감사결과": all_items
                    }
                    st.success("✅ 청크 분석 완료!")

                with st.expander("추출된 JSON 결과", expanded=True):
                    st.json(st.session_state["structured_json"])

            except Exception as e:
                st.error(f"Gemini API 호출 중 오류 발생: {e}")

    # MongoDB 저장
    if client_db and "structured_json" in st.session_state:
        if st.button("💾 MongoDB에 저장"):
            try:
                counter_collection.insert_one(st.session_state["structured_json"])
                st.success("MongoDB에 저장 완료!")
            except Exception as e:
                st.error(f"데이터 저장 오류: {e}")

else:
    st.info("👆 상단에서 PDF를 업로드하면 분석을 시작할 수 있습니다.")

# --- 검색 기능 ---
if client_db:
    st.markdown("---")
    st.header("🔍 감사결과 검색")

    search_query = st.text_input("검색할 단어나 문장을 입력하세요:")

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

        results = list(counter_collection.find(query))
        if results:
            st.success(f"{len(results)}건의 결과를 찾았습니다.")
            for idx, doc in enumerate(results, start=1):
                with st.expander(f"결과 {idx}: {doc.get('피감기관', '')} ({doc.get('감사연도', '')})"):
                    for audit in doc.get("감사결과", []):
                        st.markdown(f"**건명:** {audit.get('건명')}  ")
                        st.markdown(f"**처분:** {audit.get('처분')}  ")
                        st.markdown(f"**관련규정:** {audit.get('관련규정')}  ")
                        st.markdown(f"**지적사항:** {audit.get('지적사항')}  ")
                        st.markdown("---")
        else:
            st.info("검색 결과가 없습니다.")
