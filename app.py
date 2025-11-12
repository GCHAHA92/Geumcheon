import streamlit as st
from pymongo import MongoClient
from pydantic import BaseModel
import google.generativeai as genai 

from pdfminer.high_level import extract_text
import json

st.set_page_config(layout="wide", page_title="테스트")

# Gemini API 키 설정 (환경 변수 또는 직접 입력)
genai.configure(api_key="AIzaSyAbSFAR87Nbr1NvJJThnCIV9gnn0Fstzcs") # <<< 여기에 Gemini API 키를 입력하세요!

# 🔐 secrets.toml에서 키 불러오기
genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
client_db = MongoClient(st.secrets["MONGO_URI"])

db = client_db['json_db']
counter_collection = db['Yangsan_Audit']
model = genai.GenerativeModel('gemini-pro')

# Gemini 모델 초기화
model = genai.GenerativeModel('gemini-pro')

def extract_text_from_doc(file):
    text = extract_text(file)
    return text

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
        extracted_text = extract_text_from_doc(uploaded_file)
        st.session_state['extracted_text'] = extracted_text
        with st.expander("PDF에서 추출된 텍스트 확인하기"):
            st.write(st.session_state['extracted_text'])

with col2:
    if uploaded_file is not None:
        st.write("")
        st.subheader("RAG_Parse_PDF")
        with st.spinner('Structured Outputs...'):
            prompt = f"""You are an expert at structured data extraction. You will be given unstructured text from a research paper and should convert it into the given JSON structure.

            The output should be a JSON object that strictly adheres to the following Pydantic model:
            ```json
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
            ```

            Extract information from the following text based on '시정','주의','기타','회수(추징)','추급(환급)','징계','훈계(경고)' as disposition results. Do not summarize the related regulations; include all of them.

            Text to parse:
            {extracted_text}
            """
            
            try:
                response = model.generate_content(prompt)
                raw_json_string = response.text
                
                if raw_json_string.startswith("```json"):
                    raw_json_string = raw_json_string[len("```json"):].strip()
                if raw_json_string.endswith("```"):
                    raw_json_string = raw_json_string[:-len("```")].strip()

                structured_response_dict = json.loads(raw_json_string)
                st.session_state['structured_json'] = ResearchPaperExtraction(**structured_response_dict)

                st.write("구조화된 JSON 데이터:")
                with st.expander("구조화된 JSON 데이터:"):
                    st.json(st.session_state['structured_json'].dict())

            except json.JSONDecodeError as e:
                st.error(f"JSON 파싱 오류가 발생했습니다. 응답 내용을 확인하세요: {e}")
                st.write(f"Raw Gemini response: {response.text}")
            except Exception as e:
                st.error(f"Gemini API 호출 또는 응답 처리 중 오류 발생: {e}")
                if 'response' in locals():
                    st.write(f"Raw Gemini response: {response.text}")

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
        st.markdown("""본 서비스는 문서 기반 RAG(Retrieval-Augmented Generation) 시스템 개발을 지원하기 위해 설계되었습니다.

        1. **홈페이지에 공개된 감사결과 PDF 파일의 데이터를 추출합니다.**
        - 이 단계에서는 정해진 URL에서 PDF 문서를 다운로드하고 해당 문서에 포함된 텍스트 데이터를 추출합니다.
        - [금천구 감사결과 공개 사이트](https://geumcheon.go.kr/portal/selectBbsNttList.do?bbsNo=634&key=342)

        2. **추출된 데이터는 인공지능 서비스를 이용하여 파싱합니다.**
        - 텍스트 데이터는 자연어 처리(NLP) 기술을 활용하여 의미 있는 정보 단위로 파싱됩니다. 
        - 감사결과 행정상 주의, 시정, 기타 등 기준으로 감사결과를 문장 구조로 분석하여 JSON 형식으로 생성합니다.
                    
        3. **인공지능의 사전지식을 활용하여 데이터베이스에 저장합니다.**
        - 파싱된 정보는 추가적인 분석과 검색을 용이하게 하기 위해 구조화된 형식으로 데이터베이스에 저장됩니다.
        - 이는 RAG 시스템이 추후 신속하고 효율적인 정보 검색을 가능하게 합니다.
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

        results = counter_collection.find(query)

        result_list = list(results)
        if result_list:
            for idx, doc in enumerate(result_list, start=1):
                st.markdown(f"### 결과 {idx}")
                st.write(f"**감사연도:** {doc.get('감사연도')}")
                st.write(f"**피감기관:** {doc.get('피감기관')}")
                for audit in doc.get('감사결과', []):
                    if (search_query.lower() in audit.get('건명', '').lower() or
                        search_query.lower() in audit.get('처분', '').lower() or
                        search_query.lower() in audit.get('관련규정', '').lower() or
                        search_query.lower() in audit.get('지적사항', '').lower()):
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