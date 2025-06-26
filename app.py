"""
MultiModal RAG 시스템 Streamlit 프론트엔드
"""
import streamlit as st
import requests
import json
from typing import Dict, Any, List
import time
import uuid
import hashlib

from config import Config

# 페이지 설정
st.set_page_config(
    page_title="MultiModal RAG 시스템",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS 스타일링
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 1rem 0;
    }
    
    .user-message {
        background-color: #E3F2FD;
        border-left: 4px solid #1E88E5;
    }
    
    .bot-message {
        background-color: #F5F5F5;
        border-left: 4px solid #4CAF50;
    }
    
    .source-box {
        background-color: #FFF3E0;
        border: 1px solid #FFB74D;
        border-radius: 0.25rem;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    
    .confidence-high {
        color: #4CAF50;
        font-weight: bold;
    }
    
    .confidence-medium {
        color: #FF9800;
        font-weight: bold;
    }
    
    .confidence-low {
        color: #F44336;
        font-weight: bold;
    }
    
    .query-analysis {
        background-color: #e3f2fd;
        padding: 12px;
        border-radius: 8px;
        border-left: 4px solid #2196f3;
        margin: 10px 0;
        font-style: italic;
    }
</style>
""", unsafe_allow_html=True)

class RAGClient:
    """FastAPI 백엔드와 통신하는 클라이언트"""
    
    def __init__(self):
        self.base_url = f"http://{Config.FASTAPI_HOST}:{Config.FASTAPI_PORT}"
    
    def check_health(self) -> bool:
        """백엔드 서버 상태 확인"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            return response.status_code == 200
        except:
            return False
    
    def query(self, question: str, top_k: int = 5) -> Dict[str, Any]:
        """질의응답 요청"""
        try:
            response = requests.post(
                f"{self.base_url}/query",
                json={"query": question, "top_k": top_k},
                timeout=60
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"서버 오류: {response.status_code}"}
        except Exception as e:
            return {"error": f"연결 오류: {str(e)}"}
    
    def search(self, query: str, top_k: int = 10) -> Dict[str, Any]:
        """문서 검색 요청"""
        try:
            response = requests.post(
                f"{self.base_url}/search",
                json={"query": query, "top_k": top_k},
                timeout=30
            )
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"서버 오류: {response.status_code}"}
        except Exception as e:
            return {"error": f"연결 오류: {str(e)}"}
    
    def get_stats(self) -> Dict[str, Any]:
        """시스템 통계 조회"""
        try:
            response = requests.get(f"{self.base_url}/stats", timeout=10)
            if response.status_code == 200:
                return response.json()
            else:
                return {"error": f"서버 오류: {response.status_code}"}
        except Exception as e:
            return {"error": f"연결 오류: {str(e)}"}

def _generate_unique_button_key(message_idx: int, question_idx: int, question: str) -> str:
    """
    매우 고유한 버튼 키 생성
    - 시간 타임스탬프
    - UUID
    - 메시지 수와 질문 내용을 결합한 강력한 해시
    """
    # 현재 시간 타임스탬프 (마이크로초 포함)
    timestamp = int(time.time() * 1000000)  # 마이크로초 단위
    
    # 짧은 UUID 생성 (8자리)
    short_uuid = str(uuid.uuid4())[:8]
    
    # 강력한 해시 생성 (SHA256 사용)
    hash_input = f"{message_idx}_{question_idx}_{question}_{timestamp}_{short_uuid}"
    question_hash = hashlib.sha256(hash_input.encode('utf-8')).hexdigest()[:12]
    
    # 최종 고유 키 조합
    unique_key = f"related_q_{message_idx}_{question_idx}_{timestamp}_{short_uuid}_{question_hash}"
    
    return unique_key

def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "rag_client" not in st.session_state:
        st.session_state.rag_client = RAGClient()

def display_message(role: str, content: str, sources: List[Dict] = None, confidence: float = None, 
                   query_analysis: str = None, related_questions: List[str] = None):
    """메시지 표시"""
    if role == "user":
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>👤 사용자:</strong><br>
            {content}
        </div>
        """, unsafe_allow_html=True)
    else:
        # 답변 내용 검증 및 개선
        if not content or content.strip() == "":
            content = "답변을 생성하지 못했습니다. 다른 질문을 시도해보세요."
        
        # 답변이 너무 짧거나 오류 메시지인 경우 처리
        if len(content.strip()) < 20 and "오류" not in content and "죄송" not in content:
            if sources and len(sources) > 0:
                content = "검색된 문서를 참고하여 질문에 대한 정보를 확인해보세요. 구체적인 답변을 위해 더 명확한 질문을 해주시기 바랍니다."
            else:
                content = "관련 문서를 찾지 못했습니다. 다른 키워드로 질문을 다시 해보세요."
        
        # f-string에서 백슬래시 사용을 피하기 위해 사전에 처리
        formatted_content = content.replace('\n', '<br>')
        st.markdown(f"""
        <div class="chat-message bot-message">
            <strong>🤖 AI 어시스턴트:</strong><br>
            {formatted_content}
        </div>
        """, unsafe_allow_html=True)
        
        # 질문 분석 표시
        if query_analysis:
            st.markdown(f"""
            <div class="query-analysis">
                <strong>🎯 질문 분석:</strong><br>
                {query_analysis}
            </div>
            """, unsafe_allow_html=True)
        
        # 신뢰도 표시 개선
        if confidence is not None and confidence > 0:
            confidence_class = (
                "confidence-high" if confidence > 0.8 
                else "confidence-medium" if confidence > 0.5 
                else "confidence-low"
            )
            confidence_text = "높음" if confidence > 0.8 else "보통" if confidence > 0.5 else "낮음"
            st.markdown(f"""
            <p class="{confidence_class}">신뢰도: {confidence:.1%} ({confidence_text})</p>
            """, unsafe_allow_html=True)
        
        # 소스 정보 표시
        if sources and len(sources) > 0:
            with st.expander(f"📚 참고 문서 ({len(sources)}개)", expanded=False):
                for i, source in enumerate(sources):
                    rerank_score = source.get('rerank_score', source.get('score', 0))
                    vector_score = source.get('vector_score', source.get('score', 0))
                    
                    # 메타데이터에서 상세 정보 추출
                    metadata = source.get('metadata', {})
                    file_name = metadata.get('file_name', '알 수 없는 문서')
                    doc_type = metadata.get('doc_type', 'Unknown')
                    section_type = metadata.get('section_type', '일반')
                    
                    # 이미지 타입인 경우 추가 정보
                    extra_info = ""
                    if doc_type == "image":
                        image_type = metadata.get('image_type', '')
                        image_filename = metadata.get('image_filename', '')
                        if image_type and image_filename:
                            extra_info = f" | {image_type} 이미지: {image_filename}"
                    
                    # 점수 표시 개선
                    rerank_display = f"{rerank_score:.3f}" if rerank_score >= 0 else "N/A"
                    vector_display = f"{vector_score:.3f}" if vector_score >= 0 else "N/A"
                    
                    st.markdown(f"""
                    <div class="source-box">
                        <strong>📄 문서 {i+1}:</strong> {file_name}<br>
                        <strong>📝 타입:</strong> {doc_type} 문서 ({section_type} 섹션){extra_info}<br>
                        <strong>📊 점수:</strong> 리랭킹 {rerank_display} | 벡터 유사도 {vector_display}<br>
                        <strong>📋 내용:</strong> {source['content']}
                    </div>
                    """, unsafe_allow_html=True)
        
        # 관련 질문들 표시
        if related_questions and len(related_questions) > 0:
            with st.expander("🤔 관련 질문들", expanded=False):
                st.markdown("**이런 질문들도 해보세요:**")
                # 메시지 인덱스를 포함한 고유 키 생성
                message_idx = len(st.session_state.messages)
                for i, question in enumerate(related_questions):
                    st.markdown(f"{i+1}. {question}")
                    # 더 강력한 고유 키 생성 시스템
                    unique_key = _generate_unique_button_key(message_idx, i, question)
                    
                    if st.button(f"🔍 질문하기", key=unique_key, help=question):
                        st.session_state.suggested_question = question
                        st.rerun()

def main():
    """메인 함수"""
    init_session_state()
    
    # 헤더
    st.markdown('<h1 class="main-header">🤖 MultiModal RAG 시스템</h1>', unsafe_allow_html=True)
    st.markdown("한국은행 뉴스 데이터를 활용한 질의응답 시스템입니다.")
    
    # 사이드바
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 서버 상태 확인
        if st.button("🔄 서버 상태 확인"):
            with st.spinner("서버 상태 확인 중..."):
                if st.session_state.rag_client.check_health():
                    st.success("✅ 서버가 정상 작동 중입니다.")
                else:
                    st.error("❌ 서버에 연결할 수 없습니다.")
        
        # 검색 설정
        st.subheader("🔍 검색 설정")
        top_k = st.slider("검색할 문서 수", min_value=1, max_value=20, value=5)
        
        # 시스템 정보
        st.subheader("📊 시스템 정보")
        if st.button("통계 조회"):
            with st.spinner("통계 조회 중..."):
                stats = st.session_state.rag_client.get_stats()
                if "error" not in stats:
                    st.json(stats)
                else:
                    st.error(stats["error"])
        
        # 대화 기록 초기화
        if st.button("🗑️ 대화 기록 초기화"):
            st.session_state.messages = []
            st.rerun()
    
    # 메인 영역
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("💬 질의응답")
        
        # 기존 메시지 표시
        for message in st.session_state.messages:
            display_message(
                message["role"], 
                message["content"],
                message.get("sources"),
                message.get("confidence"),
                message.get("query_analysis"),
                message.get("related_questions")
            )
        
        # 질문 입력
        with st.container():
            # 관련 질문 클릭 시 자동 입력
            default_question = ""
            if hasattr(st.session_state, 'suggested_question'):
                default_question = st.session_state.suggested_question
                del st.session_state.suggested_question
            
            question = st.text_area(
                "질문을 입력하세요:",
                value=default_question,
                height=100,
                placeholder="예: 한국은행의 최근 정책 동향은 어떻게 되나요?"
            )
            
            col_btn1, col_btn2 = st.columns([1, 4])
            with col_btn1:
                ask_button = st.button("🚀 질문하기", type="primary")
    
    with col2:
        st.subheader("🔍 문서 검색")
        
        # 검색 입력
        search_query = st.text_input("검색어 입력:", placeholder="검색하고 싶은 키워드")
        
        if st.button("🔍 검색"):
            if search_query:
                with st.spinner("검색 중..."):
                    search_results = st.session_state.rag_client.search(search_query, top_k=10)
                    
                    if "error" not in search_results:
                        st.write(f"**검색 결과 ({len(search_results['results'])}개):**")
                        
                        for i, result in enumerate(search_results['results']):
                            score = result['score']
                            with st.expander(f"문서 {i+1} (유사도: {score:.3f})"):
                                st.write(f"**타입:** {result['metadata'].get('doc_type', 'Unknown')}")
                                st.write(f"**내용:** {result['content'][:300]}...")
                    else:
                        st.error(search_results["error"])
            else:
                st.warning("검색어를 입력해주세요.")
    
    # 질문 처리
    if ask_button and question:
        # 사용자 메시지 추가
        st.session_state.messages.append({
            "role": "user",
            "content": question
        })
        
        # 답변 생성
        with st.spinner("답변 생성 중... (최대 1분 소요)"):
            result = st.session_state.rag_client.query(question, top_k=top_k)
            
            if "error" not in result:
                # AI 답변 추가
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result.get("sources", []),
                    "confidence": result.get("confidence", 0.0),
                    "query_analysis": result.get("query_analysis", ""),
                    "related_questions": result.get("related_questions", [])
                })
            else:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": f"죄송합니다. 오류가 발생했습니다: {result['error']}"
                })
        
        st.rerun()
    
    # 하단 정보
    st.markdown("---")
    st.markdown("**💡 사용 팁:**")
    st.markdown("- 구체적이고 명확한 질문을 하시면 더 정확한 답변을 받을 수 있습니다.")
    st.markdown("- 한국은행과 관련된 정책, 뉴스, 경제 동향에 대해 질문해보세요.")
    st.markdown("- 우측 사이드바에서 검색 설정을 조정할 수 있습니다.")

if __name__ == "__main__":
    main() 