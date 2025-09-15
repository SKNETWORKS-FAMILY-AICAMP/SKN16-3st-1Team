"""
RAG Pipeline

RAG (Retrieval-Augmented Generation) 파이프라인
벡터 검색, 컨텍스트 생성, LLM 답변 생성을 통합
"""

from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
import time
import structlog
from pathlib import Path

from .korean_llm_chain import KoreanLLMChain
from .conversation_manager import ConversationManager

logger = structlog.get_logger(__name__)


@dataclass
class RAGResult:
    """RAG 파이프라인 결과"""
    question: str
    answer: str
    context: Optional[str] = None
    source_documents: Optional[List[Dict[str, Any]]] = None
    processing_time: float = 0.0
    retrieval_time: float = 0.0
    generation_time: float = 0.0
    context_used: bool = False


@dataclass
class RetrievalResult:
    """검색 결과"""
    documents: List[Dict[str, Any]]
    query: str
    retrieval_time: float
    total_documents: int


class RAGPipeline:
    """
    RAG 파이프라인 메인 클래스

    문서 검색, 컨텍스트 구성, LLM 답변 생성을 통합하는 파이프라인
    """

    def __init__(
        self,
        vector_store=None,  # ChromaDB 벡터 스토어 (추후 구현)
        retrieval_system=None,  # 하이브리드 검색 시스템 (추후 구현)
        llm_model: str = "gpt-3.5-turbo",
        llm_temperature: float = 0.7,
        llm_max_tokens: int = 1000,
        max_context_length: int = 4000,
        top_k_documents: int = 5,
        relevance_threshold: float = 0.5,
        storage_dir: str = "data/conversations"
    ):
        """
        Args:
            vector_store: 벡터 저장소 (ChromaDB)
            retrieval_system: 검색 시스템 (BM25 + Vector 하이브리드)
            llm_model: 사용할 LLM 모델
            llm_temperature: LLM temperature
            llm_max_tokens: LLM 최대 토큰
            max_context_length: 최대 컨텍스트 길이
            top_k_documents: 검색할 문서 개수
            relevance_threshold: 관련도 임계값
            storage_dir: 대화 저장 디렉토리
        """
        self.vector_store = vector_store
        self.retrieval_system = retrieval_system
        self.max_context_length = max_context_length
        self.top_k_documents = top_k_documents
        self.relevance_threshold = relevance_threshold

        # LLM 체인 초기화
        self.llm_chain = KoreanLLMChain(
            model_name=llm_model,
            temperature=llm_temperature,
            max_tokens=llm_max_tokens
        )

        # 대화 관리자 초기화
        self.conversation_manager = ConversationManager(storage_dir=storage_dir)

        logger.info(
            "RAGPipeline initialized",
            model=llm_model,
            max_context_length=self.max_context_length,
            top_k_documents=self.top_k_documents
        )

    def retrieve_documents(self, query: str) -> RetrievalResult:
        """
        쿼리에 대해 관련 문서 검색

        Args:
            query: 검색 쿼리

        Returns:
            RetrievalResult: 검색 결과
        """
        start_time = time.time()

        try:
            # TODO: 실제 벡터 스토어와 검색 시스템 구현 후 교체
            if self.retrieval_system:
                # 하이브리드 검색 (BM25 + Vector)
                documents = self.retrieval_system.search(
                    query=query,
                    top_k=self.top_k_documents,
                    score_threshold=self.relevance_threshold
                )
            elif self.vector_store:
                # 벡터 검색만
                documents = self.vector_store.similarity_search_with_score(
                    query=query,
                    k=self.top_k_documents
                )
                # 점수 기준으로 필터링
                documents = [
                    doc for doc, score in documents
                    if score >= self.relevance_threshold
                ]
            else:
                # 임시: 검색 시스템이 없는 경우 빈 결과
                logger.warning("No retrieval system available, returning empty results")
                documents = []

            retrieval_time = time.time() - start_time

            result = RetrievalResult(
                documents=documents,
                query=query,
                retrieval_time=retrieval_time,
                total_documents=len(documents)
            )

            logger.info(
                "Document retrieval completed",
                query=query[:50] + "...",
                documents_found=len(documents),
                retrieval_time=retrieval_time
            )

            return result

        except Exception as e:
            logger.error("Error during document retrieval", error=str(e), query=query)
            return RetrievalResult(
                documents=[],
                query=query,
                retrieval_time=time.time() - start_time,
                total_documents=0
            )

    def build_context(
        self,
        documents: List[Dict[str, Any]],
        max_length: Optional[int] = None
    ) -> Tuple[str, List[Dict[str, Any]]]:
        """
        검색된 문서들로부터 컨텍스트 구성

        Args:
            documents: 검색된 문서 리스트
            max_length: 최대 컨텍스트 길이

        Returns:
            Tuple[컨텍스트 문자열, 소스 정보 리스트]
        """
        if not documents:
            return "", []

        max_length = max_length or self.max_context_length
        context_parts = []
        source_info = []
        current_length = 0

        for i, doc in enumerate(documents):
            # 문서 정보 추출 (구조는 실제 벡터 스토어에 따라 달라질 수 있음)
            if isinstance(doc, dict):
                content = doc.get('content', str(doc))
                metadata = doc.get('metadata', {})
            else:
                content = str(doc)
                metadata = {}

            # 길이 체크
            if current_length + len(content) > max_length:
                # 남은 공간에 맞게 잘라서 추가
                remaining = max_length - current_length
                if remaining > 100:  # 최소 100자 이상일 때만 추가
                    content = content[:remaining] + "..."
                else:
                    break

            # 컨텍스트 구성
            section_header = f"\n[문서 {i+1}]"
            if metadata.get('source_file'):
                section_header += f" (출처: {metadata['source_file']})"

            context_part = f"{section_header}\n{content}\n"
            context_parts.append(context_part)
            current_length += len(context_part)

            # 소스 정보 수집
            source_info.append({
                'index': i + 1,
                'content_preview': content[:100] + "..." if len(content) > 100 else content,
                'metadata': metadata,
                'source_file': metadata.get('source_file', 'Unknown'),
                'relevance_score': metadata.get('score', 0.0)
            })

        context = "\n".join(context_parts)

        logger.info(
            "Context built",
            documents_used=len(context_parts),
            context_length=len(context),
            max_length=max_length
        )

        return context, source_info

    def generate_answer(
        self,
        question: str,
        use_rag: bool = True,
        session_title: Optional[str] = None
    ) -> RAGResult:
        """
        질문에 대한 답변 생성 (RAG 파이프라인의 메인 메서드)

        Args:
            question: 사용자 질문
            use_rag: RAG 사용 여부
            session_title: 새 세션 생성 시 제목

        Returns:
            RAGResult: RAG 결과
        """
        total_start_time = time.time()

        try:
            # 현재 세션이 없으면 새로 생성
            if not self.conversation_manager.current_session:
                self.conversation_manager.create_new_session(session_title)

            context = ""
            source_documents = []
            retrieval_time = 0.0
            context_used = False

            if use_rag:
                # 1. 문서 검색
                retrieval_result = self.retrieve_documents(question)
                retrieval_time = retrieval_result.retrieval_time

                if retrieval_result.documents:
                    # 2. 컨텍스트 구성
                    context, source_documents = self.build_context(retrieval_result.documents)
                    context_used = len(context.strip()) > 0

            # 3. LLM으로 답변 생성
            generation_start_time = time.time()

            llm_result = self.llm_chain.generate_answer(
                question=question,
                context=context if context_used else None,
                source_info=source_documents
            )

            generation_time = time.time() - generation_start_time
            total_time = time.time() - total_start_time

            # 4. 대화 관리자에 기록
            self.conversation_manager.add_turn(
                question=question,
                answer=llm_result["answer"],
                context_used=context_used,
                context_content=context if context_used else None,
                source_files=[info["source_file"] for info in source_documents],
                processing_time=total_time
            )

            # 5. 결과 구성
            result = RAGResult(
                question=question,
                answer=llm_result["answer"],
                context=context if context_used else None,
                source_documents=source_documents,
                processing_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time,
                context_used=context_used
            )

            logger.info(
                "RAG pipeline completed",
                question_length=len(question),
                answer_length=len(result.answer),
                context_used=context_used,
                processing_time=total_time,
                retrieval_time=retrieval_time,
                generation_time=generation_time
            )

            return result

        except Exception as e:
            logger.error("Error in RAG pipeline", error=str(e), question=question)

            # 에러 시에도 기본 응답 생성 시도
            try:
                llm_result = self.llm_chain.generate_answer(question=question)
                return RAGResult(
                    question=question,
                    answer=llm_result.get("answer", "죄송합니다. 오류가 발생했습니다."),
                    processing_time=time.time() - total_start_time,
                    context_used=False
                )
            except Exception as inner_e:
                logger.error("Error in fallback answer generation", error=str(inner_e))
                return RAGResult(
                    question=question,
                    answer="죄송합니다. 시스템에 문제가 발생했습니다. 잠시 후 다시 시도해 주세요.",
                    processing_time=time.time() - total_start_time,
                    context_used=False
                )

    def search_similar_questions(
        self,
        question: str,
        limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        유사한 이전 질문들 검색

        Args:
            question: 현재 질문
            limit: 반환할 최대 개수

        Returns:
            유사한 질문 리스트
        """
        try:
            # 대화 기록에서 유사한 질문 검색
            similar_conversations = self.conversation_manager.search_conversations(
                query=question,
                limit=limit
            )

            logger.info(
                "Similar questions search completed",
                question=question[:50] + "...",
                found_count=len(similar_conversations)
            )

            return similar_conversations

        except Exception as e:
            logger.error("Error searching similar questions", error=str(e))
            return []

    def get_conversation_summary(self) -> Optional[Dict[str, Any]]:
        """현재 대화 세션의 요약 정보 반환"""
        return self.conversation_manager.get_current_session_info()

    def start_new_conversation(self, title: Optional[str] = None) -> str:
        """새로운 대화 시작"""
        session_id = self.conversation_manager.create_new_session(title)
        self.llm_chain.clear_memory()
        logger.info("New conversation started", session_id=session_id)
        return session_id

    def load_conversation(self, session_id: str) -> bool:
        """기존 대화 로드"""
        success = self.conversation_manager.load_session(session_id)
        if success:
            # LLM 메모리도 대화 히스토리로 복원
            self.llm_chain.clear_memory()
            history = self.conversation_manager.get_conversation_history()

            for turn in history:
                self.llm_chain.memory.save_context(
                    {"question": turn["question"]},
                    {"answer": turn["answer"]}
                )

            logger.info("Conversation loaded and LLM memory restored", session_id=session_id)

        return success

    def update_llm_settings(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """LLM 설정 업데이트"""
        self.llm_chain.update_model_params(
            temperature=temperature,
            max_tokens=max_tokens
        )

    def get_available_sessions(self) -> List[Dict[str, Any]]:
        """사용 가능한 대화 세션 목록 반환"""
        return self.conversation_manager.get_session_list()

    def export_conversation(self, session_id: Optional[str] = None) -> Optional[str]:
        """대화를 텍스트로 내보내기"""
        return self.conversation_manager.export_session_to_text(session_id)