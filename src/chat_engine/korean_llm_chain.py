"""
Korean LLM Chain for Educational Content
"""

from typing import Dict, List, Any, Optional
from langchain.schema import BaseMessage, HumanMessage, SystemMessage, AIMessage
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_openai import ChatOpenAI
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from langchain.schema.output_parser import StrOutputParser
import structlog

logger = structlog.get_logger(__name__)


class KoreanLLMChain:
    """
    RAG 컨텍스트와 대화 히스토리를 통합하여 답변 생성
    """

    def __init__(
        self,
        model_name: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: int = 1000,
        memory_k: int = 10
    ):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.memory_k = memory_k

        # LLM 초기화
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # 메모리 초기화 (최근 k개의 대화만 유지)
        self.memory = ConversationBufferWindowMemory(
            k=self.memory_k,
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

        # 프롬프트 템플릿 설정
        self._setup_prompts()

        # 체인 구성
        self._setup_chain()

        logger.info(
            "KoreanLLMChain initialized",
            model=self.model_name,
            temperature=self.temperature,
            memory_k=self.memory_k
        )

    def _setup_prompts(self):
        """프롬프트 템플릿 설정"""

        # 시스템 프롬프트 - 한국어 교육 콘텐츠 전문가 역할
        self.system_prompt = """당신은 AI LLM 교육 전문 AI 어시스턴트입니다.

주요 역할:
1. 한국 학생들을 위한 AI/ML 교육 콘텐츠 질문에 답변
2. PDF 강의 자료와 Jupyter 노트북 코드 예제 기반 설명 제공
3. 복잡한 개념을 쉽고 이해하기 쉽게 설명
4. 실습 코드와 이론을 연결하여 설명

답변 원칙:
- 한국어로 정확하게 답변
- 제공된 컨텍스트 정보를 우선적으로 활용
- 코드 예제가 있다면 구체적으로 설명
- 학습자 수준에 맞춰 단계적으로 설명
- 추가 학습을 위한 방향 제시

컨텍스트가 부족한 경우:
- 일반적인 지식으로 도움이 될 수 있는 답변 제공
- 더 구체적인 정보가 필요하다면 명시적으로 안내
- 관련 학습 자료나 키워드 제안"""

        # RAG 답변 생성 프롬프트
        self.rag_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="""다음 컨텍스트 정보를 바탕으로 질문에 답변해주세요:

컨텍스트:
{context}

이전 대화 기록:
{chat_history}

질문: {question}

답변:""")
        ])

        # 일반 대화 프롬프트 (컨텍스트 없는 경우)
        self.general_prompt_template = ChatPromptTemplate.from_messages([
            SystemMessage(content=self.system_prompt),
            HumanMessage(content="""이전 대화 기록:
{chat_history}

질문: {question}

답변:""")
        ])

    def _setup_chain(self):
        """LangChain 체인 구성"""

        # RAG 체인 (컨텍스트가 있는 경우)
        self.rag_chain = (
            RunnableParallel({
                "context": RunnablePassthrough(),
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self._format_chat_history()
            })
            | self.rag_prompt_template
            | self.llm
            | StrOutputParser()
        )

        # 일반 체인 (컨텍스트가 없는 경우)
        self.general_chain = (
            RunnableParallel({
                "question": RunnablePassthrough(),
                "chat_history": lambda x: self._format_chat_history()
            })
            | self.general_prompt_template
            | self.llm
            | StrOutputParser()
        )

    def _format_chat_history(self) -> str:
        """대화 히스토리를 문자열로 포맷팅"""
        if not hasattr(self.memory, 'chat_memory') or not self.memory.chat_memory.messages:
            return "이전 대화 없음"

        formatted_history = []
        messages = self.memory.chat_memory.messages[-self.memory_k*2:]  # 최근 k개 대화쌍

        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                formatted_history.append(f"사용자: {human_msg.content}")
                formatted_history.append(f"AI: {ai_msg.content}")

        return "\n".join(formatted_history) if formatted_history else "이전 대화 없음"

    def generate_answer(
        self,
        question: str,
        context: Optional[str] = None,
        source_info: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성

        Args:
            question: 사용자 질문
            context: RAG 검색으로 얻은 컨텍스트
            source_info: 소스 파일 정보 리스트

        Returns:
            답변과 메타데이터를 포함한 딕셔너리
        """
        try:
            if context and context.strip():
                # RAG 체인 사용
                answer = self.rag_chain.invoke({
                    "question": question,
                    "context": context
                })
                used_context = True
            else:
                # 일반 체인 사용
                answer = self.general_chain.invoke({
                    "question": question
                })
                used_context = False

            # 메모리에 대화 저장
            self.memory.save_context(
                {"question": question},
                {"answer": answer}
            )

            result = {
                "answer": answer,
                "question": question,
                "used_context": used_context,
                "context": context if used_context else None,
                "source_info": source_info if source_info else [],
                "conversation_length": len(self.memory.chat_memory.messages) // 2
            }

            logger.info(
                "Answer generated successfully",
                question_length=len(question),
                answer_length=len(answer),
                used_context=used_context,
                conversation_length=result["conversation_length"]
            )

            return result

        except Exception as e:
            logger.error("Error generating answer", error=str(e), question=question)
            return {
                "answer": "죄송합니다. 답변 생성 중 오류가 발생했습니다. 다시 시도해 주세요.",
                "question": question,
                "used_context": False,
                "context": None,
                "source_info": [],
                "error": str(e),
                "conversation_length": len(self.memory.chat_memory.messages) // 2
            }

    def clear_memory(self):
        """대화 히스토리 초기화"""
        self.memory.clear()
        logger.info("Conversation memory cleared")

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """대화 히스토리 반환"""
        history = []
        messages = self.memory.chat_memory.messages

        for i in range(0, len(messages), 2):
            if i + 1 < len(messages):
                human_msg = messages[i]
                ai_msg = messages[i + 1]
                history.append({
                    "question": human_msg.content,
                    "answer": ai_msg.content
                })

        return history

    def update_model_params(
        self,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ):
        """모델 파라미터 업데이트"""
        if temperature is not None:
            self.temperature = temperature
        if max_tokens is not None:
            self.max_tokens = max_tokens

        # LLM 재초기화
        self.llm = ChatOpenAI(
            model_name=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )

        # 체인 재구성
        self._setup_chain()

        logger.info(
            "Model parameters updated",
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )