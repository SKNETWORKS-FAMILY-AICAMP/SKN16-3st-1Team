# -*- coding: utf-8 -*-
"""
RAG 시스템 메인 클래스
벡터 검색과 LLM을 결합한 질의응답 시스템
"""

import os
import re
import calendar
import datetime as dt
from typing import List, Dict, Any, Optional, Tuple

from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.retrievers import BM25Retriever, EnsembleRetriever
from langchain.schema import Document, HumanMessage
from langchain.prompts import PromptTemplate

# langchain-openai 버전에 따라 model / model_name 차이가 있으므로 안전 생성기 사용
from langchain_openai import ChatOpenAI

import dateparser


# -----------------------------
# 유틸: 길이 가드
# -----------------------------
def _safe_join_texts(texts: List[str], max_chars: int = 18000, sep: str = "\n\n") -> str:
    joined, total = [], 0
    for t in texts:
        t = (t or "").strip()
        if not t:
            continue
        add = len(t) + len(sep)
        if total + add > max_chars:
            break
        joined.append(t)
        total += add
    return sep.join(joined)


# -----------------------------
# 유틸: ChatOpenAI 생성 (버전 호환)
# -----------------------------
def _build_llm(llm_model: str, temperature: float = 0.1, max_tokens: int = 2000) -> ChatOpenAI:
    """
    langchain-openai 0.1.x/0.2.x 계열의 model vs model_name 호환 처리
    """
    try:
        # 최신 권장: model=
        return ChatOpenAI(model=llm_model, temperature=temperature, max_tokens=max_tokens)
    except TypeError:
        # 구버전: model_name=
        return ChatOpenAI(model_name=llm_model, temperature=temperature, max_tokens=max_tokens)


class RAGSystem:
    """RAG 시스템 메인 클래스"""

    def __init__(
        self,
        db_path: str = "./chroma_db",
        embedding_model: str = "google/embeddinggemma-300m",
        llm_model: str = "gpt-4o-mini",
        k: int = 5
    ):
        """
        RAG 시스템 초기화

        Args:
            db_path: Chroma DB 경로
            embedding_model: 임베딩 모델명
            llm_model: LLM 모델명 (OpenAI)
            k: 검색할 문서 수
        """
        self.db_path = db_path
        self.k = k

        # 임베딩 모델 초기화 (모델 미존재/에러 시 합리적 폴백)
        try:
            self.embeddings = HuggingFaceEmbeddings(
                model_name=embedding_model,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
        except Exception as e:
            print(f"[경고] 임베딩 '{embedding_model}' 로드 실패: {e}")
            print("[대체] 'intfloat/multilingual-e5-base'로 폴백합니다.")
            self.embeddings = HuggingFaceEmbeddings(
                model_name="intfloat/multilingual-e5-base",
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )

        # LLM 초기화
        self.llm = _build_llm(llm_model, temperature=0.1, max_tokens=2000)

        # 벡터 스토어 로드
        self.vectorstore = None
        self.ensemble_retriever = None
        self._load_vectorstore()

        # 프롬프트 템플릿
        self.prompt_template = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "다음은 LLM 강의 자료에서 검색된 관련 정보입니다:\n\n"
                "{context}\n\n"
                "위 정보를 바탕으로 다음 질문에 답해주세요:\n"
                "질문: {question}\n\n"
                "답변 시 다음 사항을 고려해주세요:\n"
                "1. 제공된 강의 자료의 내용만을 기반으로 답변하세요\n"
                "2. 소스 코드가 포함된 경우, 해당 파일명과 셀 번호를 명시하세요\n"
                "3. 강의 날짜가 언급된 경우, 해당 날짜를 포함하여 답변하세요\n"
                "4. 확실하지 않은 내용은 추측하지 말고 '제공된 자료에서 확인할 수 없습니다'라고 말하세요\n\n"
                "답변:"
            )
        )

    def _load_vectorstore(self):
        """벡터 스토어 로드"""
        if not os.path.exists(self.db_path):
            print(f"[알림] 벡터 DB가 존재하지 않습니다: {self.db_path}")
            return

        try:
            self.vectorstore = Chroma(
                persist_directory=self.db_path,
                embedding_function=self.embeddings
            )
            # 앙상블 리트리버 설정
            self._setup_ensemble_retriever()
            print("[완료] 벡터 스토어가 성공적으로 로드되었습니다.")
        except Exception as e:
            print(f"[오류] 벡터 스토어 로드 중 오류: {e}")

    def _setup_ensemble_retriever(self):
        """앙상블 리트리버 설정 (벡터 검색 + BM25)"""
        if not self.vectorstore:
            return

        try:
            all_docs = self.vectorstore.get()
            documents = [
                Document(page_content=content, metadata=metadata or {})
                for content, metadata in zip(all_docs.get('documents', []), all_docs.get('metadatas', []))
            ]

            # 문서가 없으면 벡터 리트리버만
            if not documents:
                self.ensemble_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})
                return

            bm25_retriever = BM25Retriever.from_documents(documents)
            bm25_retriever.k = self.k

            vector_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

            self.ensemble_retriever = EnsembleRetriever(
                retrievers=[vector_retriever, bm25_retriever],
                weights=[0.7, 0.3]
            )
        except Exception as e:
            print(f"[경고] 앙상블 리트리버 설정 중 오류(벡터만 사용): {e}")
            self.ensemble_retriever = self.vectorstore.as_retriever(search_kwargs={"k": self.k})

    def search_documents(self, query: str, filter_metadata: Optional[Dict] = None) -> List[Document]:
        """
        문서 검색
        """
        if not self.vectorstore:
            return []

        try:
            if filter_metadata:
                # 메타데이터 필터가 있는 경우 벡터 검색만 사용
                results = self.vectorstore.similarity_search(
                    query,
                    k=self.k,
                    filter=filter_metadata
                )
            else:
                # 앙상블 리트리버 사용(없으면 벡터 폴백)
                retriever = self.ensemble_retriever or self.vectorstore.as_retriever(search_kwargs={"k": self.k})
                results = retriever.get_relevant_documents(query)
            return results
        except Exception as e:
            print(f"[오류] 문서 검색 중 오류: {e}")
            return []

    def get_code_snippets(self, query: str) -> List[Dict[str, Any]]:
        """
        코드 스니펫 검색
        """
        filter_metadata = {"content_type": "code"}
        code_docs = self.search_documents(query, filter_metadata)

        code_snippets = []
        for doc in code_docs:
            md = doc.metadata or {}
            snippet_info = {
                "content": doc.page_content,
                "filename": md.get("filename", "Unknown"),
                "cell_index": md.get("cell_index", "Unknown"),
                "lecture_date": md.get("lecture_date", "Unknown"),
                "libraries": md.get("libraries", "").split(", ") if md.get("libraries") else [],
                "model_types": md.get("model_types", "").split(", ") if md.get("model_types") else []
            }
            code_snippets.append(snippet_info)
        return code_snippets

    def get_lecture_summary(self, date: str) -> str:
        """
        특정 날짜 강의 요약 (YYYY-MM-DD)
        """
        filter_metadata = {"lecture_date": date}
        docs = self.search_documents("강의 내용 요약", filter_metadata)

        if not docs:
            return f"{date} 날짜의 강의 자료를 찾을 수 없습니다."

        context = _safe_join_texts([d.page_content for d in docs], max_chars=18000)

        summary_prompt = (
            f"다음은 {date} 강의 자료입니다:\n\n"
            f"{context}\n\n"
            f"위 자료를 바탕으로 {date} 강의의 주요 내용을 요약해주세요.\n"
            f"- 다룬 주제들\n- 실습한 코드나 모델\n- 학습 목표\n- 주요 개념\n\n요약:"
        )
        try:
            resp = self.llm.invoke([HumanMessage(content=summary_prompt)])
            return resp.content
        except Exception as e:
            return f"요약 생성 중 오류가 발생했습니다: {e}"

    def _parse_year_month_week(self, text: str) -> Optional[tuple]:
        """
        '2025년 6월 3주차', '6월 3주차', '6/3주차' → (year, month, week_idx)
        """
        m = re.search(r"(?:(\d{4})\s*년\s*)?(\d{1,2})\s*월\s*(\d)\s*주차", text)
        if not m:
            m = re.search(r"(?:(\d{4}))?[/\-\.]?\s*(\d{1,2})\s*[/\-\.]?\s*(\d)\s*주차", text)
        if not m:
            return None
        year = int(m.group(1)) if m.group(1) else dt.date.today().year
        month = int(m.group(2))
        week_idx = int(m.group(3))
        return year, month, week_idx

    def _week_mon_to_fri_dates(self, year: int, month: int, week_idx: int) -> List[str]:
        """
        해당 월의 week_idx번째 '월~금' 날짜 리스트(YYYY-MM-DD)
        """
        cal = calendar.Calendar(firstweekday=calendar.MONDAY)
        mondays = [d for d in cal.itermonthdates(year, month) if d.month == month and d.weekday() == 0]
        if week_idx < 1 or week_idx > len(mondays):
            return []
        mon = mondays[week_idx - 1]
        dates = []
        for i in range(5):
            d = mon + dt.timedelta(days=i)
            if d.month == month and d.weekday() < 5:
                dates.append(d.strftime("%Y-%m-%d"))
        return dates

    def get_week_summary(self, year_month_week: str) -> str:
        """
        '6월 3주차', '2025년 6월 3주차' → 해당 주 월~금 강의 전부 모아 요약
        """
        parsed = self._parse_year_month_week(year_month_week)
        if not parsed:
            return "주차 표현을 이해하지 못했습니다. 예) '2025년 6월 3주차'"

        y, m, w = parsed
        dates = self._week_mon_to_fri_dates(y, m, w)
        if not dates:
            return f"{y}년 {m}월 {w}주차(월~금)에 해당하는 날짜가 없습니다."

        # 내부 API 접근(버전 민감) → try/except
        try:
            raw = self.vectorstore._collection.get(  # 내부 접근(Chroma 버전 바뀌면 실패 가능)
                where={"lecture_date": {"$in": dates}},
                limit=2000,
                include=["documents", "metadatas"]
            )
            docs = [
                Document(page_content=txt, metadata=meta or {})
                for txt, meta in zip(raw.get("documents", []), raw.get("metadatas", []))
            ]
        except Exception:
            # 폴백: 날짜별로 similarity_search
            docs = []
            for d_ in dates:
                docs.extend(self.search_documents("강의 내용 요약", {"lecture_date": d_}))

        if not docs:
            return f"{y}년 {m}월 {w}주차(월~금) 강의 자료를 찾지 못했습니다."

        context = _safe_join_texts([d.page_content for d in docs], max_chars=18000)

        prompt = (
            f"다음은 {y}년 {m}월 {w}주차(월~금) 강의 자료입니다.\n\n"
            f"{context}\n\n"
            f"위 자료를 바탕으로 해당 주차의 핵심을 구조적으로 요약해줘.\n"
            f"- 날짜별 주요 주제\n- 실습한 코드/모델 요지\n- 핵심 개념/정의\n- 주의사항/한계\n- 다음 학습 추천"
        )
        try:
            resp = self.llm.invoke([HumanMessage(content=prompt)])
            return resp.content
        except Exception as e:
            return f"요약 생성 중 오류가 발생했습니다: {e}"

    def answer_question(self, question: str) -> Dict[str, Any]:
        """
        질문에 대한 답변 생성
        """
        if not self.vectorstore:
            return {
                "answer": "벡터 DB가 로드되지 않았습니다. 먼저 DB를 구축해주세요.",
                "sources": [],
                "metadata": {}
            }

        try:
            relevant_docs = self.search_documents(question)

            if not relevant_docs:
                return {
                    "answer": "관련된 정보를 찾을 수 없습니다.",
                    "sources": [],
                    "metadata": {}
                }

            context_parts, sources = [], []
            for i, doc in enumerate(relevant_docs):
                md = doc.metadata or {}
                filename = md.get("filename", "Unknown")
                content_type = md.get("content_type", "Unknown")
                cell_index = md.get("cell_index", "")

                context_parts.append(f"[문서 {i+1}] {filename} ({content_type})\n{doc.page_content}")

                sources.append({
                    "filename": filename,
                    "content_type": content_type,
                    "cell_index": cell_index,
                    "lecture_date": md.get("lecture_date", "Unknown")
                })

            context = _safe_join_texts(context_parts, max_chars=18000, sep="\n\n")

            prompt_text = self.prompt_template.format(context=context, question=question)
            response = self.llm.invoke([HumanMessage(content=prompt_text)])

            return {
                "answer": response.content,
                "sources": sources,
                "metadata": {
                    "num_sources": len(relevant_docs),
                    "query": question
                }
            }

        except Exception as e:
            return {
                "answer": f"답변 생성 중 오류가 발생했습니다: {e}",
                "sources": [],
                "metadata": {}
            }

    def explain_code(self, code_content: str, context: str = "") -> str:
        """
        코드 설명 생성
        """
        explanation_prompt = (
            "다음 Python 코드를 분석하고 설명해주세요:\n\n"
            f"{context}\n\n"
            "코드:\n```python\n"
            f"{code_content}\n"
            "```\n\n"
            "설명 시 포함할 내용:\n"
            "1. 코드의 전체적인 목적과 기능\n"
            "2. 주요 구성 요소 (클래스, 함수, 변수 등)\n"
            "3. 사용된 라이브러리와 그 역할\n"
            "4. 코드의 동작 순서\n"
            "5. 주의사항이나 개선점\n\n"
            "설명:"
        )
        try:
            response = self.llm.invoke([HumanMessage(content=explanation_prompt)])
            return response.content
        except Exception as e:
            return f"코드 설명 생성 중 오류가 발생했습니다: {e}"


def main():
    """테스트용 메인 함수"""
    rag_system = RAGSystem()

    # 간단 점검
    print("has get_lecture_summary:", hasattr(rag_system, "get_lecture_summary"))

    # 테스트 질문들
    test_questions = [
        "CNN 모델 만드는 소스 코드 찾아줘",
        "RNN과 CNN 비교 설명해줘",
        "RAG 시스템이 뭐야?"
    ]

    for question in test_questions:
        print(f"\n질문: {question}")
        result = rag_system.answer_question(question)
        print(f"답변: {result['answer'][:400]} ...")  # 길이 제한 출력
        print(f"소스 수: {result['metadata'].get('num_sources', 0)}")


if __name__ == "__main__":
    main()
