# SKN16-3st-1Team
SKN 16기 3차 단위프로젝트 **RAG 강의 자료기반 AI Chatbot**

벡터 데이터베이스에 저장된 AI 강의 자료(PDF, Jupyter Notebook)를 사용자가 쿼리하고, LLM 기반 응답, 대화 요약, 소스 파일 경로 쿼리 등을 제공하는 채팅 기반 교육 시스템입니다.

## 주요 기능

- **문서 처리**: PDF 및 Jupyter Notebook 처리 (셀 분리 기능 포함)
- **벡터 검색**: Chroma DB를 사용한 교육 자료 시맨틱 검색
- **채팅 인터페이스**: Gradio를 사용한 대화형 채팅 (대화 컨텍스트 유지)
- **소스 코드 포맷팅**: UI 응답 내 구문 강조
- **PDF 내보내기**: 다운로드 가능한 PDF 형식의 대화 요약
- **파일 경로 쿼리**: 소스 파일 위치 쿼리
- **강의 날짜별 검색**: 강의 날짜 별 소스 코드 및 강의 자료 쿼리
- **소스 코드 설명**: 소스 코드에 대한 설명 기능
- **강의 회고록 초안 자동생성**: 설정한 기간에 대한 강의 회고록 초안 자동 생성 기능
- **검증 보고**: Retriever 응답에 대한 품질 지표

## 프로젝트 구조

```
📁 SKN16-3st-1Team/
    ├── main.py                     # 🚀 Gradio 메인 애플리케이션
    📁 src/                        # 📦 핵심 라이브러리
    ├── types/                    # 📊 데이터 Classes (Pydantic)
    │   ├── conversation.py        # 대화 모델
    │   ├── document_chunk.py      # 문서 청크 모델
    │   ├── educational_material.py # 교육 자료 모델
    │   ├── ensemble_result.py     # 앙상블 결과 모델
    │   ├── message.py             # 메시지 모델
    │   ├── query_result.py        # 쿼리 결과 모델
    │   ├── retriever_configuration.py # 검색기 설정 모델
    │   ├── search_result.py       # 검색 결과 모델
    │   └── validation_report.py   # 검증 리포트 모델
    ├── document_processor/        # 📄 PDF/노트북 처리
    ├── vector_store/             # 🗃️ Chroma DB 
    ├── retrieval_system/         # 🔍 고급 검색 시스템 (BM25+Vector)
    ├── chat_engine/              # 🤖 LLM 엔진 & RAG 파이프라인
    ├── ui_interface/             # 🖥️ Gradio UI 컴포넌트
    ├── utils/                    # 🔧 유틸리티
    │   └── logger.py             # 로깅 유틸리티
    ├── config/                   # ⚙️ 설정
        └── settings.py           # 애플리케이션 설정

    📁 data/                      # 📂 데이터
    ├── vector_db/               # 벡터 데이터베이스
    ├── educational_materials/   # 교육 자료
    │   ├── pdfs/               # PDF 파일
    │   ├── notebooks/          # Jupyter 노트북
    │   └── uploads/            # 업로드된 파일
```

## 🚀 실행 방법

```bash
# 1. 환경 설정
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 2. 애플리케이션 실행
python main.py

# 3. 브라우저 접속
# http://localhost:7860
```