# 🎓 LLM 강의 검색 & Help RAG Agent

LLM 관련 강의 자료(Jupyter 노트북, PDF)를 기반으로 한 RAG(Retrieval-Augmented Generation) 시스템입니다.

---

## 📋 프로젝트 개요

- **목적**: LLM 강의 자료를 임베딩하여 사용자 질의에 대한 답변 제공
- **데이터**: `educational_materials/` 디렉토리의 `.ipynb` 및 `.pdf` 파일
- **기능**: 코드 검색, 강의 요약, 질의응답, 코드 설명  
  - 🆕 **주차 요약**: `강의 요약: 6월 3주차` 같은 **자연어 주차 입력** 지원  
  - 🆕 **회고록 생성**: `회고록: 7월 3주차`, `회고록: 2025-07-21` 등 **블로그 스타일 회고**  
  - 🆕 **의도 인식 코드 검색**: `CNN`, `RAG`, `LangChain`, `Retriever` 같은 **키워드만**으로도 관련 코드/설명 우선 검색  
  - 🆕 **엔SEMBLE/Fallback**: BM25+벡터 결합(가능 시) → 버전 호환 안 되면 **벡터 검색만 자동 사용**

---

## 🛠 기술 스택

- **Python 3.10+**
- **LangChain** (RAG 파이프라인/LLM)
- **Chroma DB** (벡터 DB)
- **Gradio** (웹 UI)
- **HuggingFace 임베딩**: `google/embeddinggemma-300m`
- **OpenAI LLM**: `gpt-4o-mini`
- **PyMuPDF** (PDF 처리)
- 🆕 **dateparser** (한국어 자연어 날짜/주차 파싱)

---

## 📁 프로젝트 구조



```
3rd_project2/
├── educational_materials/ # 강의 자료 디렉토리
│ ├── notebooks/ # Jupyter 노트북 파일들
│ └── pdfs/ # PDF 강의 자료들
├── src/ # 소스 코드
│ ├── init.py
│ ├── vector_db_builder.py # 벡터 DB 구축 모듈
│ ├── rag_system.py # RAG 시스템 메인 클래스 (주차/회고록/의도 인식 포함) ← 🆕
│ └── gradio_app.py # Gradio 웹 인터페이스 (명령어 라우팅) ← 🆕
├── main.py # 메인 실행 스크립트
├── requirements.txt # Python 의존성 (dateparser, pillow 포함 권장) ← 🆕
├── PRD.md # 프로젝트 요구사항 문서
└── README.md # 이 파일
```

## 🚀 설치 및 실행

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

### 2. OpenAI API 키 설정

```bash
# .env 파일 생성
cp .env.example .env

# .env 파일에서 OpenAI API 키 설정
OPENAI_API_KEY=your_actual_api_key_here
```

### 3. 강의 자료 준비

`educational_materials/` 디렉토리에 강의 자료를 배치:

```
educational_materials/
├── notebooks/
│   ├── 2024-09-16_LangChain_RAG.ipynb
│   ├── 2024-09-17_CNN_Tutorial.ipynb
│   └── ...
└── pdfs/
    ├── 2024-09-16_AI_Lecture.pdf
    └── ...
```

### 4. 애플리케이션 실행

```bash
python main.py
```

브라우저에서 `http://localhost:7860`으로 접속

## 💬 사용법

### 기본 질의응답
- "RNN과 CNN의 차이점은?"
- "딥러닝 모델 학습 방법"
- "오버피팅 방지 기법"

### 코드 검색
- "코드 검색: CNN 모델"
- "코드 검색: 데이터 전처리"
- 🆕 키워드만 입력해도 됨: CNN, RAG, LangChain, Retriever …

### 강의 요약
- "강의 요약: 2024-09-16"
- "강의 요약: 20240916"
- 🆕 강의 요약: 6월 3주차 (자연어 주차 인식)

회고록 (블로그 스타일)

🆕 회고록: 7월 3주차
🆕 회고록: 2025-07-21

### 코드 설명
1. 코드 검색으로 원하는 코드 찾기
2. "코드 설명: [선택한 코드]"

## 🔧 주요 기능

### 1. 벡터 DB 구축 (`vector_db_builder.py`)
- Jupyter 노트북의 코드/마크다운 셀 분리 처리
- PDF 문서 텍스트 추출
- 메타데이터 기반 검색 지원
- 파일명에서 강의 날짜 자동 추출

### 2. RAG 시스템 (`rag_system.py`)
- 앙상블 리트리버 (벡터 검색 + BM25)
- 컨텍스트 기반 답변 생성
- 코드 스니펫 전용 검색
- 강의별 요약 (날짜/주차) ← 🆕
- 회고록 생성 (날짜/주차) ← 🆕

### 3. Gradio 인터페이스 (`gradio_app.py`)
- 실시간 채팅 인터페이스
- 벡터 DB 구축 버튼
- 코드 하이라이팅
- 응답 소스 표시

## ⚙️ 설정

### 임베딩 모델 변경
`src/vector_db_builder.py`에서 모델 변경:

```python
self.embeddings = HuggingFaceEmbeddings(
    model_name="your-embedding-model",
    model_kwargs={'device': 'cpu'},
    encode_kwargs={'normalize_embeddings': True}
)
```

### LLM 모델 변경
`src/rag_system.py`에서 모델 변경:

```python
self.llm = Ollama(model="your-llm-model")
```

## 🐛 문제 해결

### Chroma DB 오류
```bash
# DB 디렉토리 삭제 후 재구축
rm -rf chroma_db/
```

### OpenAI API 오류
```bash
# API 키 확인
echo $OPENAI_API_KEY

# .env 파일 확인
cat .env
```


## 🚧 향후 개선사항

- [ ] 한국어 리랭킹 모델 통합 -> "name": "dragonkue/bge-reranker-v2-m3-ko"
- [ ] 코드 실행 기능
- [ ] 시각화 결과 표시
- [ ] 다중 파일 업로드
- [ ] 검색 결과 필터링