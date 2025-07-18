# PDF RAG 시스템 프로젝트

## 📋 프로젝트 개요

이 프로젝트는 **PDF 문서를 업로드하고 AI 기반 질문-답변을 수행하는 RAG(Retrieval-Augmented Generation) 시스템**입니다. FastAPI 기반 백엔드와 웹 인터페이스를 통해 사용자가 PDF 문서를 업로드하고, 문서 내용에 대해 자연어로 질문할 수 있습니다.

## 🎯 주요 기능

### 🔄 PDF 처리 및 분석
- **PDF 텍스트 추출**: PyMuPDF를 사용한 고품질 텍스트 추출
- **계층화된 청킹**: 2단계 청킹 전략 (MainChunk + SubChunk)
- **구조 보존**: 섹션, 제목, 페이지 정보 유지

### 🔍 하이브리드 검색 시스템
- **벡터 검색 (70%)**: Google Gemini 임베딩 기반 의미적 검색
- **키워드 검색 (30%)**: BM25 알고리즘 기반 키워드 매칭
- **가중치 조정**: 도메인별 최적화 가능

### 🤖 AI 답변 생성
- **Google Gemini API**: 최신 LLM을 활용한 자연스러운 답변
- **컨텍스트 기반**: 관련 문서 섹션을 기반으로 한 정확한 답변
- **출처 제공**: 페이지 번호와 섹션 정보 포함

### 📊 웹 인터페이스
- **드래그 앤 드롭**: 간편한 PDF 업로드
- **실시간 진행률**: 파일 처리 상태 시각화
- **채팅 UI**: 직관적인 질문-답변 인터페이스

## 🏗️ 시스템 아키텍처

### 계층화된 청킹 전략

```
📄 PDF 문서
    ↓
📖 MainChunk (섹션/챕터 단위)
    ├── SubChunk 1 (2-3문장)
    ├── SubChunk 2 (2-3문장)
    └── SubChunk 3 (2-3문장)
```

**장점:**
- 정밀한 검색을 위한 SubChunk 벡터화
- 풍부한 컨텍스트를 위한 MainChunk 활용
- 문서 구조 정보 보존

### 데이터 처리 파이프라인

```
PDF 업로드 → 텍스트 추출 → 청킹 → 임베딩 생성 → 벡터 저장 → 인덱싱
```

## 🛠️ 기술 스택

### 백엔드
- **FastAPI**: 웹 API 프레임워크
- **Python 3.11+**: 프로그래밍 언어
- **Poetry**: 의존성 관리

### AI/ML
- **Google Gemini API**: LLM 및 임베딩 생성
- **LangChain**: AI 모델 통합 프레임워크
- **tiktoken**: 토큰 카운팅

### 데이터 저장 및 검색
- **ChromaDB**: 벡터 데이터베이스
- **BM25**: 키워드 검색 알고리즘
- **scikit-learn**: 텍스트 처리

### PDF 및 텍스트 처리
- **PyMuPDF**: PDF 텍스트 추출
- **정규표현식**: 텍스트 분할 및 전처리

### 프론트엔드
- **HTML/CSS/JavaScript**: 웹 인터페이스
- **반응형 디자인**: 모바일 친화적 UI

## 📁 프로젝트 구조

```
RAG_lecture/
├── rag_api_llm.py              # 메인 FastAPI 서버
├── index.html                  # 웹 인터페이스
├── pyproject.toml              # Poetry 설정
├── requirements.txt            # pip 의존성
├── .env                        # 환경변수 (API 키)
├── chroma_db/                  # ChromaDB 데이터 저장소
├── sample.txt                  # 샘플 텍스트
├── test.pdf                    # 테스트 PDF
└── [01-11]-AI-LLM*.ipynb      # 학습용 Jupyter 노트북
```

## 🚀 설치 및 실행

### 1. 의존성 설치

**Poetry 사용 (권장):**
```bash
poetry install
```

**pip 사용:**
```bash
pip install -r requirements.txt
```

### 2. 환경 변수 설정

`.env` 파일 생성 후 Google API 키 입력:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

### 3. 서버 실행

```bash
# Poetry 환경에서 실행
poetry run python rag_api_llm.py

# 또는 직접 실행
python rag_api_llm.py
```

서버가 `http://localhost:8080`에서 실행됩니다.

### 4. 웹 인터페이스 접속

브라우저에서 `index.html` 파일을 열거나, 서버 루트 경로에 접속하여 웹 인터페이스를 사용할 수 있습니다.

## 📊 API 엔드포인트

### 문서 관리
- `POST /upload-pdf`: PDF 파일 업로드 및 처리
- `GET /documents`: 업로드된 문서 목록 조회
- `GET /document/{document_id}`: 특정 문서 상세 정보
- `DELETE /delete-document/{document_id}`: 특정 문서 삭제
- `DELETE /delete-all`: 모든 문서 삭제

### 질문-답변
- `POST /question`: 문서 기반 질문 답변

### 시스템 상태
- `GET /`: API 상태 확인
- `GET /status`: 시스템 상태 모니터링

## 💡 사용 방법

### 1. PDF 업로드
1. 웹 인터페이스의 업로드 영역에 PDF 파일을 드래그 앤 드롭
2. 자동으로 파일 처리 진행률 표시
3. 처리 완료 후 문서 목록에 추가됨

### 2. 질문하기
1. 업로드된 문서 중 하나 선택
2. 질문 입력창에 문서 관련 질문 입력
3. AI가 문서 내용을 바탕으로 답변 생성
4. 답변과 함께 관련 페이지 정보 제공

### 3. 문서 관리
- 문서 목록에서 개별 문서 삭제 가능
- 시스템 상태 및 처리된 문서 통계 확인

## 🔧 주요 설정

### 청킹 파라미터
```python
target_tokens = 150          # SubChunk 목표 토큰 수
sentences_per_subchunk = 2-3 # SubChunk당 문장 수
```

### 검색 가중치
```python
vector_weight = 0.7    # 벡터 검색 가중치 (70%)
keyword_weight = 0.3   # 키워드 검색 가중치 (30%)
```

### 배치 처리
```python
batch_size = 30        # 임베딩 생성 배치 크기
```

## 📚 학습 자료

프로젝트에는 다음과 같은 Jupyter 노트북이 포함되어 있습니다:

1. `01-AI-APIKey.ipynb`: API 키 설정 및 환경 구축
2. `02-AI-LLM.ipynb`: 기본 LLM 사용법
3. `03-AI-LLM topic-question.ipynb`: 주제별 질문 처리
4. `04-AI-LLM llm-chain.ipynb`: LLM 체인 구성
5. `05-AI-LLM StrOutPutParser.ipynb`: 출력 파싱
6. `06-AI-LLM LCEL.ipynb`: LangChain Expression Language
7. `07-AI-LLM Text Split.ipynb`: 텍스트 분할 기법
8. `08-AI-LLM embedding-FAISS.ipynb`: 임베딩과 FAISS
9. `09-AI-LLM retriever-embedding-FAISS.ipynb`: 리트리버 구현
10. `10-AI-LLM react_agent.ipynb`: ReAct 에이전트
11. `11-AI-LLM rag_api_llm.ipynb`: RAG 시스템 구현

## 🎯 성능 특징

### 최적화 요소
- **배치 처리**: 임베딩 생성을 30개 단위로 배치 처리
- **비동기 처리**: FastAPI의 비동기 기능 활용
- **인덱스 캐싱**: BM25 인덱스 사전 계산 및 메모리 캐싱
- **효율적 검색**: ChromaDB의 벡터 유사도 검색 최적화

### 확장성
- **멀티 문서 지원**: 여러 PDF 동시 관리
- **문서별 격리**: 각 문서의 독립적인 검색 가능
- **메타데이터 관리**: 풍부한 문서 정보 저장

## 🔐 보안 및 설정

### 환경 변수 관리
- `.env` 파일을 통한 API 키 보안 관리
- `.gitignore`에 민감 정보 제외

### CORS 설정
- 웹 인터페이스와의 통신을 위한 CORS 허용

## 🛡️ 에러 처리

### 강건한 에러 핸들링
- PDF 처리 실패 시 임시 파일 정리
- API 호출 실패 시 적절한 에러 메시지
- 파일 업로드 중단 기능

### 로깅
- 상세한 처리 과정 로깅
- 에러 추적 및 디버깅 지원

## 📈 모니터링

### 시스템 상태 확인
- 업로드된 문서 수 및 상태
- 벡터 데이터베이스 상태
- 메모리 사용량 추적

## 🤝 기여 및 개발

### 개발 환경 설정
1. Poetry를 사용한 가상환경 구성
2. 개발 의존성 설치: `poetry install --dev`
3. 코드 스타일: PEP 8 준수

### 확장 가능성
- 다른 LLM 모델 통합
- 추가 문서 형식 지원 (DOCX, TXT 등)
- 고급 검색 알고리즘 구현
- 사용자 인증 및 권한 관리

## 📄 라이선스

이 프로젝트는 학습 및 연구 목적으로 제작되었습니다.

## 🆘 문제 해결

### 자주 발생하는 문제
1. **API 키 에러**: `.env` 파일에 올바른 Google API 키 설정 확인
2. **포트 충돌**: 8080 포트가 사용 중인 경우 다른 포트로 변경
3. **의존성 문제**: Poetry를 사용하여 올바른 의존성 설치

### 디버깅
- 로그 레벨을 INFO로 설정하여 상세한 처리 과정 확인
- ChromaDB 연결 상태 확인
- 임시 파일 저장 공간 확인

---

💡 **이 프로젝트는 PDF 문서 분석과 AI 기반 질문-답변 시스템의 실무적 구현 예제입니다. RAG 시스템의 핵심 개념과 최신 AI 기술의 실제 적용을 학습할 수 있습니다.**