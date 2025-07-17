from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pymupdf
from langchain_google_genai import ChatGoogleGenerativeAI
import tiktoken
import re
from typing import List, Dict, Any
import uuid
import tempfile
import os
import chromadb
from rank_bm25 import BM25Okapi
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import asyncio
from concurrent.futures import ThreadPoolExecutor
import logging
import time
import shutil
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# 환경변수 로드
load_dotenv()
os.environ["CHROMA_TELEMETRY_ANONYMOUS"] = "False"
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GOOGLE_API_KEY)

# 로깅 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
# chromadb telemetry 로그 레벨 WARNING 이상으로 격상(숨김)
logging.getLogger("chromadb.telemetry").setLevel(logging.WARNING)
logging.getLogger("chromadb.telemetry.product.posthog").setLevel(logging.CRITICAL)


app = FastAPI(title="PDF QA System (Gemini)", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ChromaDB 클라이언트 초기화
chroma_client = chromadb.PersistentClient(path="./chroma_db")

# 전역 변수들
collection = None
documents_registry = {}  # 문서 정보를 저장하는 레지스트리
documents_chunks = {}  # 문서별 청크 데이터 저장
documents_bm25 = {}  # 문서별 BM25 인덱스 저장
tokenizer = tiktoken.get_encoding("cl100k_base")  # GPT-4 토크나이저(임베딩 토큰 카운트용)

class QuestionRequest(BaseModel):
    question: str
    document_id: str = None

class MainChunk:
    """구조화된 메인 청크 클래스"""
    def __init__(self, text: str, chunk_id: str, page_start: int, page_end: int, section: str = "", title: str = ""):
        self.text = text
        self.chunk_id = chunk_id
        self.page_start = page_start
        self.page_end = page_end
        self.section = section
        self.title = title
        self.subchunks = []  # 이 청크에 속한 서브청크들

class SubChunk:
    """벡터화될 서브청크 클래스"""
    def __init__(self, text: str, subchunk_id: str, parent_chunk_id: str, sentence_start: int, sentence_end: int):
        self.text = text
        self.subchunk_id = subchunk_id
        self.parent_chunk_id = parent_chunk_id
        self.sentence_start = sentence_start
        self.sentence_end = sentence_end

def count_tokens(text: str) -> int:
    """텍스트의 토큰 수를 계산합니다."""
    return len(tokenizer.encode(text))

def extract_text_from_pdf(pdf_path: str) -> List[Dict[str, Any]]:
    """PDF에서 텍스트를 추출하고 페이지별로 구조화합니다."""
    doc = pymupdf.open(pdf_path)
    pages_data = []
    
    for page_num in range(len(doc)):
        page = doc.load_page(page_num)
        text = page.get_text()
        
        # 목차나 섹션 제목 추출 (개선된 휴리스틱)
        lines = text.split('\n')
        section_title = ""
        
        # 제목 패턴 찾기 (숫자. 제목, 대문자 제목, 등)
        for line in lines[:10]:  # 상위 10줄 확인
            line = line.strip()
            if line:
                # 숫자로 시작하는 제목 (1. 제목, 1.1 제목 등)
                if re.match(r'^\d+\.?\d*\.?\s+[A-Za-z가-힣]', line):
                    section_title = line
                    break
                # 전체 대문자 제목
                elif line.isupper() and 5 <= len(line) <= 50:
                    section_title = line
                    break
                # 첫 글자만 대문자이고 적절한 길이
                elif line[0].isupper() and 10 <= len(line) <= 80 and line.count(' ') <= 8:
                    section_title = line
                    break
        
        pages_data.append({
            'page_num': page_num + 1,
            'text': text,
            'section': section_title
        })
    
    doc.close()
    return pages_data

def create_main_chunks(pages_data: List[Dict[str, Any]]) -> List[MainChunk]:
    """페이지 데이터를 큰 구조화된 청크로 나눕니다."""
    chunks = []
    current_section = ""
    current_chunk_text = ""
    current_pages = []
    section_start_page = 1
    
    for i, page_data in enumerate(pages_data):
        page_num = page_data['page_num']
        text = page_data['text']
        section = page_data['section']
        
        # 새로운 섹션이 시작되면 이전 청크를 완성
        if section and section != current_section and current_chunk_text:
            chunk_id = str(uuid.uuid4())
            chunks.append(MainChunk(
                text=current_chunk_text.strip(),
                chunk_id=chunk_id,
                page_start=section_start_page,
                page_end=current_pages[-1] if current_pages else section_start_page,
                section=current_section,
                title=current_section
            ))
            
            # 새 청크 시작
            current_chunk_text = text
            current_section = section
            current_pages = [page_num]
            section_start_page = page_num
        else:
            # 기존 청크에 페이지 추가
            current_chunk_text += "\n\n" + text if current_chunk_text else text
            current_pages.append(page_num)
            if section and not current_section:
                current_section = section
                section_start_page = page_num
    
    # 마지막 청크 추가
    if current_chunk_text.strip():
        chunk_id = str(uuid.uuid4())
        chunks.append(MainChunk(
            text=current_chunk_text.strip(),
            chunk_id=chunk_id,
            page_start=section_start_page,
            page_end=current_pages[-1] if current_pages else section_start_page,
            section=current_section,
            title=current_section
        ))
    
    # 청크가 너무 적으면 페이지 단위로 분할
    if len(chunks) < 3:
        chunks = []
        for page_data in pages_data:
            chunk_id = str(uuid.uuid4())
            chunks.append(MainChunk(
                text=page_data['text'],
                chunk_id=chunk_id,
                page_start=page_data['page_num'],
                page_end=page_data['page_num'],
                section=page_data['section'],
                title=f"페이지 {page_data['page_num']}"
            ))
    
    return chunks

def create_subchunks_from_main_chunk(main_chunk: MainChunk, target_tokens: int = 150) -> List[SubChunk]:
    """메인 청크에서 2-3문장 단위의 서브청크를 생성합니다."""
    subchunks = []
    text = main_chunk.text
    
    # 문장 단위로 분할 (개선된 정규식)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z가-힣])', text)
    sentences = [s.strip() for s in sentences if s.strip()]
    
    current_subchunk = ""
    current_tokens = 0
    sentence_start_idx = 0
    sentences_in_subchunk = 0
    
    for i, sentence in enumerate(sentences):
        sentence_tokens = count_tokens(sentence)
        
        # 2-3문장이거나 토큰 수가 목표에 도달하면 서브청크 생성
        should_create_subchunk = (
            (sentences_in_subchunk >= 2 and current_tokens + sentence_tokens > target_tokens) or
            sentences_in_subchunk >= 3 or
            (current_tokens + sentence_tokens > target_tokens * 1.5 and sentences_in_subchunk >= 1)
        )
        
        if should_create_subchunk and current_subchunk:
            subchunk_id = str(uuid.uuid4())
            subchunks.append(SubChunk(
                text=current_subchunk.strip(),
                subchunk_id=subchunk_id,
                parent_chunk_id=main_chunk.chunk_id,
                sentence_start=sentence_start_idx,
                sentence_end=i - 1
            ))
            
            # 새 서브청크 시작
            current_subchunk = sentence
            current_tokens = sentence_tokens
            sentence_start_idx = i
            sentences_in_subchunk = 1
        else:
            # 기존 서브청크에 문장 추가
            current_subchunk += " " + sentence if current_subchunk else sentence
            current_tokens += sentence_tokens
            sentences_in_subchunk += 1
    
    # 마지막 서브청크 추가
    if current_subchunk.strip():
        subchunk_id = str(uuid.uuid4())
        subchunks.append(SubChunk(
            text=current_subchunk.strip(),
            subchunk_id=subchunk_id,
            parent_chunk_id=main_chunk.chunk_id,
            sentence_start=sentence_start_idx,
            sentence_end=len(sentences) - 1
        ))
    
    return subchunks

async def create_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """텍스트 배치에 대한 Gemini 임베딩을 생성합니다."""
    try:
        # Gemini 임베딩 생성 (langchain 사용)
        from langchain_google_genai import GoogleGenerativeAIEmbeddings
        embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        embeddings = embeddings_model.embed_documents(texts)
        return embeddings
    except Exception as e:
        logger.error(f"임베딩 생성 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"임베딩 생성 실패: {str(e)}")

def create_bm25_index(subchunks: List[SubChunk]) -> BM25Okapi:
    """서브청크들에 대한 BM25 인덱스를 생성합니다."""
    tokenized_subchunks = []
    for subchunk in subchunks:
        tokens = re.findall(r'\b\w+\b', subchunk.text.lower())
        tokenized_subchunks.append(tokens)
    return BM25Okapi(tokenized_subchunks)


@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    """PDF 파일을 업로드하고 처리합니다."""
    global collection, documents_registry, documents_chunks, documents_bm25
    
    if not file.filename.endswith('.pdf'):
        raise HTTPException(status_code=400, detail="PDF 파일만 업로드 가능합니다.")
    
    try:
        # 임시 파일로 저장
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # PDF에서 텍스트 추출
        logger.info("PDF에서 텍스트 추출 중...")
        pages_data = extract_text_from_pdf(tmp_file_path)
        
        # 메인 청크 생성 (구조화된 큰 청크)
        logger.info("구조화된 메인 청크 생성 중...")
        main_chunks = create_main_chunks(pages_data)
        
        if not main_chunks:
            raise HTTPException(status_code=400, detail="PDF에서 텍스트를 추출할 수 없습니다.")
        
        # 각 메인 청크에서 서브청크 생성
        logger.info("서브청크 생성 중...")
        all_subchunks = []
        for main_chunk in main_chunks:
            subchunks = create_subchunks_from_main_chunk(main_chunk, target_tokens=150)
            main_chunk.subchunks = subchunks
            all_subchunks.extend(subchunks)
        
        if not all_subchunks:
            raise HTTPException(status_code=400, detail="서브청크를 생성할 수 없습니다.")
        
        # 문서 ID 생성
        document_id = f"doc_{uuid.uuid4().hex[:8]}_{int(time.time())}"
        
        # 문서별 데이터 저장
        documents_chunks[document_id] = {
            'main_chunks': main_chunks,
            'subchunks': all_subchunks
        }
        
        # BM25 인덱스 생성 (해당 문서의 서브청크만)
        logger.info("BM25 인덱스 생성 중...")
        documents_bm25[document_id] = create_bm25_index(all_subchunks)
        
        # ChromaDB 컬렉션 생성/업데이트
        collection_name = "rag"  # 기본 컬렉션 이름
        
        try:
            collection = chroma_client.get_collection(collection_name)
        except:
            collection = chroma_client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        
        # 서브청크 임베딩 생성 및 저장 (배치 처리)
        logger.info("서브청크 임베딩 생성 및 벡터 데이터베이스에 저장 중...")
        batch_size = 30  # OpenAI API 제한을 고려한 배치 크기
        
        for i in range(0, len(all_subchunks), batch_size):
            batch_subchunks = all_subchunks[i:i + batch_size]
            batch_texts = [subchunk.text for subchunk in batch_subchunks]
            
            # 임베딩 생성
            embeddings = await create_embeddings_batch(batch_texts)
            
            # 각 서브청크의 부모 청크 정보 찾기
            metadata_list = []
            for subchunk in batch_subchunks:
                parent_chunk = next((chunk for chunk in main_chunks if chunk.chunk_id == subchunk.parent_chunk_id), None)
                metadata_list.append({
                    "document_id": document_id,
                    "subchunk_id": subchunk.subchunk_id,
                    "parent_chunk_id": subchunk.parent_chunk_id,
                    "parent_section": parent_chunk.section if parent_chunk else "",
                    "parent_title": parent_chunk.title if parent_chunk else "",
                    "page_start": parent_chunk.page_start if parent_chunk else 0,
                    "page_end": parent_chunk.page_end if parent_chunk else 0,
                    "sentence_start": subchunk.sentence_start,
                    "sentence_end": subchunk.sentence_end
                })
            
            # ChromaDB에 저장 (문서별 고유 ID 사용)
            collection.add(
                documents=batch_texts,
                embeddings=embeddings,
                metadatas=metadata_list,
                ids=[f"{document_id}_{subchunk.subchunk_id}" for subchunk in batch_subchunks]
            )
        
        # 문서 정보를 레지스트리에 저장
        documents_registry[document_id] = {
            "document_id": document_id,
            "filename": file.filename,
            "upload_time": time.time(),
            "main_chunks_count": len(main_chunks),
            "subchunks_count": len(all_subchunks),
            "total_pages": len(pages_data),
            "chunks_info": [
                {
                    "chunk_id": chunk.chunk_id,
                    "title": chunk.title,
                    "section": chunk.section,
                    "pages": f"{chunk.page_start}-{chunk.page_end}",
                    "subchunks_count": len(chunk.subchunks),
                    "text_preview": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text
                }
                for chunk in main_chunks
            ]
        }
        
        # 임시 파일 삭제
        os.unlink(tmp_file_path)
        
        logger.info(f"PDF 처리 완료: {len(main_chunks)}개 메인청크, {len(all_subchunks)}개 서브청크 생성")
        
        return JSONResponse({
            "message": "PDF 업로드 및 처리 완료",
            "document_id": document_id,
            "collection_name": collection_name,
            "main_chunks_count": len(main_chunks),
            "subchunks_count": len(all_subchunks),
            "total_pages": len(pages_data),
            "chunks_info": documents_registry[document_id]["chunks_info"]
        })
    
    except Exception as e:
        logger.error(f"PDF 처리 중 오류: {e}")
        if 'tmp_file_path' in locals():
            try:
                os.unlink(tmp_file_path)
            except:
                pass
        raise HTTPException(status_code=500, detail=f"PDF 처리 실패: {str(e)}")

async def hybrid_search(query: str, document_id: str = None, vector_weight: float = 0.7, keyword_weight: float = 0.3, top_k: int = 5) -> List[Dict[str, Any]]:
    """하이브리드 검색 (벡터 + 키워드)를 수행합니다."""
    if not collection:
        raise HTTPException(status_code=400, detail="먼저 PDF를 업로드해주세요.")
    
    # 쿼리 임베딩 생성
    query_embedding = await create_embeddings_batch([query])
    query_embedding = query_embedding[0]
    
    # 벡터 검색 (ChromaDB에서 직접)
    if document_id:
        # 특정 문서에서만 검색
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, 100),
            where={"document_id": document_id}
        )
    else:
        # 모든 문서에서 검색
        vector_results = collection.query(
            query_embeddings=[query_embedding],
            n_results=min(top_k * 3, 100)
        )
    
    if not vector_results['ids'][0]:
        return []
    
    # BM25 검색을 위한 준비
    query_tokens = re.findall(r'\b\w+\b', query.lower())
    
    # 검색 결과 처리
    search_results = []
    
    for i, (vector_id, distance, metadata, document_text) in enumerate(zip(
        vector_results['ids'][0],
        vector_results['distances'][0], 
        vector_results['metadatas'][0],
        vector_results['documents'][0]
    )):
        # 벡터 점수 (거리를 유사도로 변환)
        vector_score = 1 - distance
        
        # BM25 점수 계산 (해당 문서의 BM25 인덱스 사용)
        doc_id = metadata.get('document_id', '')
        bm25_score = 0.0
        
        if doc_id in documents_bm25 and document_text:
            # 문서 텍스트를 토큰화
            doc_tokens = re.findall(r'\b\w+\b', document_text.lower())
            
            # BM25 점수 계산 (간단한 TF-IDF 기반)
            query_term_scores = []
            for term in query_tokens:
                if term in doc_tokens:
                    tf = doc_tokens.count(term)
                    # 간단한 BM25 근사
                    query_term_scores.append(tf / (tf + 1.0))
            
            if query_term_scores:
                bm25_score = sum(query_term_scores) / len(query_term_scores)
        
        # 하이브리드 점수 계산
        hybrid_score = (vector_weight * vector_score) + (keyword_weight * bm25_score)
        
        # 부모 청크 정보 찾기
        parent_chunk = None
        if doc_id in documents_chunks:
            parent_chunk_id = metadata.get('parent_chunk_id', '')
            for chunk in documents_chunks[doc_id]['main_chunks']:
                if chunk.chunk_id == parent_chunk_id:
                    parent_chunk = chunk
                    break
        
        # 서브청크 정보 찾기
        subchunk = None
        if doc_id in documents_chunks:
            subchunk_id = metadata.get('subchunk_id', '')
            for sc in documents_chunks[doc_id]['subchunks']:
                if sc.subchunk_id == subchunk_id:
                    subchunk = sc
                    break
        
        # 서브청크가 없으면 임시로 생성
        if not subchunk:
            subchunk = SubChunk(
                text=document_text,
                subchunk_id=metadata.get('subchunk_id', f'temp_{i}'),
                parent_chunk_id=metadata.get('parent_chunk_id', ''),
                sentence_start=0,
                sentence_end=0
            )
        
        search_results.append({
            'subchunk': subchunk,
            'parent_chunk': parent_chunk,
            'score': hybrid_score,
            'vector_score': vector_score,
            'bm25_score': bm25_score,
            'document_id': doc_id,
            'metadata': metadata
        })
    
    # 점수 기준으로 정렬
    search_results.sort(key=lambda x: x['score'], reverse=True)
    
    return search_results[:top_k]

@app.post("/question")
async def ask_question(request: QuestionRequest):
    """사용자 질문에 답변합니다."""
    try:
        # 하이브리드 검색 수행
        search_results = await hybrid_search(request.question, request.document_id, top_k=5)
        
        if not search_results:
            raise HTTPException(status_code=404, detail="관련 정보를 찾을 수 없습니다.")
        
        # 검색된 서브청크들과 해당하는 부모 청크 정보 수집
        context_info = []
        used_parent_chunks = set()
        
        for result in search_results:
            subchunk = result['subchunk']
            parent_chunk = result['parent_chunk']
            
            # 부모 청크의 전체 컨텍스트 사용
            if parent_chunk and parent_chunk.chunk_id not in used_parent_chunks:
                context_info.append({
                    'parent_text': parent_chunk.text,
                    'subchunk_text': subchunk.text,
                    'title': parent_chunk.title,
                    'section': parent_chunk.section,
                    'page_start': parent_chunk.page_start,
                    'page_end': parent_chunk.page_end,
                    'score': result['score'],
                    'document_id': result['document_id']
                })
                used_parent_chunks.add(parent_chunk.chunk_id)
            elif not parent_chunk:
                # 부모 청크가 없는 경우 서브청크만 사용
                context_info.append({
                    'parent_text': subchunk.text,
                    'subchunk_text': subchunk.text,
                    'title': result['metadata'].get('parent_title', '제목 없음'),
                    'section': result['metadata'].get('parent_section', '섹션 없음'),
                    'page_start': result['metadata'].get('page_start', 0),
                    'page_end': result['metadata'].get('page_end', 0),
                    'score': result['score'],
                    'document_id': result['document_id']
                })
        
        # 컨텍스트 구성 (부모 청크의 전체 텍스트 사용)
        context_parts = []
        for info in context_info:
            context_part = f"[{info['title']} - 페이지 {info['page_start']}"
            if info['page_end'] != info['page_start']:
                context_part += f"-{info['page_end']}"
            context_part += f", 섹션: {info['section']}]\n{info['parent_text']}"
            context_parts.append(context_part)
        
        context = "\n\n" + "="*50 + "\n\n".join(context_parts)
        
        # LLM 프롬프트 구성
        prompt = f"""당신은 업로드된 PDF 문서를 기반으로 질문에 답변하는 AI 어시스턴트입니다.

다음 문서의 관련 섹션들을 참고하여 사용자의 질문에 정확하고 도움이 되는 답변을 제공해주세요:

=== 관련 문서 섹션들 ===
{context}

=== 사용자 질문 ===
{request.question}

=== 답변 지침 ===
1. 제공된 문서 내용만을 기반으로 답변하세요
2. 문서에 없는 내용은 추측하지 마세요
3. 관련 섹션과 페이지 번호를 언급해주세요
4. 여러 섹션의 정보를 종합하여 답변하세요
5. 답변이 불분명하다면 그 이유를 설명해주세요
6. 한국어로 자연스럽고 이해하기 쉽게 답변해주세요

답변:"""

        # Gemini API 호출
        response = llm.invoke(prompt)
        answer = response.content
        
        return JSONResponse({
            "question": request.question,
            "answer": answer,
            "context_sources": [
                {
                    "document_id": info['document_id'],
                    "title": info['title'],
                    "section": info['section'],
                    "pages": f"{info['page_start']}-{info['page_end']}" if info['page_end'] != info['page_start'] else str(info['page_start']),
                    "score": round(info['score'], 3),
                    "matched_subchunk": info['subchunk_text'][:200] + "..." if len(info['subchunk_text']) > 200 else info['subchunk_text'],
                    "full_section_preview": info['parent_text'][:300] + "..." if len(info['parent_text']) > 300 else info['parent_text']
                }
                for info in context_info
            ],
            "total_sections_found": len(context_info)
        })
    
    except Exception as e:
        logger.error(f"질문 처리 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"질문 처리 실패: {str(e)}")

@app.delete("/delete-document/{document_id}")
async def delete_document(document_id: str):
    """특정 문서와 관련된 데이터를 삭제합니다."""
    global collection, documents_registry, documents_chunks, documents_bm25
    
    try:
        deleted_items = []
        
        # 1. 문서 레지스트리에서 삭제
        if document_id in documents_registry:
            del documents_registry[document_id]
            deleted_items.append("문서 레지스트리 정보")
        
        # 2. 문서별 청크 데이터 삭제
        if document_id in documents_chunks:
            chunk_count = len(documents_chunks[document_id]['main_chunks'])
            subchunk_count = len(documents_chunks[document_id]['subchunks'])
            del documents_chunks[document_id]
            deleted_items.append(f"메인 청크 데이터 ({chunk_count}개)")
            deleted_items.append(f"서브 청크 데이터 ({subchunk_count}개)")
        
        # 3. 문서별 BM25 인덱스 삭제
        if document_id in documents_bm25:
            del documents_bm25[document_id]
            deleted_items.append("BM25 인덱스")
        
        # 4. ChromaDB에서 해당 문서의 데이터만 삭제
        try:
            if collection:
                # 해당 문서의 모든 벡터 데이터 삭제
                collection.delete(where={"document_id": document_id})
                deleted_items.append(f"벡터 데이터 (document_id: {document_id})")
        except Exception as e:
            logger.warning(f"ChromaDB에서 문서 삭제 중 오류: {e}")
        
        logger.info(f"문서 삭제 완료: {document_id}, 삭제된 항목: {deleted_items}")
        
        return JSONResponse({
            "message": f"문서 '{document_id}' 삭제 완료",
            "document_id": document_id,
            "deleted_items": deleted_items,
            "timestamp": time.time()
        })
    
    except Exception as e:
        logger.error(f"문서 삭제 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서 삭제 실패: {str(e)}")

@app.delete("/delete-all")
async def delete_all_documents():
    """모든 문서와 데이터를 삭제합니다."""
    global collection, documents_registry, documents_chunks, documents_bm25
    
    try:
        deleted_items = []
        
        # 1. 문서 레지스트리 초기화
        doc_count = len(documents_registry)
        documents_registry.clear()
        deleted_items.append(f"문서 레지스트리 ({doc_count}개 문서)")
        
        # 2. 문서별 청크 데이터 초기화
        chunk_doc_count = len(documents_chunks)
        documents_chunks.clear()
        deleted_items.append(f"문서별 청크 데이터 ({chunk_doc_count}개 문서)")
        
        # 3. 문서별 BM25 인덱스 초기화
        bm25_doc_count = len(documents_bm25)
        documents_bm25.clear()
        deleted_items.append(f"문서별 BM25 인덱스 ({bm25_doc_count}개 문서)")
        
        # 4. ChromaDB rag 컬렉션 초기화
        try:
            if collection:
                # 컬렉션의 모든 데이터 삭제
                all_data = collection.get()
                if all_data['ids']:
                    collection.delete(ids=all_data['ids'])
                    deleted_items.append(f"rag 컬렉션 데이터 ({len(all_data['ids'])}개)")
        except Exception as e:
            logger.warning(f"ChromaDB 컬렉션 초기화 중 오류: {e}")
        
        # 5. 임시 파일들 정리
        temp_dir = tempfile.gettempdir()
        temp_files_deleted = 0
        try:
            for filename in os.listdir(temp_dir):
                if filename.endswith('.pdf') and 'tmp' in filename:
                    temp_file_path = os.path.join(temp_dir, filename)
                    try:
                        os.unlink(temp_file_path)
                        temp_files_deleted += 1
                    except:
                        pass
            if temp_files_deleted > 0:
                deleted_items.append(f"임시 PDF 파일들 ({temp_files_deleted}개)")
        except Exception as e:
            logger.warning(f"임시 파일들 정리 중 오류: {e}")
        
        logger.info(f"모든 문서 삭제 완료, 삭제된 항목: {deleted_items}")
        
        return JSONResponse({
            "message": "모든 문서와 데이터 삭제 완료",
            "deleted_items": deleted_items,
            "timestamp": time.time()
        })
    
    except Exception as e:
        logger.error(f"전체 삭제 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"전체 삭제 실패: {str(e)}")

@app.get("/documents")
async def list_documents():
    """현재 시스템에 등록된 모든 문서 목록을 조회합니다."""
    try:
        # ChromaDB에서 실제 저장된 문서 수 확인
        vector_count = 0
        unique_documents = set()
        
        if collection:
            try:
                all_data = collection.get()
                vector_count = len(all_data['ids']) if all_data['ids'] else 0
                
                # 문서별 벡터 수 계산
                for metadata in all_data['metadatas'] or []:
                    if 'document_id' in metadata:
                        unique_documents.add(metadata['document_id'])
                        
            except Exception as e:
                logger.warning(f"ChromaDB 데이터 조회 중 오류: {e}")
        
        # 문서 레지스트리 정보와 실제 벡터DB 정보 결합
        documents_list = []
        for doc_id, doc_info in documents_registry.items():
            # 실제 벡터DB에서 해당 문서의 벡터 수 확인
            actual_vectors = 0
            if collection:
                try:
                    doc_vectors = collection.get(where={"document_id": doc_id})
                    actual_vectors = len(doc_vectors['ids']) if doc_vectors['ids'] else 0
                except:
                    pass
            
            documents_list.append({
                **doc_info,
                "actual_vectors_count": actual_vectors,
                "upload_time_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(doc_info['upload_time']))
            })
        
        return JSONResponse({
            "total_documents": len(documents_registry),
            "total_vectors": vector_count,
            "unique_documents_in_db": len(unique_documents),
            "collection_name": "rag",
            "current_memory_status": {
                "documents_chunks_loaded": len(documents_chunks),
                "documents_bm25_loaded": len(documents_bm25),
                "vector_db_ready": collection is not None
            },
            "documents": documents_list
        })
    
    except Exception as e:
        logger.error(f"문서 목록 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서 목록 조회 실패: {str(e)}")

@app.get("/document/{document_id}")
async def get_document_detail(document_id: str):
    """특정 문서의 상세 정보를 조회합니다."""
    try:
        if document_id not in documents_registry:
            raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없습니다: {document_id}")
        
        doc_info = documents_registry[document_id].copy()
        
        # ChromaDB에서 실제 벡터 데이터 확인
        actual_vectors = 0
        vector_samples = []
        
        if collection:
            try:
                doc_vectors = collection.get(
                    where={"document_id": document_id},
                    limit=5  # 샘플 5개만 가져오기
                )
                actual_vectors = len(doc_vectors['ids']) if doc_vectors['ids'] else 0
                
                # 벡터 샘플 정보
                if doc_vectors['documents']:
                    for i, (doc_text, metadata) in enumerate(zip(doc_vectors['documents'], doc_vectors['metadatas'] or [])):
                        vector_samples.append({
                            "sample_id": i + 1,
                            "text_preview": doc_text[:150] + "..." if len(doc_text) > 150 else doc_text,
                            "parent_section": metadata.get('parent_section', ''),
                            "page_range": f"{metadata.get('page_start', '')}-{metadata.get('page_end', '')}"
                        })
                        
            except Exception as e:
                logger.warning(f"문서 벡터 데이터 조회 중 오류: {e}")
        
        doc_info.update({
            "actual_vectors_count": actual_vectors,
            "upload_time_formatted": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(doc_info['upload_time'])),
            "vector_samples": vector_samples
        })
        
        return JSONResponse(doc_info)
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"문서 상세 조회 중 오류: {e}")
        raise HTTPException(status_code=500, detail=f"문서 상세 조회 실패: {str(e)}")

@app.get("/")
async def root():
    """API 상태 확인"""
    return {
        "message": "PDF QA System API",
        "status": "running",
        "endpoints": {
            "upload": "/upload-pdf",
            "question": "/question",
            "delete_document": "/delete-document/{document_id}",
            "delete_all": "/delete-all",
            "list_documents": "/documents",
            "document_detail": "/document/{document_id}",
            "status": "/status"
        }
    }

@app.get("/status")
async def get_status():
    """현재 시스템 상태 확인"""
    total_main_chunks = sum(len(data['main_chunks']) for data in documents_chunks.values())
    total_subchunks = sum(len(data['subchunks']) for data in documents_chunks.values())
    
    return {
        "total_documents": len(documents_registry),
        "documents_chunks_loaded": len(documents_chunks),
        "documents_bm25_loaded": len(documents_bm25),
        "total_main_chunks": total_main_chunks,
        "total_subchunks": total_subchunks,
        "vector_db_ready": collection is not None,
        "documents_info": [
            {
                "document_id": doc_id,
                "filename": doc_info["filename"],
                "main_chunks": len(documents_chunks[doc_id]['main_chunks']) if doc_id in documents_chunks else 0,
                "subchunks": len(documents_chunks[doc_id]['subchunks']) if doc_id in documents_chunks else 0
            }
            for doc_id, doc_info in documents_registry.items()
        ]
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
