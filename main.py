import os
from dotenv import load_dotenv
import torch
from src.chat_engine.rag_pipeline import RAGPipeline
from src.ui_interface.main_ui import main_ui
from src.vector_store.chroma_client import create_chroma_client
from src.vector_store.embeddings import create_embedding_generator

from typing import cast

try:
    from langchain_community.vectorstores import Chroma
except ImportError:
    from langchain.vectorstores import Chroma

EMBEDDING_MODEL = "google/embeddinggemma-300m"


def get_device():
    # Determine best available device (MPS > CUDA > CPU)
    if torch.backends.mps.is_available():
        device = torch.device("mps")  # Mac M-series GPU
    elif torch.cuda.is_available():
        device = torch.device("cuda")  # NVIDIA GPU
    else:
        device = torch.device("cpu")   # CPU fallback
    return device

def build_rag_chain() -> RAGPipeline:
    device = get_device()

    embedding = create_embedding_generator(model_name = EMBEDDING_MODEL,
                                           device=device)
    vector_db_client = create_chroma_client(persist_directory="data/vector_db",
                                     collection_name="edu_content",
                                     embedding_function=embedding)
    if not vector_db_client.is_connected():        
        vector_db_client.connect()

    c_info = vector_db_client.get_collection_info()
    print(c_info)

    vector_db= vector_db_client.get_vectorstore()
    print(f"coll count: {len(vector_db)}")

    pipeline = RAGPipeline(
        llm_model="gpt-3.5-turbo",
        llm_temperature=0.7,
        vector_store=vector_db
    )

    # # 질문 처리
    # result = pipeline.generate_answer(
    #     question="머신러닝의 기본 개념은?",
    #     use_rag=True
    # )

    # print(result.answer)
    return pipeline

def excute_ui():    
    pipeline = build_rag_chain()
    main_ui(pipeline)


def initialize_vector_db(force_rebuild=False, forDevel=False):
    """벡터 DB 초기화"""
    import os
    import glob
    from pathlib import Path
    from src.document_processor.core import DocumentProcessorFactory


    print("🚀 벡터 DB 초기화 중...")

    try:
        # 디바이스 설정
        device = get_device()
        print(f"📱 디바이스: {device}")

        # LangChain 호환 임베딩 생성기 초기화
        embedding = create_embedding_generator(model_name = EMBEDDING_MODEL,
                                        device=str(device))

        # 벡터 DB 클라이언트 초기화
        vector_db_client = create_chroma_client(
            persist_directory="data/vector_db",
            collection_name="edu_content",
            embedding_function=embedding
        )

        # 연결
        if not vector_db_client.connect():
            print("❌ 벡터 DB 연결 실패")
            return False

        # 강제 재구축인 경우 컬렉션 초기화
        if force_rebuild:
            print("🔄 기존 컬렉션 재설정...")
            if not vector_db_client.reset_collection():
                print("❌ 컬렉션 재설정 실패")
                return False

        # 교육 자료 디렉토리 설정
        materials_dir = Path("data/educational_materials")
        pdf_dir = materials_dir / "pdfs"
        notebook_dir = materials_dir / "notebooks"

        if forDevel: # 처리할 파일 수집 (PDF는 5개만, 노트북도 20개만)
            pdf_files = list(glob.glob(str(pdf_dir / "*.pdf")))[:5] if pdf_dir.exists() else []
            notebook_files = list(glob.glob(str(notebook_dir / "*.ipynb")))[:20] if notebook_dir.exists() else []
        else:
            pdf_files = list(glob.glob(str(pdf_dir / "*.pdf"))) if pdf_dir.exists() else []
            notebook_files = list(glob.glob(str(notebook_dir / "*.ipynb"))) if notebook_dir.exists() else []
        all_files = pdf_files + notebook_files
        total_files = len(all_files)

        if total_files == 0:
            print("⚠️ 처리할 교육 자료가 없습니다.")
            return False

        print(f"📄 총 {total_files}개 파일 처리 예정:")
        print(f"   - PDF: {len(pdf_files)}개")
        print(f"   - Notebook: {len(notebook_files)}개")
        print("="*50)

        # 벡터스토어 가져오기
        vectorstore = vector_db_client.get_vectorstore()
        if vectorstore == None:
            print("Vector store 가져오기 실패")
            return False
        vectorstore = cast(Chroma, vectorstore)
        
        processed_count = 0
        failed_count = 0
        total_chunks = 0

        # 각 파일 처리
        for file_path in all_files:
            file_name = Path(file_path).name
            print(f"📖 처리 중: {file_name}")

            try:
                # 적절한 프로세서 생성
                processor = DocumentProcessorFactory.create_processor(file_path)

                # 파일 처리
                result = processor.process_file(file_path)

                if result.success:
                    chunks = result.chunks
                    chunk_count = len(chunks)

                    if chunk_count > 0:
                        # 청크들을 벡터 DB에 추가
                        texts = [chunk.content for chunk in chunks]
                        metadatas = [chunk.metadata for chunk in chunks]
                        ids = [f"{file_name}_{chunk.source_location}" for chunk in chunks]

                        # 벡터스토어에 추가
                        vectorstore.add_texts(
                            texts=texts,
                            metadatas=metadatas,
                            ids=ids
                        )

                        processed_count += 1
                        total_chunks += chunk_count
                        print(f"   ✅ 성공: {chunk_count}개 청크 생성")
                    else:
                        print(f"   ⚠️ 경고: 청크가 생성되지 않음")
                        failed_count += 1
                else:
                    print(f"   ❌ 실패: {result.error_message}")
                    failed_count += 1

            except Exception as e:
                print(f"   ❌ 오류: {str(e)}")
                failed_count += 1

        # 벡터스토어 저장
        vectorstore.persist()

        # 결과 출력
        print("="*50)
        print(f"🎯 처리 완료!")
        print(f"   - 성공: {processed_count}개 파일")
        print(f"   - 실패: {failed_count}개 파일")
        print(f"   - 총 청크: {total_chunks}개")

        # 연결 해제
        vector_db_client.disconnect()

        return processed_count > 0

    except Exception as e:
        print(f"❌ 벡터 DB 초기화 실패: {str(e)}")
        return False

load_dotenv()

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="고등학생 학과 선택 도우미 - 환경 설정")
    parser.add_argument("--init-db", action="store_true", 
                       help="벡터 DB 초기화만 실행")
    parser.add_argument("--force-rebuild", action="store_true",
                       help="기존 DB 삭제 후 강제 재구축")
    parser.add_argument("--setup-only", action="store_true",
                       help="DB 초기화 없이 환경 설정만 실행")
    
    args = parser.parse_args()
    
    print("=" * 60)
    
    # DB 초기화만 실행하는 경우
    if args.init_db:
        print("🚀 벡터 DB 초기화 모드")
        print("-" * 40)
                    
        # 벡터 DB 초기화 실행
        success = initialize_vector_db(force_rebuild=args.force_rebuild, forDevel=True)
        
        if success:
            print("\n🎉 벡터 DB 초기화가 완료되었습니다!")
        else:
            print("\n❌ 벡터 DB 초기화에 실패했습니다.")
            
        return True

    excute_ui()
    
if __name__ == "__main__":
    main()