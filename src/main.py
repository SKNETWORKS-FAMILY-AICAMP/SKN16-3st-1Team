import os
import os
from dotenv import load_dotenv


def main_ui():
    print("main ui")


def initialize_vector_db(force_rebuild=False):
    """벡터 DB 초기화"""
    print("🚀 벡터 DB 초기화 중...")

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
        success = initialize_vector_db(force_rebuild=args.force_rebuild)
        
        if success:
            print("\n🎉 벡터 DB 초기화가 완료되었습니다!")
        else:
            print("\n❌ 벡터 DB 초기화에 실패했습니다.")
            
        return True

    main_ui()
    
if __name__ == "__main__":
    main()