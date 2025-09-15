#!/bin/bash

# 프로젝트 최상위 디렉토리에서 실행해주세요.

# src 하위 폴더들 생성
mkdir -p src/schema \
         src/document_processor \
         src/vector_store \
         src/retrieval_system \
         src/chat_engine \
         src/ui_interface \
         src/utils \
         src/config

# data 하위 폴더들 생성
mkdir -p data/vector_db \
         data/educational_materials/pdfs \
         data/educational_materials/notebooks \
         data/educational_materials/uploads



# 비어있는 디렉토리를 위한 .gitkeep 파일 생성
touch data/vector_db/.gitkeep \
      data/educational_materials/pdfs/.gitkeep \
      data/educational_materials/notebooks/.gitkeep \
      data/educational_materials/uploads/.gitkeep

echo "프로젝트 구조 생성이 완료되었습니다."
