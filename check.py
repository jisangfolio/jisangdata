import google.generativeai as genai
import os
from dotenv import load_dotenv

# 1. API 키 설정 (본인의 키를 직접 넣거나 .env에서 로드)
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

if not api_key:
    # .env가 없으면 여기에 직접 키를 입력해서 테스트해보세요
    api_key = "여기에_님의_API_키를_넣으세요"

genai.configure(api_key=api_key)

print("🔍 사용 가능한 임베딩 모델을 찾는 중...\n")

try:
    # 2. 모든 모델 목록 가져오기
    count = 0
    for m in genai.list_models():
        # 'embedContent' 기능을 지원하는 모델만 필터링
        if 'embedContent' in m.supported_generation_methods:
            print(f"✅ 모델명: {m.name}")
            print(f"   - 설명: {m.description}")
            print(f"   - 버전: {m.version}")
            print("-" * 50)
            count += 1
    
    if count == 0:
        print("❌ 사용 가능한 임베딩 모델이 없습니다. API 키 권한을 확인하세요.")
    else:
        print(f"\n🎉 총 {count}개의 임베딩 모델을 찾았습니다.")
        print("위 목록에 있는 '모델명'을 rag.py의 EMBEDDING_MODEL 변수에 넣으세요.")

except Exception as e:
    print(f"❌ 에러 발생: {e}")