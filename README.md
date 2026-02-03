# 📂 AI Data Analyst Chatbot (Gemini RAG)

**Google Gemini API**를 활용하여 사용자가 업로드한 CSV/Excel 데이터를 분석하고, 자연어로 대화할 수 있는 **RAG(Retrieval-Augmented Generation) 기반 AI 챗봇**입니다.

Streamlit으로 구축된 직관적인 UI를 제공하며, FAISS 벡터 DB를 활용하여 데이터에 근거한 정확한 답변을 제공합니다.

---

## ✨ 주요 기능 (Features)

- **📂 범용 파일 지원**: `.csv` 및 `.xlsx` (Excel) 파일을 업로드하면 자동으로 내용을 파싱하고 벡터화합니다.
- **🧠 Google Gemini 1.5 Flash**: 최신 Google LLM을 사용하여 빠르고 비용 효율적인 답변을 생성합니다.
- **🔍 RAG (검색 증강 생성)**: 사용자의 질문과 가장 관련성 높은 데이터를 FAISS 벡터 저장소에서 찾아 답변에 활용합니다.
- **⚡ 대용량 처리 최적화**: API Rate Limit(사용량 제한)을 고려한 배치 처리 및 진행률 표시(Progress Bar) 기능을 탑재했습니다.
- **💬 대화형 인터페이스**: 이전 대화 맥락을 기억하여 자연스러운 꼬리 물기 질문이 가능합니다.
- **🛡️ 보안 강화**: API Key를 `.env` 환경 변수로 관리하여 코드 유출 시에도 키를 보호합니다.

---

## 🛠️ 기술 스택 (Tech Stack)

- **Language**: Python 3.10+
- **Framework**: [Streamlit](https://streamlit.io/)
- **LLM & Embedding**: [Google Gemini API (LangChain Google GenAI)](https://ai.google.dev/)
- **Vector DB**: [FAISS (Facebook AI Similarity Search)](https://github.com/facebookresearch/faiss)
- **Data Processing**: Pandas, Openpyxl