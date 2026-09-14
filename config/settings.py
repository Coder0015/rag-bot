import os

from dotenv import load_dotenv

from utils.path_tool import get_project_path

PROJECT_PATH = get_project_path()

load_dotenv(os.path.join(PROJECT_PATH, ".env"))


class Settings:
    project_path = PROJECT_PATH

    deepseek_api_key = os.getenv("DEEPSEEK_API_KEY", "") or ""
    deepseek_model = os.getenv("DEEPSEEK_MODEL") or "deepseek-flash"

    ollama_base_url = os.getenv("OLLAMA_BASE_URL") or "http://localhost:11434"
    ollama_chat_model = os.getenv("OLLAMA_CHAT_MODEL") or ""
    ollama_embedding_model = os.getenv("OLLAMA_EMBEDDING_MODEL") or ""

    chroma_path = os.getenv("CHROMA_PATH", os.path.join(PROJECT_PATH)) or os.path.join(PROJECT_PATH, "chroma_db")
    collection_name = os.getenv("COLLECTION_NAME") or "knowledge_base"
    data_dir = os.getenv("DATA_DIR") or os.path.join(PROJECT_PATH, "data")
    embedding_model_path = os.getenv("EMBEDDING_MODEL_PATH") or ""
    reranker_path = os.getenv("RERANKER_PATH") or ""


settings = Settings()
if __name__ == '__main__':
    print(settings.chroma_path)