import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
KNOWLEDGE_BASE_DIR = os.getenv("KNOWLEDGE_BASE_DIR", "knowledge_base")
MAX_AGENT_ITERATIONS = int(os.getenv("MAX_AGENT_ITERATIONS", "10"))
TOP_K_RESULTS = int(os.getenv("TOP_K_RESULTS", "3"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
