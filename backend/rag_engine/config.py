import os

class RAGConfig:
    """Configuration for the RAG engine"""
    
    # OpenRouter API Key
    OPENROUTER_API_KEY = os.environ.get(
        "OPENROUTER_API_KEY", 
        "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"
    )
    
    # OpenRouter Model
    OPENROUTER_MODEL = "deepseek/deepseek-prover-v2:free"
    
    # OpenRouter Embedding Model
    OPENROUTER_EMBEDDING_MODEL = "openai/text-embedding-ada-002"
    
    # Vector dimension
    VECTOR_DIM = 384
    
    # Database paths
    FACE_DB_PATH = "database/faces.db"
    LOG_DB_PATH = "database/logs.db"
    
    # Vector store paths
    VECTOR_STORE_PATH = "database/face_vectors"
    LOGS_EMBEDDING_PATH = "database/logs_embeddings.json"
    
    # RAG settings
    MAX_CONTEXT_DOCS = 5
    
    # System prompt
    SYSTEM_PROMPT = """
    You are an AI assistant for a face recognition system. You can answer questions about 
    face registration and recognition events based on the logs provided.
    
    When answering questions:
    1. Only use the information provided in the context.
    2. If the information is not in the context, say you don't have that information.
    3. Be concise and accurate in your responses.
    4. Format dates and times in a readable format.
    5. If asked about security concerns, emphasize proper use and privacy considerations.
    """ 