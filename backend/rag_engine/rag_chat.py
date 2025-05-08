import os
import json
import time
import requests
import numpy as np
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.schema import BaseRetriever
from typing import List, Dict, Any, Optional

from .vector_store import FaceLogVectorStore
from ..utils.logger import setup_logger

logger = setup_logger('rag_chat')

class SimpleEmbeddings(Embeddings):
    """Simple embedding class for testing or placeholder"""
    def __init__(self, vector_dim=384):
        self.vector_dim = vector_dim
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Generate embeddings for a list of documents"""
        return [self.embed_query(text) for text in texts]
    
    def embed_query(self, text: str) -> List[float]:
        """Generate embeddings for a query"""
        # Simple deterministic embedding based on hash of text
        # Just for testing - not suitable for production
        np.random.seed(hash(text) % 2**32)
        return np.random.rand(self.vector_dim).astype('float32').tolist()

class OpenRouterAPI:
    """Class for interacting with OpenRouter API"""
    def __init__(self, api_key=None):
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY", "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")
        self.api_url = "https://openrouter.ai/api/v1/chat/completions"
        self.embedding_url = "https://openrouter.ai/api/v1/embeddings"
    
    def generate_chat_completion(self, messages, model="deepseek/deepseek-prover-v2:free"):
        """Generate a chat completion using OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "messages": messages
        }
        
        try:
            logger.info(f"Sending request to OpenRouter: {len(messages)} messages")
            response = requests.post(self.api_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            logger.info(f"Received response from OpenRouter: {len(result['choices'])}")
            
            return result["choices"][0]["message"]["content"]
        
        except Exception as e:
            logger.error(f"Error calling OpenRouter API: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"
    
    def generate_embeddings(self, texts, model="openai/text-embedding-ada-002"):
        """Generate embeddings using OpenRouter API"""
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": model,
            "input": texts if isinstance(texts, list) else [texts]
        }
        
        try:
            response = requests.post(self.embedding_url, json=payload, headers=headers)
            response.raise_for_status()
            
            result = response.json()
            return [item["embedding"] for item in result["data"]]
        
        except Exception as e:
            logger.error(f"Error generating embeddings: {str(e)}")
            # Fall back to simple embeddings if OpenRouter fails
            simple_embeddings = SimpleEmbeddings()
            return simple_embeddings.embed_documents(texts if isinstance(texts, list) else [texts])

"""Retriever for face recognition logs"""
class FaceLogRetriever(BaseRetriever):
    vector_store: FaceLogVectorStore = None
    embeddings_provider: Any = None
    
    def __init__(self, vector_store: FaceLogVectorStore, embeddings_provider):
        super().__init__()
        object.__setattr__(self, "vector_store", vector_store)
        object.__setattr__(self, "embeddings_provider", embeddings_provider)
    
    def get_relevant_documents(self, query: str) -> List[Document]:
        """Get relevant documents based on query"""
        # Generate embedding for query
        try:
            if hasattr(self.embeddings_provider, 'embed_query'):
                query_embedding = self.embeddings_provider.embed_query(query)
            else:
                # Assume it's the OpenRouterAPI class
                query_embedding = self.embeddings_provider.generate_embeddings(query)[0]
            
            # Search vector store
            results = self.vector_store.search(query_embedding, top_k=5)
            
            # Convert to LangChain documents
            documents = []
            for result in results:
                metadata = result["metadata"]
                # Create a document
                doc = Document(
                    page_content=result["text"],
                    metadata=metadata
                )
                documents.append(doc)
            
            return documents
        
        except Exception as e:
            logger.error(f"Error retrieving documents: {str(e)}")
            return []
    
    async def aget_relevant_documents(self, query: str) -> List[Document]:
        """Async version of get_relevant_documents"""
        return self.get_relevant_documents(query)

class RAGChatEngine:
    """RAG-based chat engine for querying face activity"""
    def __init__(self):
        # Initialize OpenRouter API
        self.openrouter = OpenRouterAPI()
        
        # Initialize vector store
        self.vector_store = FaceLogVectorStore()
        
        # Initialize retriever
        self.retriever = FaceLogRetriever(self.vector_store, self.openrouter)
        
        # System message
        self.system_message = """
        You are an AI assistant for a face recognition system. You can answer questions about 
        face registration and recognition events based on the logs provided.
        
        When answering questions:
        1. Only use the information provided in the context.
        2. If the information is not in the context, say you don't have that information.
        3. Be concise and accurate in your responses.
        4. Format dates and times in a readable format.
        5. If asked about security concerns, emphasize proper use and privacy considerations.
        """
    
    def update_vector_store(self):
        """Update the vector store with new logs"""
        # Use OpenRouter embeddings for actual implementation
        return self.vector_store.update_vector_store(
            embedding_fn=lambda text: self.openrouter.generate_embeddings(text)[0]
        )
    
    def create_prompt_with_context(self, query, context_docs):
        """Create a prompt with context for the LLM"""
        context_text = "\n\n".join([doc.page_content for doc in context_docs])
        
        prompt = f"""
        Context information from face recognition logs:
        {context_text}
        
        Based only on the above context, answer the following question:
        {query}
        """
        
        return prompt
    
    def chat(self, query, chat_history=None):
        """
        Process a query using RAG approach
        
        Args:
            query: User query string
            chat_history: List of previous messages (optional)
            
        Returns:
            str: Response text
        """
        try:
            logger.info(f"Received query: {query}")
            
            # Update vector store to include latest logs
            self.update_vector_store()
            
            # Retrieve relevant documents
            context_docs = self.retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(context_docs)} relevant documents")
            
            # Create messages array for OpenRouter
            messages = []
            
            # Add system message
            messages.append({
                "role": "system",
                "content": self.system_message
            })
            
            # Add chat history if provided
            if chat_history:
                for msg in chat_history:
                    messages.append(msg)
            
            # Add retrieved context and query
            prompt = self.create_prompt_with_context(query, context_docs)
            
            messages.append({
                "role": "user",
                "content": prompt
            })
            
            # Generate response
            response = self.openrouter.generate_chat_completion(messages)
            
            logger.info(f"Generated response: {response[:100]}...")
            return response
        
        except Exception as e:
            logger.error(f"Error in RAG chat: {str(e)}")
            return f"Sorry, I encountered an error: {str(e)}"

# For testing
if __name__ == "__main__":
    rag_engine = RAGChatEngine()
    response = rag_engine.chat("Who was recognized recently?")
    print(f"Response: {response}") 












