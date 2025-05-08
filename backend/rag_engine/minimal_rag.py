"""
Minimal RAG engine that doesn't use Pydantic models
This is a fallback to allow the face recognition system to work
"""

import logging

logger = logging.getLogger('minimal_rag')

class MinimalRAGEngine:
    """Minimal RAG engine for testing purposes"""
    
    def __init__(self):
        logger.info("Initializing minimal RAG engine")
    
    def chat(self, query, chat_history=None):
        """
        Simple chat method that just returns a predefined response
        
        Args:
            query: User query string
            chat_history: List of previous messages (optional)
            
        Returns:
            str: Response text
        """
        logger.info(f"Received query: {query}")
        
        return "This is a minimal RAG engine response. The full RAG engine is disabled for testing." 