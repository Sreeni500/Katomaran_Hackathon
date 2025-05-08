import os
import sqlite3
import numpy as np
import faiss
import json
from datetime import datetime
import random
from typing import List, Dict, Any, Optional, Tuple

class FaceLogVectorStore:
    def __init__(self, db_path='database/logs.db', vector_dim=384):
        """
        Initialize the Face Log Vector Store
        
        Args:
            db_path: Path to the log database
            vector_dim: Dimension of the vectors
        """
        self.db_path = db_path
        self.vector_dim = vector_dim
        self.vector_store_path = 'database/face_vectors'
        self.logs_embedding_path = 'database/logs_embeddings.json'
        
        # Create directories if they don't exist
        os.makedirs('database', exist_ok=True)
        os.makedirs(self.vector_store_path, exist_ok=True)
        
        # Initialize FAISS index
        self.index = faiss.IndexFlatL2(vector_dim)
        
        # Load existing embeddings or initialize empty
        self.logs_embeddings = self._load_embeddings()
    
    def _load_embeddings(self):
        """Load existing embeddings from file"""
        if os.path.exists(self.logs_embedding_path):
            try:
                with open(self.logs_embedding_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error loading embeddings: {str(e)}")
                return {}
        return {}
    
    def _save_embeddings(self):
        """Save embeddings to file"""
        try:
            with open(self.logs_embedding_path, 'w') as f:
                json.dump(self.logs_embeddings, f)
        except Exception as e:
            print(f"Error saving embeddings: {str(e)}")
    
    def _get_face_logs(self, start_date=None, end_date=None, action=None, name=None, limit=1000):
        """
        Get face logs from database
        
        Args:
            start_date: Start date for filtering (optional)
            end_date: End date for filtering (optional)
            action: Filter by action type (optional)
            name: Filter by person name (optional)
            limit: Maximum number of logs to return
            
        Returns:
            list: List of log entries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            query = "SELECT id, action, name, timestamp, details FROM face_logs"
            params = []
            
            where_clauses = []
            
            if start_date:
                where_clauses.append("timestamp >= ?")
                params.append(start_date)
            
            if end_date:
                where_clauses.append("timestamp <= ?")
                params.append(end_date)
            
            if action:
                where_clauses.append("action = ?")
                params.append(action)
            
            if name:
                where_clauses.append("name = ?")
                params.append(name)
            
            if where_clauses:
                query += " WHERE " + " AND ".join(where_clauses)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            logs = cursor.fetchall()
            
            # Convert to list of dictionaries
            logs_list = []
            for log in logs:
                log_dict = {
                    "id": log[0],
                    "action": log[1],
                    "name": log[2],
                    "timestamp": log[3],
                    "details": log[4]
                }
                logs_list.append(log_dict)
            
            return logs_list
        
        except Exception as e:
            print(f"Error getting face logs: {str(e)}")
            return []
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    def _log_to_text(self, log):
        """Convert a log entry to text for embedding"""
        action_desc = {
            "registration": "registered",
            "recognition": "was recognized",
            "deletion": "was deleted from the system"
        }
        
        action_text = action_desc.get(log["action"], log["action"])
        
        text = f"{log['name']} {action_text} on {log['timestamp']}"
        
        if log["details"]:
            text += f". Details: {log['details']}"
        
        return text
    
    def _create_random_embedding(self):
        """Create a random embedding for testing"""
        return np.random.rand(self.vector_dim).astype('float32')
    
    def _log_id_to_str(self, log_id):
        """Convert log ID to string identifier"""
        return f"log_{log_id}"
    
    def _str_to_log_id(self, id_str):
        """Convert string identifier to log ID"""
        if id_str.startswith("log_"):
            return int(id_str[4:])
        return -1
    
    def update_vector_store(self, embedding_fn=None):
        """
        Update the vector store with the latest logs
        
        Args:
            embedding_fn: Function to convert text to embedding vector
                        (if None, random embeddings will be used for testing)
        
        Returns:
            int: Number of new logs added
        """
        # Get all logs from database
        logs = self._get_face_logs()
        
        # Get IDs of logs we already have embeddings for
        existing_ids = set(self._str_to_log_id(id_str) for id_str in self.logs_embeddings.keys())
        
        # Filter to new logs
        new_logs = [log for log in logs if log["id"] not in existing_ids]
        
        if not new_logs:
            print("No new logs to add")
            return 0
        
        # Create vectors for new logs
        for log in new_logs:
            log_text = self._log_to_text(log)
            
            # Use provided embedding function or create random embedding for testing
            if embedding_fn:
                vector = embedding_fn(log_text)
            else:
                vector = self._create_random_embedding()
            
            # Store mapping from ID to index and text
            log_id_str = self._log_id_to_str(log["id"])
            self.logs_embeddings[log_id_str] = {
                "text": log_text,
                "vector": vector.tolist(),
                "metadata": {
                    "id": log["id"],
                    "action": log["action"],
                    "name": log["name"],
                    "timestamp": log["timestamp"],
                    "details": log["details"]
                }
            }
        
        # Save embeddings
        self._save_embeddings()
        
        # Rebuild FAISS index
        self._rebuild_index()
        
        return len(new_logs)
    
    def _rebuild_index(self):
        """Rebuild the FAISS index from stored embeddings"""
        if not self.logs_embeddings:
            print("No embeddings to build index")
            return
        
        # Get all vectors
        vectors = []
        self.id_to_index = {}
        
        for i, (id_str, data) in enumerate(self.logs_embeddings.items()):
            vectors.append(np.array(data["vector"], dtype=np.float32))
            self.id_to_index[id_str] = i
        
        # Convert to numpy array
        vectors_np = np.vstack(vectors).astype(np.float32)
        
        # Create new index
        self.index = faiss.IndexFlatL2(self.vector_dim)
        
        # Add vectors to index
        self.index.add(vectors_np)
        
        print(f"Rebuilt index with {len(vectors)} vectors")
    
    def search(self, query_vector, top_k=5):
        """
        Search for the most similar logs
        
        Args:
            query_vector: Vector to search for
            top_k: Number of results to return
            
        Returns:
            list: List of (log_text, metadata) tuples
        """
        # Ensure vector is right shape
        query_vector = np.array(query_vector).astype('float32').reshape(1, -1)
        
        # Search index
        distances, indices = self.index.search(query_vector, top_k)
        
        # Get results
        results = []
        index_to_id = {v: k for k, v in self.id_to_index.items()}
        
        for i, idx in enumerate(indices[0]):
            if idx < 0 or idx >= len(index_to_id):
                continue
                
            id_str = index_to_id[idx]
            data = self.logs_embeddings[id_str]
            
            results.append({
                "text": data["text"],
                "metadata": data["metadata"],
                "distance": float(distances[0][i])
            })
        
        return results

# For testing
if __name__ == "__main__":
    vector_store = FaceLogVectorStore()
    
    # Update with random embeddings for testing
    num_added = vector_store.update_vector_store()
    print(f"Added {num_added} new logs to vector store")
    
    # Test search with random vector
    test_vector = np.random.rand(384).astype('float32')
    results = vector_store.search(test_vector, top_k=3)
    
    for i, result in enumerate(results):
        print(f"Result {i+1}: {result['text']} (Distance: {result['distance']:.4f})") 