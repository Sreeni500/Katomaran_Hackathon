import sqlite3
import os
import json
from datetime import datetime

class FaceDatabase:
    def __init__(self, db_path='database/faces.db', log_path='database/logs.db'):
        """
        Initialize the Face Database
        
        Args:
            db_path: Path to the face database
            log_path: Path to the log database
        """
        # Create database directory if it doesn't exist
        os.makedirs('database', exist_ok=True)
        
        self.db_path = db_path
        self.log_path = log_path
        
        # Initialize the databases
        self._init_face_db()
        self._init_log_db()
    
    def _init_face_db(self):
        """Initialize the face database schema"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create faces table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            encoding TEXT,
            timestamp TEXT
        )
        """)
        
        conn.commit()
        conn.close()
    
    def _init_log_db(self):
        """Initialize the log database schema"""
        conn = sqlite3.connect(self.log_path)
        cursor = conn.cursor()
        
        # Create face_logs table if it doesn't exist
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS face_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            action TEXT,
            name TEXT,
            timestamp TEXT,
            details TEXT
        )
        """)
        
        conn.commit()
        conn.close()
    
    def add_face(self, name, encoding):
        """
        Add a face to the database
        
        Args:
            name: Name of the person
            encoding: Face embedding
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Convert encoding to JSON string if it's not already
            if not isinstance(encoding, str):
                encoding = json.dumps(encoding)
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute(
                "INSERT INTO faces (name, encoding, timestamp) VALUES (?, ?, ?)",
                (name, encoding, timestamp)
            )
            
            conn.commit()
            
            # Add log entry
            self.add_log("registration", name)
            
            return True
        
        except Exception as e:
            print(f"Error adding face: {str(e)}")
            return False
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_all_faces(self):
        """
        Get all registered faces
        
        Returns:
            list: List of tuples (id, name, encoding, timestamp)
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM faces")
            faces = cursor.fetchall()
            
            return faces
        
        except Exception as e:
            print(f"Error getting faces: {str(e)}")
            return []
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_face_by_name(self, name):
        """
        Get a face by name
        
        Args:
            name: Name of the person
            
        Returns:
            list: List of matching face entries
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute("SELECT * FROM faces WHERE name = ?", (name,))
            faces = cursor.fetchall()
            
            return faces
        
        except Exception as e:
            print(f"Error getting face by name: {str(e)}")
            return []
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    def delete_face(self, face_id):
        """
        Delete a face by ID
        
        Args:
            face_id: ID of the face to delete
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get name before deleting for log
            cursor.execute("SELECT name FROM faces WHERE id = ?", (face_id,))
            result = cursor.fetchone()
            
            if not result:
                return False
            
            name = result[0]
            
            cursor.execute("DELETE FROM faces WHERE id = ?", (face_id,))
            conn.commit()
            
            # Add log entry
            self.add_log("deletion", name)
            
            return True
        
        except Exception as e:
            print(f"Error deleting face: {str(e)}")
            return False
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    def add_log(self, action, name, details=None):
        """
        Add a log entry
        
        Args:
            action: Type of action (registration, recognition, deletion)
            name: Name of the person
            details: Additional details (optional)
            
        Returns:
            bool: Success status
        """
        try:
            conn = sqlite3.connect(self.log_path)
            cursor = conn.cursor()
            
            timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            cursor.execute(
                "INSERT INTO face_logs (action, name, timestamp, details) VALUES (?, ?, ?, ?)",
                (action, name, timestamp, details)
            )
            
            conn.commit()
            return True
        
        except Exception as e:
            print(f"Error adding log: {str(e)}")
            return False
        
        finally:
            if 'conn' in locals():
                conn.close()
    
    def get_logs(self, limit=100, action=None, name=None):
        """
        Get logs with optional filters
        
        Args:
            limit: Maximum number of logs to return
            action: Filter by action type
            name: Filter by person name
            
        Returns:
            list: List of log entries
        """
        try:
            conn = sqlite3.connect(self.log_path)
            cursor = conn.cursor()
            
            query = "SELECT * FROM face_logs"
            params = []
            
            if action or name:
                query += " WHERE"
                
                if action:
                    query += " action = ?"
                    params.append(action)
                    
                    if name:
                        query += " AND"
                
                if name:
                    query += " name = ?"
                    params.append(name)
            
            query += " ORDER BY timestamp DESC LIMIT ?"
            params.append(limit)
            
            cursor.execute(query, params)
            logs = cursor.fetchall()
            
            return logs
        
        except Exception as e:
            print(f"Error getting logs: {str(e)}")
            return []
        
        finally:
            if 'conn' in locals():
                conn.close()

# For testing
if __name__ == "__main__":
    db = FaceDatabase()
    print("Database initialized") 