import sqlite3
import json
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from pathlib import Path
import asyncio


class AdvancedMemory:

    def __init__(
        self,
        db_path: str = "memory.db",
        max_messages: int = 20,
        summarize_threshold: int = 30,
        max_context_tokens: int = 4000
    ):
        self.db_path = db_path
        self.max_messages = max_messages
        self.summarize_threshold = summarize_threshold
        self.max_context_tokens = max_context_tokens
        
        self.messages: List[Dict[str, str]] = []
        self.summaries: List[str] = []
        
        self._init_db()
    
    def _init_db(self) -> None:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                session_id TEXT,
                tokens INTEGER DEFAULT 0
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS summaries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                summary TEXT NOT NULL,
                message_count INTEGER,
                session_id TEXT
            )
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_timestamp 
            ON conversations(timestamp)
        ''')
        
        cursor.execute('''
            CREATE INDEX IF NOT EXISTS idx_session 
            ON conversations(session_id)
        ''')
        
        conn.commit()
        conn.close()
    
    def add_message(
        self,
        role: str,
        content: str,
        session_id: Optional[str] = None,
        tokens: int = 0
    ) -> None:
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            "tokens": tokens
        }
        
        self.messages.append(message)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO conversations (timestamp, role, content, session_id, tokens)
               VALUES (?, ?, ?, ?, ?)''',
            (message["timestamp"], role, content, session_id, tokens)
        )
        conn.commit()
        conn.close()
        
        if len(self.messages) >= self.summarize_threshold:
            asyncio.create_task(self._auto_summarize(session_id))
    
    async def _auto_summarize(self, session_id: Optional[str] = None) -> None:
        if len(self.messages) < 10:
            return
        
        messages_to_summarize = self.messages[:-10]
        
        summary_text = self._create_summary(messages_to_summarize)
        
        self.summaries.append(summary_text)
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute(
            '''INSERT INTO summaries (timestamp, summary, message_count, session_id)
               VALUES (?, ?, ?, ?)''',
            (datetime.now().isoformat(), summary_text, len(messages_to_summarize), session_id)
        )
        conn.commit()
        conn.close()
        
        self.messages = self.messages[-10:]
    
    def _create_summary(self, messages: List[Dict[str, str]]) -> str:
        summary_parts = []
        
        for msg in messages:
            role = msg["role"].upper()
            content_preview = msg["content"][:100]
            summary_parts.append(f"{role}: {content_preview}...")
        
        return "\n".join(summary_parts)
    
    def get_messages(self, limit: Optional[int] = None) -> List[Dict[str, str]]:
        if limit:
            return self.messages[-limit:]
        return self.messages.copy()
    
    def get_context_string(self, include_summaries: bool = True) -> str:
        context_parts = []
        
        if include_summaries and self.summaries:
            context_parts.append("Previous conversation summary:")
            context_parts.append("\n".join(self.summaries))
            context_parts.append("\nRecent conversation:")
        
        for msg in self.messages:
            context_parts.append(f"{msg['role'].upper()}: {msg['content']}")
        
        return "\n\n".join(context_parts)
    
    def search_history(
        self,
        query: str,
        limit: int = 5,
        session_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute(
                '''SELECT timestamp, role, content, tokens 
                   FROM conversations 
                   WHERE content LIKE ? AND session_id = ?
                   ORDER BY timestamp DESC LIMIT ?''',
                (f'%{query}%', session_id, limit)
            )
        else:
            cursor.execute(
                '''SELECT timestamp, role, content, tokens 
                   FROM conversations 
                   WHERE content LIKE ?
                   ORDER BY timestamp DESC LIMIT ?''',
                (f'%{query}%', limit)
            )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "timestamp": row[0],
                "role": row[1],
                "content": row[2],
                "tokens": row[3]
            })
        
        conn.close()
        return results
    
    def get_session_history(
        self,
        session_id: str,
        limit: Optional[int] = None
    ) -> List[Dict[str, str]]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if limit:
            cursor.execute(
                '''SELECT timestamp, role, content, tokens 
                   FROM conversations 
                   WHERE session_id = ?
                   ORDER BY timestamp DESC LIMIT ?''',
                (session_id, limit)
            )
        else:
            cursor.execute(
                '''SELECT timestamp, role, content, tokens 
                   FROM conversations 
                   WHERE session_id = ?
                   ORDER BY timestamp DESC''',
                (session_id,)
            )
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "timestamp": row[0],
                "role": row[1],
                "content": row[2],
                "tokens": row[3]
            })
        
        conn.close()
        return list(reversed(results))
    
    def clear(self, session_id: Optional[str] = None) -> None:
        self.messages.clear()
        self.summaries.clear()
        
        if session_id:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute('DELETE FROM conversations WHERE session_id = ?', (session_id,))
            cursor.execute('DELETE FROM summaries WHERE session_id = ?', (session_id,))
            conn.commit()
            conn.close()
    
    def get_statistics(self, session_id: Optional[str] = None) -> Dict[str, any]:
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute(
                'SELECT COUNT(*), SUM(tokens) FROM conversations WHERE session_id = ?',
                (session_id,)
            )
        else:
            cursor.execute('SELECT COUNT(*), SUM(tokens) FROM conversations')
        
        row = cursor.fetchone()
        message_count = row[0] or 0
        total_tokens = row[1] or 0
        
        if session_id:
            cursor.execute(
                'SELECT COUNT(*) FROM summaries WHERE session_id = ?',
                (session_id,)
            )
        else:
            cursor.execute('SELECT COUNT(*) FROM summaries')
        
        summary_count = cursor.fetchone()[0] or 0
        
        conn.close()
        
        return {
            "total_messages": message_count,
            "total_tokens": total_tokens,
            "summaries_created": summary_count,
            "current_buffer_size": len(self.messages)
        }
    
    def prune_old_data(self, days: int = 30, session_id: Optional[str] = None) -> int:
        from datetime import timedelta
        
        cutoff_date = (datetime.now() - timedelta(days=days)).isoformat()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        if session_id:
            cursor.execute(
                'DELETE FROM conversations WHERE timestamp < ? AND session_id = ?',
                (cutoff_date, session_id)
            )
            cursor.execute(
                'DELETE FROM summaries WHERE timestamp < ? AND session_id = ?',
                (cutoff_date, session_id)
            )
        else:
            cursor.execute(
                'DELETE FROM conversations WHERE timestamp < ?',
                (cutoff_date,)
            )
            cursor.execute(
                'DELETE FROM summaries WHERE timestamp < ?',
                (cutoff_date,)
            )
        
        deleted_count = cursor.rowcount
        conn.commit()
        conn.close()
        
        return deleted_count
    
    def export_conversation(
        self,
        session_id: Optional[str] = None,
        output_file: Optional[str] = None
    ) -> str:
        messages = self.get_session_history(session_id) if session_id else self.get_messages()
        
        export_data = {
            "exported_at": datetime.now().isoformat(),
            "session_id": session_id,
            "message_count": len(messages),
            "messages": messages,
            "summaries": self.summaries
        }
        
        json_data = json.dumps(export_data, indent=2)
        
        if output_file:
            with open(output_file, 'w') as f:
                f.write(json_data)
        
        return json_data
