import sqlite3
from datetime import datetime
from config import DB_PATH


def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS chat_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            user_message TEXT,
            model_response TEXT,
            model_thinking TEXT,
            word_limit INTEGER,
            word_count INTEGER
        )
    ''')
    conn.commit()
    conn.close()

def log_chat(user_message, model_response, model_thinking="", word_limit=0, word_count=0):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO chat_logs (timestamp, user_message, model_response, model_thinking, word_limit, word_count)
        VALUES (?, ?, ?, ?, ?, ?)
    ''', (timestamp, user_message, model_response, model_thinking, word_limit, word_count))

    conn.commit()
    conn.close()
    
