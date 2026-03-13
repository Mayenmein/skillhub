"""
Lightning-fast database connection management.
"""

import os
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values
from contextlib import contextmanager
from pathlib import Path
from dotenv import load_dotenv
import logging

env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

logger = logging.getLogger(__name__)

class DB:
    """Fast, minimal database connection manager"""
    
    def __init__(self):
        self.config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "dbname": os.getenv("DB_NAME", "job_market_db"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "")
        }
    
    @contextmanager
    def cursor(self, dict_mode=False, commit=True):
        """Get a cursor - that's it. One method does it all."""
        conn = psycopg2.connect(**self.config)
        cur = conn.cursor(cursor_factory=RealDictCursor if dict_mode else None)
        try:
            yield cur
            if commit:
                conn.commit()
        except Exception as e:
            conn.rollback()
            raise e
        finally:
            cur.close()
            conn.close()
    
    def execute(self, sql, params=None, dict_mode=False):
        """Execute single query, return results if any."""
        with self.cursor(dict_mode) as cur:
            cur.execute(sql, params or ())
            if cur.description:  # SELECT query
                return cur.fetchall()
            return None
    
    def execute_many(self, sql, params_list, page_size=1000):
        """Fast bulk inserts/updates."""
        with self.cursor() as cur:
            psycopg2.extras.execute_batch(cur, sql, params_list, page_size)

    def test_connection(self):
        """Test database connection."""
        try:
            with self.cursor() as cur:
                cur.execute("SELECT 1")
                result = cur.fetchone()
                print(f"Database connection test successful")
                return True
        except Exception as e:
            print(f"Database connection test failed: {e}")
            return False