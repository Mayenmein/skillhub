import logging
from pathlib import Path
from .connection import DB
import psycopg2

logger = logging.getLogger(__name__)

class SchemaManager:
    """Simply runs the SQL files you already have."""
    
    def __init__(self):
        self.db = DB()
        self.sql_dir = Path(__file__).parent.parent.parent / "sql"
    
    def ensure_tables_exist(self):
        """Run your create_cleaned_table.sql to ensure tables exist."""
        create_table_file = self.sql_dir / "00 schema" / "create_cleaned_table.sql"
        
        if not create_table_file.exists():
            logger.error(f"SQL file not found: {create_table_file}")
            return False
        
        with open(create_table_file, 'r') as f:
            sql = f.read()
        
        # Split and execute statements
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        
        with self.db.cursor() as cur:
            for stmt in statements:
                if stmt:
                    try:
                        cur.execute(stmt)
                        logger.debug(f"Executed: {stmt[:50]}...")
                    except Exception as e:
                        # Check if it's a "already exists" error (shouldn't happen with IF NOT EXISTS)
                        error_str = str(e).lower()
                        if "already exists" in error_str:
                            logger.warning(f"Object already exists, skipping: {stmt[:50]}...")
                        else:
                            # Re-raise unexpected errors
                            logger.error(f"Unexpected error in statement: {stmt[:100]}...")
                            raise e
        
        logger.info("Tables ready (from your SQL definition)")
        return True