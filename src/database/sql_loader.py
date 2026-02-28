import logging
from pathlib import Path
from .connection import DB
import psycopg2

logger = logging.getLogger(__name__)

class SQLLoader:
    """Runs your existing SQL files - nothing more."""
    
    def __init__(self):
        self.db = DB()
        self.sql_dir = Path(__file__).parent.parent.parent / "sql"
    
    def run_file(self, filepath):
        """Run a single SQL file - simple and straightforward."""
        with open(filepath, 'r', encoding='utf-8') as f:
            sql = f.read()
        
        # Split by semicolon and filter out empty statements
        statements = [s.strip() for s in sql.split(';') if s.strip()]
        
        with self.db.cursor() as cur:
            for stmt in statements:
                if stmt:
                    try:
                        cur.execute(stmt)
                    except Exception as e:
                        # Just log and continue for mapping files, raise for others
                        if "mapping" in str(filepath).lower():
                            logger.warning(f"Error in {filepath.name} (continuing): {e}")
                        else:
                            logger.error(f"Error in {filepath.name}: {e}")
                            raise
        
        logger.info(f"✓ {Path(filepath).name}")
    
    def run_all_cleaning_steps(self):
        """Run all cleaning steps in order."""
        steps_dir = self.sql_dir / "02 cleaning_steps"
        if steps_dir.exists():
            for sql_file in sorted(steps_dir.glob("*.sql")):
                try:
                    self.run_file(sql_file)
                except Exception as e:
                    logger.error(f" Failed to run {sql_file.name}: {e}")
                    raise