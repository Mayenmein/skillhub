import pandas as pd
import logging
from pathlib import Path

from src.database.connection import DB
from src.database.schema_manager import SchemaManager
from src.database.sql_loader import SQLLoader 

logger = logging.getLogger(__name__)

class SQLCleaner:
    """
    Uses your SQL files for bulk operations, Python for complex logic.
    """
    def __init__(self):
        self.db = DB()
        self.sql = SQLLoader()
        self.schema = SchemaManager()  
    
    def setup_database(self):
        """One-time setup - runs your SQL files."""
        logger.info("Setting up database tables and mappings...")
        self.schema.ensure_tables_exist()          
        logger.info("Database setup complete")
    
    def run_sql_cleaning(self):
        """Run all SQL-based cleaning steps."""
        logger.info(" Running SQL cleaning pipeline...")
        self.sql.run_all_cleaning_steps()      
        logger.info("SQL cleaning complete")
    
    def check_progress(self):
        """Check cleaning progress."""
        with self.db.cursor(dict_mode=True) as cur:
            cur.execute("""
                SELECT 
                    COUNT(*) as total,
                    COUNT(country_standardized) as with_country,
                    COUNT(job_type) as with_job_type,
                    COUNT(seniority_level) as with_seniority,
                    COUNT(skills) as with_skills,
                    COUNT(salary_min_usd) as with_salary,
                    COUNT(standardized_title) as with_title
                FROM cleaned_jobs
            """)
            return cur.fetchone()
   