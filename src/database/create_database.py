"""
Database creation script that uses credentials from .env file.
Run this once to create your database before running the scraper.
"""

import os
import sys
from pathlib import Path
from dotenv import load_dotenv
import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
from psycopg2 import sql

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

# Load environment variables from .env file
env_path = project_root / '.env'
if not env_path.exists():
    print(f".env file not found at {env_path}")
    print("Please create a .env file with your database credentials")
    print("You can copy .env.example and fill in your details")
    sys.exit(1)

load_dotenv(dotenv_path=env_path)

class DatabaseCreator:
    """Create a PostgreSQL database using credentials from environment variables"""
    
    def __init__(self):
        # Load configuration from environment
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "dbname": "postgres",  # Connect to default postgres database first
            "user": os.getenv("PG_SUPERUSER", os.getenv("DB_USER", "postgres")),
            "password": os.getenv("PG_SUPERUSER_PASSWORD", os.getenv("DB_PASSWORD", ""))
        }
        
        self.target_db = os.getenv("DB_NAME", "job_market_db")
        
    def validate_config(self):
        """Check if required environment variables are set"""
        required_vars = ["DB_PASSWORD"]
        missing = [var for var in required_vars if not os.getenv(var)]
        
        if missing:
            print("Missing required environment variables:")
            for var in missing:
                print(f"   - {var}")
            print("\nPlease add them to your .env file")
            return False
        
        # Check if we have superuser credentials (optional but recommended)
        if not os.getenv("PG_SUPERUSER_PASSWORD") and os.getenv("DB_PASSWORD"):
            print("Using DB_PASSWORD for database creation. If this fails,")
            print("   you may need to set PG_SUPERUSER_PASSWORD in .env")
        
        return True
    
    def database_exists(self, cursor, db_name):
        """Check if database already exists"""
        cursor.execute(
            "SELECT 1 FROM pg_database WHERE datname = %s",
            (db_name,)
        )
        return cursor.fetchone() is not None
    
    def create_database(self):
        """Create the target database if it doesn't exist"""
        try:
            # Connect to default postgres database
            print(f"Connecting to PostgreSQL at {self.db_config['host']}:{self.db_config['port']}...")
            conn = psycopg2.connect(**self.db_config)
            conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
            cursor = conn.cursor()
            
            print(f"Connected to PostgreSQL as {self.db_config['user']}")
            
            # Check if database already exists
            if self.database_exists(cursor, self.target_db):
                print(f"Database '{self.target_db}' already exists")
                return True
            
            # Create database
            print(f"Creating database '{self.target_db}'...")
            cursor.execute(
                sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.target_db))
            )
            print(f"Database '{self.target_db}' created successfully!")
            
            # Optional: Create additional users or set permissions
            self.setup_permissions(cursor)
            
            cursor.close()
            conn.close()
            return True
            
        except psycopg2.Error as e:
            print(f"Database creation failed: {e}")
            
            # Provide helpful error messages
            if "password authentication failed" in str(e):
                print("\nPassword authentication failed. Check your .env file:")
                print("   - DB_USER and DB_PASSWORD must match your PostgreSQL credentials")
                print("   - For default PostgreSQL installation, user is usually 'postgres'")
            elif "connection refused" in str(e):
                print("\nCould not connect to PostgreSQL. Is it running?")
                print("   Start it with: sudo service postgresql start (Linux)")
                print("   Or check PostgreSQL services in Task Manager (Windows)")
            elif "permission denied" in str(e):
                print("\nPermission denied. Your user may not have CREATE DATABASE privileges.")
                print("   Try using a superuser account or run with PG_SUPERUSER_PASSWORD set")
            
            return False
    
    def setup_permissions(self, cursor):
        """Optional: Set up additional permissions"""
        try:
            # Grant all privileges on the new database to the regular user
            regular_user = os.getenv("DB_USER", "postgres")
            if regular_user != self.db_config["user"]:
                grant_query = sql.SQL(
                    "GRANT ALL PRIVILEGES ON DATABASE {} TO {}"
                ).format(
                    sql.Identifier(self.target_db),
                    sql.Identifier(regular_user)
                )
                cursor.execute(grant_query)
                print(f"Granted privileges to user '{regular_user}'")
        except Exception as e:
            print(f"Could not set permissions: {e}")
    
    def test_connection(self):
        """Test connection to the newly created database"""
        try:
            # Try connecting to the new database
            test_config = {
                "host": os.getenv("DB_HOST", "localhost"),
                "port": os.getenv("DB_PORT", "5432"),
                "dbname": self.target_db,
                "user": os.getenv("DB_USER", "postgres"),
                "password": os.getenv("DB_PASSWORD", "")
            }
            
            conn = psycopg2.connect(**test_config)
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            conn.close()
            print(f"Successfully connected to '{self.target_db}' with regular user credentials")
            return True
        except Exception as e:
            print(f"Could not connect with regular user: {e}")
            return False
        
    def run(self):
        """Run the database creation process"""
        if not self.validate_config():
            return
        
        if self.create_database():
            self.test_connection()

