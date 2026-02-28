"""
Fetch latest exchange rates from API and store in database.
Run this weekly via cron/Airflow to keep rates updated.
Stores ALL currencies from the API - no filtering.
"""

import requests
import logging
from datetime import datetime
from src.database.connection import DB

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Free API options (no key required)
# Option 1: exchangerate.host - provides ALL currencies
EXCHANGE_API = "https://api.exchangerate.host/latest?base=USD"

# Option 2: frankfurter.app (alternative if first fails)
BACKUP_API = "https://api.frankfurter.app/latest?from=USD"
db = DB()
def create_rates_table():
    """Ensure exchange_rates table exists - stores ALL currencies"""
    
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS exchange_rates (
        id SERIAL PRIMARY KEY,
        currency_code VARCHAR(3) NOT NULL,
        rate_to_usd DECIMAL(12, 6) NOT NULL,
        effective_date DATE NOT NULL,
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(currency_code, effective_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_exchange_rates_currency 
    ON exchange_rates(currency_code, effective_date);
    
    CREATE INDEX IF NOT EXISTS idx_exchange_rates_date 
    ON exchange_rates(effective_date);
    """
    
    with db.cursor(dict_mode=True) as cur:
        cur.execute(create_table_sql)
    logger.info("exchange_rates table ready")

def fetch_rates_from_api():
    """Fetch ALL latest exchange rates from free API"""
    try:
        # Try primary API first - exchangerate.host provides all currencies
        response = requests.get(EXCHANGE_API, timeout=10)
        if response.status_code == 200:
            data = response.json()
            rates = data['rates']
            logger.info(f"Fetched {len(rates)} currencies from exchangerate.host")
            
            # Log some sample currencies
            sample_currencies = list(rates.keys())[:10]
            logger.info(f"   Sample: {', '.join(sample_currencies)}...")
            
            return rates
    except Exception as e:
        logger.warning(f"Primary API failed: {e}")
    
    try:
        # Try backup API - frankfurter has fewer currencies but still good
        response = requests.get(BACKUP_API, timeout=10)
        if response.status_code == 200:
            data = response.json()
            rates = data['rates']
            logger.info(f"Fetched {len(rates)} currencies from frankfurter.app")
            return rates
    except Exception as e:
        logger.error(f"Backup API also failed: {e}")
    
    return None

def update_database_rates(rates):
    """Store ALL rates in database - no filtering"""
    if not rates:
        logger.error("No rates to update")
        return False
    
    today = datetime.now().date()
    
    with db.cursor(dict_mode=True) as cursor:
        # Insert USD to USD (always 1.0)
        cursor.execute("""
            INSERT INTO exchange_rates (currency_code, rate_to_usd, effective_date)
            VALUES (%s, %s, %s)
            ON CONFLICT (currency_code, effective_date) 
            DO UPDATE SET rate_to_usd = EXCLUDED.rate_to_usd
        """, ('USD', 1.0, today))
        
        # Insert ALL currencies from API - no filtering
        inserted = 0
        for currency, rate in rates.items():
            if currency != 'USD':  # Skip USD as we already inserted it
                cursor.execute("""
                    INSERT INTO exchange_rates (currency_code, rate_to_usd, effective_date)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (currency_code, effective_date) 
                    DO UPDATE SET rate_to_usd = EXCLUDED.rate_to_usd
                """, (currency, rate, today))
                inserted += 1
        
        logger.info(f"Updated {inserted} currency rates for {today}")
        
        # Log some statistics
        cursor.execute("SELECT COUNT(DISTINCT currency_code) FROM exchange_rates")
        total_currencies = cursor.fetchone()[0]
        logger.info(f"Total unique currencies in database: {total_currencies}")
        
        return True
