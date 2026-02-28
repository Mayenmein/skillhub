import os
import requests
import psycopg2
import ijson
import time
from psycopg2.extras import execute_values
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv




# Load environment variables
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class JobDataAPIStreamingScraper:
    def __init__(self, base_url="https://jobdataapi.com/api/jobs/", sql_file: Path = None):
        self.BASE_URL = base_url
        
        if sql_file is None:
            sql_file = Path(__file__).parent.parent.parent / "sql" / "00 schema" / "create_raw_tables_datajobs.sql"
        self.sql_file = sql_file

        # Database configuration
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "dbname": os.getenv("DB_NAME", "job_market_db"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "")
        }
        
        # Batch size for database inserts
        self.batch_size = 50
        
        # Tracking sets for this session only (not persisted across runs)
        self.session_job_ids = set()
        self.session_company_ids = set()
        self.session_type_ids = set()
        self.session_location_ids = set()
        self.session_country_ids = set()

        self.create_tables_from_sql()

    def create_tables_from_sql(self):
        """
        Import and execute SQL file containing table definitions
        """
        try:
            # Read the SQL file
            with open(self.sql_file, 'r') as f:
                sql_commands = f.read()
            
            with psycopg2.connect(**self.db_config) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql_commands)
            
            print(f" Tables created successfully from {self.sql_file}")
            
            # Verify tables were created
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT table_name 
                        FROM information_schema.tables 
                        WHERE table_schema = 'public'
                    """)
                    tables = cur.fetchall()
                    print(f" Existing tables: {', '.join([t[0] for t in tables])}")
            
            return True
            
        except FileNotFoundError:
            print(f" SQL file not found: {self.sql_file}")
            return False
        except Exception as e:
            print(f"Error creating tables: {e}")
            return False
    
    def stream_jobs_from_api(self, country_code='CM'):
        """
        Stream jobs directly from the HTTP response without loading everything into memory
        """
        params = {}
        if country_code:
            params["country_code"] = country_code
        
        # Set headers to match browser but request JSON specifically
        headers = {
            'User-Agent': 'Mozilla/5.0 (Linux; Android 13; Pixel 7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Mobile Safari/537.36',
            'Accept': 'application/json',  # Explicitly request JSON only
            'Accept-Language': 'en-US,en;q=0.9',
            'Accept-Encoding': 'gzip, deflate, br',  # Support Brotli compression
            'Connection': 'keep-alive',
            'Upgrade-Insecure-Requests': '1',
            'Sec-Fetch-Dest': 'document',
            'Sec-Fetch-Mode': 'navigate',
            'Sec-Fetch-Site': 'none',
            'Sec-Fetch-User': '?1',
            'Cache-Control': 'max-age=0'
        }
        
        print(f"Streaming jobs from JobDataAPI" + (f" for country: {country_code}" if country_code else ""))
        print("=" * 60)
        
        # Make request with stream=True and proper headers
        response = requests.get(self.BASE_URL, params=params, headers=headers, stream=True, timeout=60)
        response.raise_for_status()
        
        # Handle Brotli compression
        raw_data = response.raw
        
        # Check content encoding and decompress appropriately
        content_encoding = response.headers.get('content-encoding', '')
        
        if 'br' in content_encoding:
            import brotli
            # Read the compressed data and decompress
            compressed_data = raw_data.read()
            decompressed_data = brotli.decompress(compressed_data)
            import io
            raw_data = io.BytesIO(decompressed_data)
        elif 'gzip' in content_encoding:
            import gzip
            raw_data = gzip.GzipFile(fileobj=raw_data)
        elif 'deflate' in content_encoding:
            import zlib
            compressed_data = raw_data.read()
            decompressed_data = zlib.decompress(compressed_data)
            import io
            raw_data = io.BytesIO(decompressed_data)
        
        # Reset tracking sets for this session
        self.session_job_ids.clear()
        self.session_company_ids.clear()
        self.session_type_ids.clear()
        self.session_location_ids.clear()
        self.session_country_ids.clear()
        
        # Collections for current batch
        batch_companies = []
        batch_types = []
        batch_job_types = []
        batch_countries = []
        batch_locations = []
        batch_job_locations = []
        batch_jobs = []
        
        batch_count = 0
        total_jobs = 0
         
        # Parse the JSON stream
        parser = ijson.parse(raw_data)
         
        with tqdm(desc="Processing jobs", unit="jobs") as pbar:
            for job in ijson.items(raw_data, 'results.item'):
                
                job_id = job.get('id')
                if not job_id or job_id in self.session_job_ids:
                    continue
                
                # Extract and queue company data
                company_data = job.get('company', {})
                if company_data and company_data.get('id') not in self.session_company_ids:
                    batch_companies.append(self._extract_company(company_data))
                    self.session_company_ids.add(company_data.get('id'))
                
                # Extract and queue job types
                types = job.get('types', [])
                for type_item in types:
                    type_id = type_item.get('id')
                    if type_id and type_id not in self.session_type_ids:
                        batch_types.append((type_id, type_item.get('name')))
                        self.session_type_ids.add(type_id)
                    
                    if type_id and job_id:
                        batch_job_types.append((job_id, type_id))
                
                # Extract and queue countries
                countries = job.get('countries', [])
                for country in countries:
                    country_id = country.get('id')
                    if country_id and country_id not in self.session_country_ids:
                        batch_countries.append(self._extract_country(country))
                        self.session_country_ids.add(country_id)
                
                # Extract and queue locations
                cities = job.get('cities', [])
                for city in cities:
                    geonameid = city.get('geonameid')
                    if geonameid and geonameid not in self.session_location_ids:
                        batch_locations.append(self._extract_location(city))
                        self.session_location_ids.add(geonameid)
                    
                    if geonameid and job_id:
                        batch_job_locations.append((job_id, geonameid))
                
                # Extract and queue job
                batch_jobs.append(self._extract_job(job))
                self.session_job_ids.add(job_id)
                
                batch_count += 1
                total_jobs += 1
                
                # When batch is full, insert and reset
                if batch_count >= self.batch_size:
                    self._insert_batch_data(
                        batch_companies, batch_types, batch_job_types,
                        batch_countries, batch_locations, batch_job_locations,
                        batch_jobs
                    )
                    
                    # Reset batches
                    batch_companies = []
                    batch_types = []
                    batch_job_types = []
                    batch_countries = []
                    batch_locations = []
                    batch_job_locations = []
                    batch_jobs = []
                    batch_count = 0
                    
                    # Small delay to not overwhelm database
                    time.sleep(0.1)
                
                pbar.update(1)
                pbar.set_postfix({"total": total_jobs})
        
        # Insert any remaining jobs
        if batch_jobs:
            self._insert_batch_data(
                batch_companies, batch_types, batch_job_types,
                batch_countries, batch_locations, batch_job_locations,
                batch_jobs
            )
        
        print(f"\nStreaming complete! Total jobs processed: {total_jobs}")
        return total_jobs
    
    def _extract_company(self, company):
        """Extract company fields from API data"""
        return (
            company.get('id'),
            company.get('source_id'),
            company.get('name'),
            company.get('logo'),
            company.get('website_url'),
            company.get('linkedin_url'),
            company.get('twitter_handle'),
            company.get('github_url'),
            company.get('is_agency', False)
        )
    
    def _extract_country(self, country):
        """Extract country fields from API data"""
        region = country.get('region', {})
        return (
            country.get('id'),
            country.get('code'),
            country.get('name'),
            region.get('id'),
            region.get('name')
        )
    
    def _extract_location(self, city):
        """Extract location fields from API data"""
        country = city.get('country', {})
        state = city.get('state', {})
        return (
            city.get('geonameid'),
            city.get('asciiname'),
            city.get('name'),
            country.get('code'),
            state.get('code'),
            city.get('timezone'),
            city.get('latitude'),
            city.get('longitude'),
            city.get('population', 0)
        )
    
    def _extract_job(self, job):
        """Extract job fields from API data"""
        return (
            job.get('id'),
            job.get('ext_id'),
            job.get('company', {}).get('id') if job.get('company') else None,
            job.get('title'),
            job.get('location'),
            job.get('description'),
            job.get('experience_level'),
            job.get('application_url'),
            job.get('language'),
            job.get('has_remote', False),
            job.get('published'),
            job.get('salary_min'),
            job.get('salary_max'),
            job.get('salary_currency')
        )
    
    def _insert_batch_data(self, companies, types, job_types, countries, 
                          locations, job_locations, jobs):
        """Insert all collected data into database using batch inserts"""
        
        if not jobs:
            return
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    
                    # Insert companies
                    if companies:
                        company_query = """
                        INSERT INTO datajobscompanies (id, source_id, name, logo, website_url, 
                                              linkedin_url, twitter_handle, github_url, is_agency)
                        VALUES %s
                        ON CONFLICT (id) DO NOTHING
                        """
                        execute_values(cur, company_query, companies)
                    
                    # Insert job types
                    if types:
                        type_query = """
                        INSERT INTO datajob_types (id, name) VALUES %s
                        ON CONFLICT (id) DO NOTHING
                        """
                        execute_values(cur, type_query, types)
                    
                    # Insert countries
                    if countries:
                        country_query = """
                        INSERT INTO datajobscountries (id, code, name, region_id, region_name)
                        VALUES %s
                        ON CONFLICT (id) DO NOTHING
                        """
                        execute_values(cur, country_query, countries)
                    
                    # Insert locations
                    if locations:
                        location_query = """
                        INSERT INTO datajobslocations (geonameid, asciiname, name, country_code, 
                                              state_code, timezone, latitude, longitude, population)
                        VALUES %s
                        ON CONFLICT (geonameid) DO NOTHING
                        """
                        execute_values(cur, location_query, locations)
                    
                    # Insert jobs
                    if jobs:
                        job_query = """
                        INSERT INTO datajobs (id, ext_id, company_id, title, location, description,
                                         experience_level, application_url, language, has_remote,
                                         published, salary_min, salary_max, salary_currency)
                        VALUES %s
                        ON CONFLICT (id) DO NOTHING
                        """
                        execute_values(cur, job_query, jobs)
                    
                    # Insert job-type relationships
                    if job_types:
                        job_type_query = """
                        INSERT INTO datajob_to_types (job_id, type_id) VALUES %s
                        ON CONFLICT (job_id, type_id) DO NOTHING
                        """
                        execute_values(cur, job_type_query, job_types)
                    
                    # Insert job-location relationships
                    if job_locations:
                        job_location_query = """
                        INSERT INTO datajob_locations (job_id, geonameid) VALUES %s
                        ON CONFLICT (job_id, geonameid) DO NOTHING
                        """
                        execute_values(cur, job_location_query, job_locations)
                    
                    conn.commit()
                    
        except Exception as e:
            print(f"Database insert failed: {e}")
    
    def scrape_by_country(self, country_code='CM'):
        """
        Scrape jobs, optionally filtered by country code
        Example: country_code='CM' for Cameroon
        """
        return self.stream_jobs_from_api(country_code=country_code)
    
    def get_database_stats(self):
        """Get statistics from the database"""
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT COUNT(*) FROM datajobs")
                    total_jobs = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM datajobscompanies")
                    total_companies = cur.fetchone()[0]
                    
                    cur.execute("SELECT COUNT(*) FROM datajobslocations")
                    total_locations = cur.fetchone()[0]
                    
                    print("\nDATABASE STATISTICS")
                    print("=" * 40)
                    print(f"Total jobs: {total_jobs:,}")
                    print(f"Total companies: {total_companies:,}")
                    print(f"Total locations: {total_locations:,}")
                    
                    return {
                        'jobs': total_jobs,
                        'companies': total_companies,
                        'locations': total_locations
                    }
        except Exception as e:
            print(f"Error getting stats: {e}")
            return None
