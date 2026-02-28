import os
import time
import requests
import psycopg2
from psycopg2.extras import execute_values
from pathlib import Path
from tqdm import tqdm
from dotenv import load_dotenv

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class JobScraper:
    def __init__(self, base_url="https://api.found.dev/api/open/jobs", sql_file: Path = None):
        self.BASE_URL = base_url
        self.HEADERS = {"User-Agent": "Mozilla/5.0"}
        
        # Set default SQL file path if not provided
        if sql_file is None:
            sql_file = Path(__file__).parent.parent.parent / "sql" / "00 schema" / "create_raw_tables.sql"
        self.sql_file = sql_file
        
        # Load database credentials from environment variables
        self.db_config = {
            "host": os.getenv("DB_HOST", "localhost"),
            "port": os.getenv("DB_PORT", "5432"),
            "dbname": os.getenv("DB_NAME", "job_market_db"),
            "user": os.getenv("DB_USER", "postgres"),
            "password": os.getenv("DB_PASSWORD", "")
        }
        
        # Validate that password exists
        if not self.db_config["password"]:
            print("⚠️  Warning: DB_PASSWORD not set in .env file")
            print("Please add DB_PASSWORD=your_password to your .env file")
        
        # Create tables from SQL file
        self._create_tables_from_sql()

    def _create_tables_from_sql(self):
        """Create tables using the SQL schema file"""
        try:
            # Read SQL file
            with open(self.sql_file, 'r') as f:
                sql_commands = f.read()
            
            # Connect and execute SQL
            with psycopg2.connect(**self.db_config) as conn:
                conn.autocommit = True
                with conn.cursor() as cur:
                    cur.execute(sql_commands)
            
            print("Database tables created successfully from SQL schema")
            
        except FileNotFoundError:
            print(f"SQL file not found at: {self.sql_file}")
            raise
        except Exception as e:
            print(f"Error creating tables: {e}")
            raise

    def fetch_jobs(self, page=1, skill="Data Science", ai=True):
        params = {"page": page, "skill": skill, "ai": str(ai).lower()}
        resp = requests.get(self.BASE_URL, headers=self.HEADERS, params=params)
        resp.raise_for_status()
        return resp.json()

    def process_job_data(self, jobs_data):
        """Process jobs data and prepare for database insertion"""
        if not jobs_data:
            return [], [], []
        
        companies = []
        jobs = []
        skill_details = []
        
        for entry in jobs_data:
            job = entry.get("job", {})
            company = entry.get("company", {})
            location = entry.get("location", "")
            types = entry.get("types", [])
            
            if not job or not company:
                continue
            
            # Process company (deduplicate by slug)
            company_slug = company.get('slug')
            if company_slug and not any(c.get('slug') == company_slug for c in companies):
                companies.append({
                    'slug': company_slug,
                    'name': company.get('name'),
                    'description': company.get('description'),
                    'description_gpt': company.get('description_gpt'),
                    'description_premium': company.get('description_premium'),
                    'description_linkedin': company.get('description_linkedin'),
                    'description_combined_gpt': company.get('description_combined_gpt'),
                    'description_perplexity': company.get('description_perplexity'),
                    'logo': company.get('logo'),
                    'url': company.get('url'),
                    'city': company.get('city'),
                    'country': company.get('country'),
                    'areas': company.get('areas'),
                    'size_min': company.get('size_min'),
                    'size_max': company.get('size_max'),
                    'year_founded': company.get('year_founded'),
                    'glassdoor_score': company.get('glassdoor_score'),
                    'linkedin_staff_count': company.get('linkedin_staff_count'),
                    'linkedin_follower_count': company.get('linkedin_follower_count'),
                    'twitter': company.get('twitter'),
                    'linkedin': company.get('linkedin'),
                    'facebook': company.get('facebook'),
                    'instagram': company.get('instagram'),
                    'github': company.get('github'),
                    'crunchbase': company.get('crunchbase'),
                    'angel_list': company.get('angel_list'),
                    'stack_overflow': company.get('stack_overflow'),
                    'behance': company.get('behance'),
                    'dribbble': company.get('dribbble'),
                    'product_hunt': company.get('product_hunt'),
                    'public_vs_private': company.get('public_vs_private'),
                    'company_sector': company.get('company_sector'),
                    'linkedin_tags': company.get('linkedin_tags'),
                    'key_products': company.get('key_products'),
                    'using_ai': company.get('using_ai'),
                    'ai_products_gpt': company.get('ai_products_gpt'),
                    'mission_values': company.get('mission_values'),
                    'jobs_count': company.get('jobs_count', 0),
                    'jobs_ai_count': company.get('jobs_ai_count', 0),
                    'last_job_source': company.get('last_job_source')
                })
            
            # Parse skills into arrays
            def parse_skills(skills_str):
                if skills_str and isinstance(skills_str, str):
                    return [s.strip() for s in skills_str.split(',') if s.strip()]
                return []
            
            skills = parse_skills(job.get('skills'))
            soft_skills = parse_skills(job.get('soft_skills'))
            tools = parse_skills(job.get('tools'))
            languages = parse_skills(job.get('languages'))
            frameworks = parse_skills(job.get('frameworks'))
            libraries = parse_skills(job.get('libraries'))
            
            # Parse job types
            job_types = []
            if job.get('type'):
                job_types = [t.strip() for t in job.get('type').split(',') if t.strip()]
            elif types:
                job_types = types
            
            # Create location string
            city = job.get('city', '')
            country = job.get('country', '')
            location_str = location or (f"{city}, {country}" if city and country else city or country or '')
            
            # Process job
            job_slug = job.get('slug')
            if job_slug:
                jobs.append({
                    'slug': job_slug,
                    'company_slug': company_slug,
                    'title': job.get('title'),
                    'description': job.get('description'),
                    'premium': job.get('premium', False),
                    'ai': job.get('ai', False),
                    'status': job.get('status'),
                    'created_at': job.get('created_at') if job.get('created_at') != "0001-01-01T00:00:00Z" else None,
                    'published': job.get('published') if job.get('published') != "0001-01-01T00:00:00Z" else None,
                    'pin_until': job.get('pin_until') if job.get('pin_until') != "0001-01-01T00:00:00Z" else None,
                    'skills': skills,
                    'soft_skills': soft_skills,
                    'tools': tools,
                    'languages': languages,
                    'frameworks': frameworks,
                    'libraries': libraries,
                    'roles': job.get('roles'),
                    'seniority': job.get('seniority'),
                    'types': job_types,
                    'city': city,
                    'country': country,
                    'location': location_str,
                    'salary_min': job.get('salary_min', 0),
                    'salary_max': job.get('salary_max', 0),
                    'salary_currency': job.get('salary_currency'),
                    'salary_period': job.get('salary_period'),
                    'benefits': job.get('benefits'),
                    'beneficial': job.get('beneficial'),
                    'experience': job.get('experience'),
                    'ideal_candidate': job.get('ideal_candidate'),
                    'qualifications': job.get('qualifications'),
                    'schema': job.get('schema'),
                    'force_status': job.get('force_status'),
                    'same_as': job.get('same_as', 0)
                })
                
                # Create skill details for normalized table
                skill_categories = [
                    ('skills', skills, 'technical'),
                    ('soft_skills', soft_skills, 'soft'),
                    ('tools', tools, 'tool'),
                    ('languages', languages, 'language'),
                    ('frameworks', frameworks, 'framework'),
                    ('libraries', libraries, 'library')
                ]
                
                for field, skill_list, category in skill_categories:
                    for skill in skill_list:
                        if skill:
                            skill_details.append((job_slug, skill, category))
        
        return companies, jobs, skill_details

    def save_jobs(self, companies, jobs, skill_details):
        """Save jobs to database, skipping duplicates"""
        if not jobs:
            return 0
        
        try:
            with psycopg2.connect(**self.db_config) as conn:
                with conn.cursor() as cur:
                    
                    # Insert companies
                    if companies:
                        company_query = """
                        INSERT INTO companies (
                            slug, name, description, description_gpt, description_premium,
                            description_linkedin, description_combined_gpt, description_perplexity,
                            logo, url, city, country, areas, size_min, size_max,
                            year_founded, glassdoor_score, linkedin_staff_count, linkedin_follower_count,
                            twitter, linkedin, facebook, instagram, github, crunchbase,
                            angel_list, stack_overflow, behance, dribbble, product_hunt,
                            public_vs_private, company_sector, linkedin_tags, key_products,
                            using_ai, ai_products_gpt, mission_values, jobs_count,
                            jobs_ai_count, last_job_source
                        ) VALUES %s
                        ON CONFLICT (slug) DO NOTHING
                        """
                        
                        company_values = [(
                            c['slug'], c['name'], c['description'], c['description_gpt'],
                            c['description_premium'], c['description_linkedin'], c['description_combined_gpt'],
                            c['description_perplexity'], c['logo'], c['url'], c['city'],
                            c['country'], c['areas'], c['size_min'], c['size_max'],
                            c['year_founded'], c['glassdoor_score'], c['linkedin_staff_count'],
                            c['linkedin_follower_count'], c['twitter'], c['linkedin'], c['facebook'],
                            c['instagram'], c['github'], c['crunchbase'], c['angel_list'],
                            c['stack_overflow'], c['behance'], c['dribbble'], c['product_hunt'],
                            c['public_vs_private'], c['company_sector'], c['linkedin_tags'],
                            c['key_products'], c['using_ai'], c['ai_products_gpt'], c['mission_values'],
                            c['jobs_count'], c['jobs_ai_count'], c['last_job_source']
                        ) for c in companies if c and c.get('slug')]
                        
                        if company_values:
                            execute_values(cur, company_query, company_values)
                    
                    # Insert jobs
                    job_query = """
                    INSERT INTO jobs (
                        slug, company_slug, title, description, premium, ai, status,
                        created_at, published, pin_until, skills, soft_skills, tools,
                        languages, frameworks, libraries, roles, seniority, types,
                        city, country, location, salary_min, salary_max, salary_currency,
                        salary_period, benefits, beneficial, experience, ideal_candidate,
                        qualifications, schema, force_status, same_as
                    ) VALUES %s
                    ON CONFLICT (slug) DO NOTHING
                    """
                    
                    job_values = [(
                        j['slug'], j['company_slug'], j['title'], j['description'],
                        j['premium'], j['ai'], j['status'], j['created_at'],
                        j['published'], j['pin_until'], j['skills'], j['soft_skills'],
                        j['tools'], j['languages'], j['frameworks'], j['libraries'],
                        j['roles'], j['seniority'], j['types'], j['city'],
                        j['country'], j['location'], j['salary_min'], j['salary_max'],
                        j['salary_currency'], j['salary_period'], j['benefits'],
                        j['beneficial'], j['experience'], j['ideal_candidate'],
                        j['qualifications'], j['schema'], j['force_status'], j['same_as']
                    ) for j in jobs if j and j.get('slug')]
                    
                    if job_values:
                        execute_values(cur, job_query, job_values)
                        inserted_jobs = cur.rowcount
                    
                    # Insert skill details
                    if skill_details:
                        # Remove duplicates
                        skill_details = list(set(skill_details))
                        
                        skill_query = """
                        INSERT INTO job_skills_detail (job_slug, skill_name, skill_category)
                        VALUES %s
                        ON CONFLICT (job_slug, skill_name, skill_category) DO NOTHING
                        """
                        execute_values(cur, skill_query, skill_details)
                    
                    conn.commit()
                    return inserted_jobs
                    
        except Exception as e:
            print(f"Database save failed: {e}")
            return 0

    def scrape_in_batches(self, skill="Data Science", pages_per_batch=20, ai=True, delay=1, max_batches=None):
        if pages_per_batch <= 0: 
            print("Pages per batch must be greater than 0.")
            return 0
        
        batch_num, page, total_jobs = 1, 1, 0
        print(f"Scraping {skill} jobs...")
        
        # Main progress bar for overall progress
        with tqdm(desc="Overall Progress", unit="jobs") as overall_pbar:
            overall_pbar.update(0)
            
            while True:
                if max_batches and batch_num > max_batches: 
                    print(f"\nReached maximum batch limit: {max_batches}")
                    return total_jobs
                
                all_batch_jobs = []
                
                # Batch progress bar
                with tqdm(total=pages_per_batch, desc=f"Batch {batch_num}", 
                        unit="page", leave=False) as batch_pbar:
                    for page_in_batch in range(pages_per_batch):
                        try:
                            data = self.fetch_jobs(page=page, skill=skill, ai=ai)
                            if jobs := data.get("jobs", []):
                                all_batch_jobs.extend(jobs)
                                page += 1
                                time.sleep(delay)
                            else:
                                # No more jobs - save and exit
                                if all_batch_jobs:
                                    companies, jobs_processed, skill_details = self.process_job_data(all_batch_jobs)
                                    batch_count = self.save_jobs(companies, jobs_processed, skill_details)
                                    total_jobs += batch_count
                                    overall_pbar.update(batch_count)
                                    overall_pbar.set_postfix({"total": total_jobs})
                                return total_jobs
                        except Exception as e:
                            print(f"\nError on page {page}: {e}")
                            page += 1
                        
                        batch_pbar.update(1)
                        batch_pbar.set_postfix({"batch_jobs": len(all_batch_jobs)})
                
                # End of batch - save and update totals
                if all_batch_jobs:
                    companies, jobs_processed, skill_details = self.process_job_data(all_batch_jobs)
                    batch_count = self.save_jobs(companies, jobs_processed, skill_details)
                    total_jobs += batch_count
                    
                    # Update overall progress bar
                    overall_pbar.update(batch_count)
                    overall_pbar.set_postfix({
                        "total": total_jobs,
                        "batches": batch_num
                    })
                    
                    print(f"Batch {batch_num} complete: +{batch_count} jobs (Total: {total_jobs})")
                
                batch_num += 1