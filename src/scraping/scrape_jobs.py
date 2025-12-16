import os
import time
import requests
import pandas as pd
from pathlib import Path
from tqdm import tqdm

class JobScraper:
    def __init__(self, base_url="https://api.found.dev/api/open/jobs", output_file: Path=Path("data/raw/jobs_data.csv")):
        self.BASE_URL = base_url
        self.HEADERS = {"User-Agent": "Mozilla/5.0"}
        self.output_file = output_file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

    def fetch_jobs(self, page=1, skill="Data Science", ai=True):
        params = {"page": page, "skill": skill, "ai": str(ai).lower()}
        resp = requests.get(self.BASE_URL, headers=self.HEADERS, params=params)
        resp.raise_for_status()
        return resp.json()

    def process_job_data(self, jobs_data):
        if not jobs_data: return []
        
        all_jobs = []
        for entry in jobs_data:
            job, company = entry.get("job", {}), entry.get("company", {})
            record = {
                "title": job.get("title"), "company": company.get("name"),
                "city": job.get("city"), "country": job.get("country"),
                "location": f"{job.get('city')}, {job.get('country')}",
                "skills": job.get("skills"), "type": job.get("type"),
                "salary": job.get("salary"), "salary_min": job.get("salary_min"),
                "salary_max": job.get("salary_max"), "published": job.get("published"),
                "ai": job.get("ai")
            }
            if any(record.values()): all_jobs.append(record)
        return all_jobs

    def save_jobs(self, all_jobs):
        if not all_jobs: return 0
        
        new_df = pd.DataFrame(all_jobs)
        combined_df = pd.concat([pd.read_csv(self.output_file), new_df], ignore_index=True) if os.path.exists(self.output_file) else new_df
        
        try: combined_df.to_csv(self.output_file, index=False)
        except Exception as e: print(f"file save operation failed because of : {e}"); return 0
         
        return len(new_df)
              
    def scrape_in_batches(self, skill="Data Science", pages_per_batch=20, ai=True, delay=1, max_batches=None):
        if pages_per_batch <= 0: print("Pages per batch must be greater than 0."); return 0
        
        batch_num, page, total_jobs = 1, 1, 0
        print(f"Scraping {skill} jobs...")
        
        with tqdm(desc="Batches", unit="batch") as batch_pbar:
            while True:
                if max_batches and batch_num > max_batches: print(f"Reached maximum batch limit: {max_batches}"); return total_jobs
                
                all_batch_jobs = []
                with tqdm(total=pages_per_batch, desc=f"Batch {batch_num}", unit="page", leave=False) as page_pbar:
                    for _ in range(pages_per_batch):
                        try:
                            data = self.fetch_jobs(page=page, skill=skill, ai=ai)
                            if jobs := data.get("jobs", []): all_batch_jobs.extend(jobs)
                            elif all_batch_jobs:
                                batch_count = self.save_jobs(self.process_job_data(all_batch_jobs))
                                return total_jobs + batch_count
                            page += 1; time.sleep(delay)
                        except Exception: page += 1
                        page_pbar.update(1); page_pbar.set_postfix({"jobs": len(all_batch_jobs)})

                if all_batch_jobs:
                    batch_count = self.save_jobs(self.process_job_data(all_batch_jobs))
                    total_jobs += batch_count
                    batch_pbar.update(1); batch_pbar.set_postfix({"total": total_jobs})
                
                batch_num += 1

