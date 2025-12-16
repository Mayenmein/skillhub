import os
import time
import httpx
import gc
import random
import pandas as pd
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse
from typing import Dict, Any, List, Set

class URLCollector:
    def __init__(self, max_workers: int = 8):
        self.max_workers = max_workers
        self.client_timeout = 45.0
        self.max_retries = 5
        self.retry_delays = [2, 5, 10, 15, 30]
        
    def create_client_with_headers(self) -> httpx.Client:
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
            "Mozilla/5.0 (Windows NT 10.0; rv:89.0) Gecko/20100101 Firefox/89.0",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:89.0) Gecko/20100101 Firefox/89.0"
        ]
        return httpx.Client(
            timeout=self.client_timeout, follow_redirects=True, verify=False,
            headers={'User-Agent': random.choice(user_agents)},
            limits=httpx.Limits(max_keepalive_connections=5, max_connections=10)
        )
    
    def is_valid_url(self, url: str) -> bool:
        if not url or url == 'https://api.found.dev/redirect/': return False
        try: return bool(urlparse(url).scheme and urlparse(url).netloc)
        except: return False
    
    def classify_url_type(self, final_url: str) -> str:
        if not self.is_valid_url(final_url): return 'invalid'
        url_patterns = {'api': 'dejobs.org', 'greenhouse': 'greenhouse.io', 'lever': 'lever.co', 
                       'workday': 'workday', 'icims': 'icims.com', 'jobvite': 'jobvite.com', 'taleo': 'taleo'}
        return next((url_type for url_type, pattern in url_patterns.items() if pattern in final_url.lower()), 'web')
    
    def get_final_url_with_retry(self, slug: str) -> Dict[str, Any]:
        redirect_url = f"https://api.found.dev/redirect/{slug}"
        last_exception = None
        
        for attempt in range(self.max_retries):
            try:
                with self.create_client_with_headers() as client:
                    response = client.get(redirect_url, follow_redirects=True)
                    if response.status_code >= 400:
                        raise httpx.HTTPStatusError(f"HTTP {response.status_code}", request=response.request, response=response)
                    
                    final_url = str(response.url)
                    if final_url == redirect_url or not self.is_valid_url(final_url):
                        raise ValueError(f"Invalid final URL: {final_url}")
                    
                    return {'slug': slug, 'final_url': final_url, 'type': self.classify_url_type(final_url),
                           'status_code': response.status_code, 'attempts': attempt + 1, 'error': None, 'success': True}
                    
            except Exception as e:
                last_exception = str(e)
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delays[attempt] + random.uniform(0, 2))
        
        return {'slug': slug, 'final_url': None, 'type': 'failed', 'status_code': None,
               'attempts': self.max_retries, 'error': last_exception, 'success': False}
    
    def read_slugs_from_csv(self, file_path: str, slug_column: str = 'slug') -> List[str]:
        try:
            df = pd.read_csv(file_path, usecols=[slug_column])
            return df[slug_column].dropna().unique().tolist() if slug_column in df.columns else []
        except FileNotFoundError: print(f"File not found: {file_path}"); return []
    
    def get_existing_slugs(self, output_file: str) -> Set[str]:
        try:
            if os.path.exists(output_file):
                existing_slugs = set(pd.read_parquet(output_file)['slug'].tolist())
                print(f"Found {len(existing_slugs)} existing slugs in {output_file}")
                return existing_slugs
        except Exception as e: print(f"Error reading existing output file: {e}")
        return set()
    
    def process_urls_parallel(self, slugs: List[str]) -> List[Dict[str, Any]]:
        batch_results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            future_to_slug = {executor.submit(self.get_final_url_with_retry, slug): slug for slug in slugs}
            for future in tqdm(as_completed(future_to_slug), total=len(slugs), desc="Processing URLs"):
                slug = future_to_slug[future]
                try: batch_results.append(future.result())
                except Exception as e:
                    batch_results.append({'slug': slug, 'final_url': None, 'type': 'crashed', 'status_code': None,
                                         'attempts': 0, 'error': f"Thread error: {str(e)}", 'success': False})
        return batch_results

    def save_urls_to_parquet(self, results: List[Dict[str, Any]], output_file: str):
        try:
            df = pd.DataFrame(results)[['slug', 'final_url', 'type', 'success', 'attempts', 'status_code', 'error']]
            df.to_parquet(output_file, index=False, compression='brotli')
            print(f"Saved {len(df)} URLs ({df['success'].sum()} successful) to {output_file}")
            return output_file
        except Exception as e: print(f"Error saving to Parquet: {e}"); return None
    
    def update_output_file(self, new_results: List[Dict[str, Any]], output_file: str):
        try:
            new_df = pd.DataFrame(new_results)[['slug', 'final_url', 'type', 'success']]
            if os.path.exists(output_file):
                existing_df = pd.read_parquet(output_file)
                combined_df = pd.concat([existing_df, new_df], ignore_index=True).drop_duplicates(subset=['slug'], keep='last')
                combined_df = combined_df[combined_df['success']==True]
                combined_df.to_parquet(output_file, index=False, compression='brotli')
                print(f"Updated {output_file} with {len(new_df)} new URLs. Total: {len(combined_df)} URLs ({combined_df['success'].sum()} successful)")
            else:
                new_df[new_df['success']==True].to_parquet(output_file, index=False, compression='brotli')
                print(f"Created {output_file} with {len(new_df)} URLs ({new_df['success'].sum()} successful)")
            return output_file
        except Exception as e: print(f"Error updating output file: {e}"); return None
    
    def collect_all_urls(self, input_csv: str, output_file: str = 'job_urls.parquet', 
                        slug_column: str = 'slug', batch_size: int = 300):
        all_slugs = self.read_slugs_from_csv(input_csv, slug_column)
        if not all_slugs: print("No slugs found in input file"); return None
        
        existing_slugs = self.get_existing_slugs(output_file)
        slugs_to_process = [slug for slug in all_slugs if slug not in existing_slugs]
        
        if not slugs_to_process: print("All slugs have already been processed. No new URLs to collect."); return output_file
        
        print(f"Processing {len(slugs_to_process)} new slugs out of {len(all_slugs)} total (batch size: {batch_size})")
        
        for i in range(0, len(slugs_to_process), batch_size):
            batch_slugs = slugs_to_process[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(slugs_to_process)-1)//batch_size + 1}")
            
            batch_results = self.process_urls_parallel(batch_slugs)
            self.update_output_file(batch_results, output_file)
            
            del batch_results
            gc.collect()
            if i + batch_size < len(slugs_to_process): time.sleep(1)
        
        try:
            final_df = pd.read_parquet(output_file)
            success_count = final_df['success'].sum()
            print(f"Completed: {success_count}/{len(final_df)} successful ({success_count/len(final_df)*100:.1f}%)")
        except Exception as e: print(f"Error reading final stats: {e}")
        
        return output_file