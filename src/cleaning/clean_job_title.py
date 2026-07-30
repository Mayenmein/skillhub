import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple
import re
import torch 
from sentence_transformers import SentenceTransformer, util

import pandas as pd
import numpy as np

from dotenv import load_dotenv
from pathlib import Path

from tqdm import tqdm

# Load environment variables from .env file
env_path = Path(__file__).parent.parent.parent / '.env'
load_dotenv(dotenv_path=env_path)

class HybridJobTitleClassifier:
    """
    Job title classifier optimized for CPU with:
    - Cached category embeddings
    - Multi-threaded seniority extraction
    - Optimized batch processing
    - Memory-efficient operations
    """

    def __init__(self,
                 model_name="all-MiniLM-L6-v2", 
                 category_embeddings_path="../models/categories.pkl",
                 cache_folder=os.getenv('cache_folder')):

        # Initialize model
        self.model = SentenceTransformer(
            model_name, 
            cache_folder=cache_folder, 
            local_files_only=False,
            device='cpu'  # Explicitly use CPU
        )
        
        self.category_embeddings_path = category_embeddings_path

        # Core job role categories
        self.categories = [
            "Data Scientist",
            "Machine Learning Engineer",
            "AI Engineer",
            "Data Analyst",
            "Data Engineer",
            "MLOps Engineer",
            "Data Architect",
            "Analytics Engineer",
            "Research Scientist",
            "Data Science Manager",
            "AI Researcher",
            "Product Data Scientist"
        ]
        
        # Load or compute category embeddings
        self.category_embeddings = self._load_or_compute_category_embeddings()
        
        # Cache for processed titles to avoid recomputation
        self.title_cache = {}

        # Pre-compile regex patterns for seniority extraction
        self.seniority_patterns = [
            (re.compile(r'\b(intern|internship|trainee|apprentice|student)\b', re.IGNORECASE), 'Intern'),
            (re.compile(r'\b(entry[-\s]?level|junior|jr\.?)\b', re.IGNORECASE), 'Junior'),
            (re.compile(r'\b(associate|mid[-\s]?level|intermediate)\b', re.IGNORECASE), 'Mid-level'),
            (re.compile(r'\b(senior|sr\.?)\b', re.IGNORECASE), 'Senior'),
            (re.compile(r'\b(lead|principal|specialist|staff)\b', re.IGNORECASE), 'Lead'),
            (re.compile(r'\b(manager|head|director|supervisor)\b', re.IGNORECASE), 'Manager'),
            (re.compile(r'\b(vice president|vp|chief|executive|cto|ceo|founder|co-founder)\b', re.IGNORECASE), 'Executive')
        ]

    def _load_or_compute_category_embeddings(self) -> np.ndarray:
        """Load pre-computed category embeddings or compute and save them."""
               
        # Compute embeddings if not loaded
        print("Computing category embeddings...")
        embeddings = self.model.encode(
            self.categories, 
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=32  # Optimized for CPU
        )
                
        return embeddings

    def extract_seniority(self, title: str) -> str:
        """Extract seniority level from a single title."""
        if not title or not isinstance(title, str) or title.strip() == "":
            return "Unknown"
        
        # Quick check for common patterns first (optimization)
        title_lower = title.lower()
        for pattern, label in self.seniority_patterns:
            if pattern.search(title_lower):
                return label
        return "Unspecified"

    def extract_seniority_batch(self, titles: List[str], max_workers: int = 4) -> List[str]:
        """
        Extract seniority levels for multiple titles in parallel.
        
        Args:
            titles: List of job titles
            max_workers: Number of parallel threads
            
        Returns:
            List of seniority levels
        """
        if not titles:
            return []
        
        # If small batch, don't bother with threading overhead
        if len(titles) < 100:
            return [self.extract_seniority(title) for title in titles]
        
        # Use ThreadPoolExecutor for parallel processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_title = {
                executor.submit(self.extract_seniority, title): i 
                for i, title in enumerate(titles)
            }
            
            # Initialize results list
            results = [None] * len(titles)
            
            # Collect results as they complete
            for future in as_completed(future_to_title):
                idx = future_to_title[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    results[idx] = "Unknown"
                    print(f"Error processing title at index {idx}: {e}")
        
        return results

    def classify_dataframe(self, 
                      df: pd.DataFrame, 
                      title_col: str, 
                      threshold: float = 0.47,
                      batch_size: int = 256 ) -> pd.DataFrame:
        """
        Simplified version without cache complexity - most reliable.
        """ 
        df = df.copy()
        df[title_col] = df[title_col].fillna("").astype(str)
        titles = df[title_col].tolist()
        total_titles = len(titles)
         
        # Initialize empty lists for results
        all_categories = []
        all_seniorities = []
        all_scores = []
        
        # Process all titles in batches with tqdm progress bar
        for batch_start in tqdm(range(0, total_titles, batch_size), desc="Classifying Dataframe"):
            batch_end = min(batch_start + batch_size, total_titles)
            batch_titles = titles[batch_start:batch_end]
             
            # Extract seniority in parallel
            seniorities_batch = self.extract_seniority_batch(batch_titles)
            
            # Encode titles (set show_progress_bar=True if you want sub-batch encoding progress)
            title_embeddings = self.model.encode(
                batch_titles, 
                normalize_embeddings=True, 
                show_progress_bar=False,  # Set to True for inner encoding progress
                batch_size=64  # Optimized for CPU
            )
            
            # Convert to torch tensor
            title_tensor = torch.tensor(title_embeddings)
            category_tensor = torch.tensor(self.category_embeddings)
            
            # Compute similarities
            similarities = util.cos_sim(title_tensor, category_tensor)
            best_scores, best_indices = torch.max(similarities, dim=1)
            
            # Convert to numpy
            best_scores_np = best_scores.numpy()
            best_indices_np = best_indices.numpy()
            
            # Process this batch
            for i in range(len(batch_titles)):
                score = best_scores_np[i]
                idx = best_indices_np[i]
                sen = seniorities_batch[i]
                
                # Determine category
                if score > threshold:
                    category = self.categories[idx]
                else:
                    category = 'Other'
                
                # Add to results
                all_categories.append(category)
                all_seniorities.append(sen)
                all_scores.append(score)
        
        # Add results to DataFrame
        df["cleaned_title"] = all_categories
        df["seniority_level"] = all_seniorities
        df["similarity_score"] = all_scores
         
        return df
