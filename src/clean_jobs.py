# src/processing/clean_skills.py
import pandas as pd
import numpy as np
import torch
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer, util
import joblib
import os
import re

import ast
from pathlib import Path
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataScienceJobsCleaner:
    """
    Clean and preprocess data science job listings from Found.dev API
    """
    
    # Common data science skill categories for classification
    SKILL_CATEGORIES = {
        'programming': ['python', 'r', 'sql', 'java', 'scala', 'c++', 'javascript', 'julia'],
        'ml_frameworks': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'mxnet', 'caffe'],
        'big_data': ['spark', 'hadoop', 'hive', 'kafka', 'airflow', 'dbt', 'snowflake'],
        'cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
        'visualization': ['tableau', 'powerbi', 'matplotlib', 'seaborn', 'plotly', 'd3'],
        'statistics': ['statistics', 'hypothesis testing', 'experimentation', 'a/b testing'],
        'ml_techniques': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'reinforcement learning']
    }
    
    def __init__(self, data_dir: Path = Path("../data")):
        """
        Initialize the cleaner with data directory paths
        
        Args:
            data_dir: Root data directory path
        """
        self.data_dir = data_dir
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load raw data from batch CSV files
        
        Args:
            batch_files: Specific batch files to load. If None, loads all batches
            
        Returns:
            Combined DataFrame of all raw data
        """
        if not file_path:
            raise FileNotFoundError("No file found")
        
        
        combined_df = pd.read_csv(file_path)
        
        return combined_df
    
    def clean_location_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance location information"""
        df_clean = df.copy()
        
        # Country standardization mapping
        COUNTRY_MAP = {
            # US variations
            **{state: 'USA' for state in ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
                                        'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
                                        'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
                                        'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
                                        'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']},
            **{us_var: 'USA' for us_var in ['US', 'U S.', 'UNITED STATES', 'UNITED STATES OF AMERICA', 'CA US','USA']},
            
            # Other countries
            'UK': 'United Kingdom', 'UNITED KINGDOM': 'United Kingdom','ENGLAND':'United Kingdom',
            'UAE': 'United Arab Emirates', 'FR': 'France', 'ZA': 'South Africa',
            'CY': 'Cyprus', 'UA': 'Ukraine', 'IT': 'Italy', 'MAROC': 'Morocco',
            'CZECHIA': 'Czech Republic', 'BOSNIA': 'Bosnia and Herzegovina',
            'KOREA': 'South Korea', 'UNKNOWN': 'Unknown', '[ ]': 'Unknown',
            'EUROPE': 'Region/Continent', 'NORTH AMERICA': 'Region/Continent',
            'LATIN AMERICA': 'Region/Continent','TURKIYE':'Turkey','TUNISIE':'Tunisia','BRASIL':'Brazil',
            'JP': 'Japan','UY':'Uruguay','PH':'Philippines','SG':'Singapore','NL':'Netherlands',
            'SE':'Sweden','CH':'Switzerland','GR':'Greece','IN':'India','PT':'Portugal',
            'AU':'Australia','NZ':'New Zealand','BE':'Belgium','MX':'Mexico','RU':'Russia',
            'TR':'Turkey','EG':'Egypt','FI':'Finland','SA':'Saudi Arabia','IE':'Ireland',
            'The Netherlands'.upper():'Netherlands','Bosnia and Herzegovina'.upper():'Bosnia and Herzegovina',
            'Korea, Republic Of'.upper():'South Korea'
        }
        
        # Extract and clean country data
        def extract_country(location):
            if pd.isna(location): return 'Unknown'
            country = str(location).split(':')[0].split(';')[0].strip().upper()
            return COUNTRY_MAP.get(country, country.title())
        
        df_clean['country'] = df_clean['country'].apply(extract_country)
        
        logger.info("✅ Location data cleaned")
        return df_clean
      
    def clean_job_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and categorize job types.
        Handles multilingual and mixed-format job type fields.
        """
        df_clean = df.copy()

        # Define standard mappings (lowercased)
        type_mapping = {
            'full_time': [
                'full-time', 'full time', 'permanent', 'regular', 'employee', 'voltijds', 
                'vollzeit', 'heltid', 'a jornada completa', 'fulltid', 'fuldtid',
                'pełny etat', 'полная занятость', '全职', 'período integral'
            ],
            'part_time': [
                'part-time', 'part time', 'teilzeit', 'deeltijds', 'meio período', 
                'чaстичная занятость', 'working student'
            ],
            'contract': [
                'contract', 'freelance', 'temporary', 'fixed term', 'consultant', 
                'contrat', 'contrato', 'werkvertrag', 'project based', 'fellowship', 
                'billable', 'fte or 1099', 'limited', 'cdi', 'cdi cadre'
            ],
            'internship': [
                'internship', 'intern', 'stage', 'staż', 'stagiair', 'praktikant', 
                'stagista', 'co-op', 'co op', 'graduate', 'industrial placement',
                'binance accelerator program', 'apprenticeship', 'thesis', 'training'
            ],
            'hybrid': [
                'hybrid', 'in-office', 'in office', 'in-person', 'office-based', 
                'on-site', 'onsite', 'onroll', 'on-roll', 'on-rolls', 'in-office'
            ],
            'remote': [
                'remote', 'remoto', 'remoto primeiro', 'full remoto'
            ]
        }

        def standardize_job_type(job_type):
            if pd.isna(job_type):
                return ['unknown']

            job_type_lower = str(job_type).lower()
            job_type_lower = re.sub(r'[^a-zA-Z\s,;/-]', ' ', job_type_lower)
            tokens = re.split(r'[,;/]', job_type_lower)

            matched_types = set()
            for token in tokens:
                token = token.strip()
                for category, variants in type_mapping.items():
                    if any(v in token for v in variants):
                        matched_types.add(category)

            return list(matched_types) if matched_types else ['other']

        # Apply cleaning
        df_clean['cleaned_job_type'] = df_clean['type'].apply(standardize_job_type)

        # Optional: create a simplified primary type (first in list)
        df_clean['primary_job_type'] = df_clean['cleaned_job_type'].apply(lambda x: x[0] if x else 'other')

        logger.info("✅ Job types cleaned and standardized.")
        return df_clean
    
    def clean_salary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and enhance salary information
        
        Args:
            df: DataFrame with salary columns
            
        Returns:
            DataFrame with cleaned salary data
        """
        df_clean = df.copy()
        
        # Convert 0 values to NaN (likely missing data)
        salary_cols = ['salary', 'salary_min', 'salary_max']
        for col in salary_cols:
            df_clean[col] = df_clean[col].replace(0, np.nan)
        
        # Currency mapping by country
        COUNTRY_CURRENCY = {
            'United States': '$', 'Canada': 'CAD', 'United Kingdom': '£',
            'Germany': '€', 'France': '€', 'Italy': '€', 'Spain': '€',
            'Netherlands': '€', 'Poland': 'PLN', 'India': '₹', 'Japan': '¥',
            'Australia': 'AUD', 'Brazil': 'BRL', 'Mexico': 'MXN', 'Sweden': 'SEK',
            'Denmark': 'DKK', 'Switzerland': 'CHF', 'China': 'CNY', 'Taiwan': 'NT$','Korea':'KRW',
            'Vietnam': 'VND','United Arab Emirates':'AED','Usa':'$'
        }
        
        def extract_and_convert_salary(salary_str, country):
            try:
                if pd.isna(salary_str) or not isinstance(salary_str, (str, float, int)):
                    return np.nan, np.nan
                    
                salary_str = str(salary_str).strip()
                
                if salary_str in ['', 'Market rate', 'Competitive, as per company policy', 'Negotiable']:
                    return np.nan, np.nan
                
                # Skip non-salary entries
                skip_terms = ['Market rate', 'Competitive', 'Negotiable']
                if any(term in salary_str for term in skip_terms):
                    return np.nan, np.nan
                
                # Extract currency from salary string first
                EXCHANGE_RATES = {
                    'PLN': 0.25, 'EUR': 1.10, 'INR': 0.012, 'NT$': 0.033, 'SEK': 0.095,
                    'DKK': 0.15, '¥': 0.0075, 'JPY': 0.0075, 'CNY': 0.14, '₹': 0.012,
                    'AUD': 0.67, 'BRL': 0.20, 'MXN': 0.055, 'CAD': 0.75, 'GBP': 1.27,
                    '£': 1.27, '€': 1.10, '$': 1.00, 'CHF': 1.12,'VND': 0.000043,'KRW': 0.00075,
                    'USD': 1.00
                }
                
                salary_currency = None
                for curr_symbol, rate in EXCHANGE_RATES.items():
                    if curr_symbol and curr_symbol in salary_str:
                        salary_currency = curr_symbol
                        break
                
                # Determine final currency: use salary currency if it matches country, else use country default
                if type(country) == str:
                    country = country.title().strip().strip(',').replace('Usa','United States').replace('Uk','United Kingdom').replace('United States Of America','United States')
                country_currency = COUNTRY_CURRENCY.get(country, '$')
                # If we have a valid country currency, use it. Otherwise use the salary currency.
                if country_currency and country_currency in EXCHANGE_RATES:
                    final_currency = country_currency
                elif salary_currency and salary_currency in EXCHANGE_RATES:
                    final_currency = salary_currency
                else:
                    final_currency = '$'  # Default to USD
                
                # Multiple number extraction strategies
                numbers = re.findall(r'[\d,]+\.?\d*', salary_str)
                
                cleaned_numbers = []
                for num in numbers:
                    # Remove ALL non-numeric characters except dot and comma
                    cleaned_num = re.sub(r'[^\d,.]', '', str(num))
                    # Remove commas for conversion
                    cleaned_num = cleaned_num.replace(',', '')
                    
                    if cleaned_num and cleaned_num != '.':  # Skip empty strings and lone dots
                        try:
                            num_float = float(cleaned_num)
                            cleaned_numbers.append(num_float)
                        except ValueError:
                            continue
                
                if not cleaned_numbers:
                    return np.nan, np.nan
                
                # Use min/max of found numbers
                min_salary = min(cleaned_numbers)
                max_salary = max(cleaned_numbers)
                
                # Handle single number case (use same value for min and max)
                if len(cleaned_numbers) == 1:
                    max_salary = min_salary
                
                # Period conversion
                HOURS_PER_YEAR = 2080
                MONTHS_PER_YEAR = 12
                
                salary_lower = salary_str.lower()
                salary_lower = salary_lower.split('hourly,')[0]
                if ('annual' in salary_lower) or ('year' in salary_lower):
                    pass
                elif 'hourly' in salary_lower or '/h' in salary_lower:
                    min_salary *= HOURS_PER_YEAR
                    max_salary *= HOURS_PER_YEAR
                elif 'monthly' in salary_lower:
                    min_salary *= MONTHS_PER_YEAR
                    max_salary *= MONTHS_PER_YEAR
                
                # Currency conversion
                exchange_rate = EXCHANGE_RATES.get(final_currency, 1.0)
                min_salary_usd = min_salary * exchange_rate
                max_salary_usd = max_salary * exchange_rate

                if min_salary_usd > 1_000_000 or max_salary_usd > 1_000_000:
                    return min_salary_usd / 100, max_salary_usd / 100
                
                return min_salary_usd, max_salary_usd
                
            except Exception as e:
                print(f"Error processing salary: '{salary_str}, {country}' - {str(e)}")
                return np.nan, np.nan
        
        # Apply the function to each row
        results = []
        for idx, row in df_clean.iterrows():
            salary_result = extract_and_convert_salary(row['salary'], row.get('country', ''))
            results.append(salary_result)
        
        # Create separate columns for min and max salary
        min_salaries = [r[0] for r in results]
        max_salaries = [r[1] for r in results]
        
        df_clean['salary_min_usd'] = min_salaries
        df_clean['salary_max_usd'] = max_salaries
        
        # Create salary category
        def categorize_salary(salary):
            if pd.isna(salary):
                return 'Unknown'
            salary = float(salary)
            if salary < 50000:
                return 'Low (<50k)'
            elif salary < 100000:
                return 'Medium (50k-100k)'
            elif salary < 150000:
                return 'High (100k-150k)'
            else:
                return 'Very High (>150k)'
        
        df_clean['salary_category'] = df_clean['salary_min_usd'].apply(categorize_salary)
        
        logger.info("✅ Salary data cleaned")
        return df_clean
    
    def convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date columns to proper datetime format
        
        Args:
            df: DataFrame with date columns
            
        Returns:
            DataFrame with converted dates
        """
        df_clean = df.copy()
        
        # Convert published date
        df_clean['published'] = pd.to_datetime(df_clean['published'], utc=True, errors='coerce')
        
        # Extract date components
        df_clean['published_year'] = df_clean['published'].dt.year
        df_clean['published_month'] = df_clean['published'].dt.month
        
        logger.info("✅ Date data converted")
        return df_clean
        
    def run_full_cleaning_pipeline(self, batch_files: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline
        
        Args:
            batch_files: Specific batch files to process
            
        Returns:
            Fully cleaned DataFrame
        """
        logger.info("🚀 Starting data cleaning pipeline...")
        
        # Load raw data
        df = self.load_raw_data(self.data_dir/'combined_jobs.csv')
        
        # Apply cleaning steps
        df_clean = (df
                   .pipe(self.clean_location_data)
                   .pipe(self.clean_job_type)
                   .pipe(self.clean_salary_data)
                   .pipe(self.convert_dates) 
                    )
        df_clean = df_clean.reset_index(drop=True)
        # Final data quality check
        self._quality_check(df_clean)
        
        logger.info(f"🎉 Cleaning complete! Final dataset: {len(df_clean)} records")
        
        return df_clean
    
    def _quality_check(self, df: pd.DataFrame):
        """
        Perform basic data quality checks
        
        Args:
            df: Cleaned DataFrame to check
        """
        logger.info("🔍 Performing data quality checks...")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        total_records = len(df)
        
        for column, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = (missing_count / total_records) * 100
                logger.info(f"   {column}: {missing_count} missing ({percentage:.1f}%)")
        
        # Check data types
        logger.info("📊 Data types:")
        for column, dtype in df.dtypes.items():
            logger.info(f"   {column}: {dtype}")
    
    def save_cleaned_data(self, df: pd.DataFrame, output_type: str = "interim"):
        """
        Save cleaned data to appropriate directory
        
        Args:
            df: Cleaned DataFrame to save
            output_type: "interim" for partially cleaned, "processed" for final
        """
        if output_type == "interim":
            output_path = self.interim_dir / "cleaned_jobs.csv"
        elif output_type == "processed":
            output_path = self.processed_dir / "processed_jobs.csv"
        else:
            raise ValueError("output_type must be 'interim' or 'processed'")
        
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved {len(df)} records to {output_path}")

# Convenience function for quick usage
def clean_jobs_data(data_dir: Path = Path("../data"), save_interim: bool = True) -> pd.DataFrame:
    """
    Convenience function to run the full cleaning pipeline
    
    Args:
        data_dir: Root data directory
        save_interim: Whether to save the cleaned data
        
    Returns:
        Cleaned DataFrame
    """
    cleaner = DataScienceJobsCleaner(data_dir)
    cleaned_df = cleaner.run_full_cleaning_pipeline()
    
    if save_interim:
        cleaner.save_cleaned_data(cleaned_df, "interim")
    
    return cleaned_df

class HybridJobTitleClassifier:
    """
    Hybrid job title classifier that:
    - Uses SentenceTransformer for semantic similarity.
    - Clusters emerging/unclassified roles.
    - Extracts seniority levels.
    - Preserves all original DataFrame columns.
    - Avoids duplicate rows by aligning on index, not title.
    """

    def __init__(self,
                 model_name="all-MiniLM-L6-v2",
                 threshold=0.45,
                 cluster_path="role_clusters.pkl",
                 category_embeddings_path="categories.pkl"):

        self.model = SentenceTransformer(model_name)
        self.threshold = threshold
        self.cluster_path = cluster_path
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
        self.category_embeddings = self.model.encode(self.categories, normalize_embeddings=True)

        # Seniority patterns (regex-based)
        self.seniority_patterns = [
            (r'\b(intern|internship|trainee|apprentice|student)\b', 'Intern'),
            (r'\b(entry[-\s]?level|junior|jr\.?)\b', 'Junior'),
            (r'\b(associate|mid[-\s]?level|intermediate)\b', 'Mid-level'),
            (r'\b(senior|sr\.?)\b', 'Senior'),
            (r'\b(lead|principal|specialist|staff)\b', 'Lead'),
            (r'\b(manager|head|director|supervisor)\b', 'Manager'),
            (r'\b(vice president|vp|chief|executive|cto|ceo|founder|co-founder)\b', 'Executive')
        ]

        if os.path.exists(cluster_path):
            self.clusters = joblib.load(cluster_path)
        else:
            self.clusters = {}

    def extract_seniority(self, title: str) -> str:
        """Extract seniority level using regex."""
        if not title or not isinstance(title, str):
            return "Unknown"
        title_lower = title.lower()
        for pattern, label in self.seniority_patterns:
            if re.search(pattern, title_lower):
                return label
        return "Unspecified"

    def classify_dataframe(self, df: pd.DataFrame, title_col: str, n_clusters: int = 6) -> pd.DataFrame:
        """
        Classify job titles and append results to the DataFrame.
        Returns the same number of rows with 3 new columns.
        """
        df = df.copy()
        titles = df[title_col].fillna("").tolist()

        # Encode job titles
        title_embeddings = self.model.encode(titles, normalize_embeddings=True)
        similarities = util.cos_sim(title_embeddings, self.category_embeddings)

        results = []
        residuals, residual_vecs = [], []

        # Step 1: Semantic classification
        for i, title in enumerate(titles):
            best_idx = torch.argmax(similarities[i]).item()
            best_score = similarities[i][best_idx].item()
            seniority = self.extract_seniority(title)

            if best_score < self.threshold:
                residuals.append(title)
                residual_vecs.append(title_embeddings[i])
                results.append((title, "Unclassified", seniority, best_score))
            else:
                results.append((title, self.categories[best_idx], seniority, best_score))

        # Step 2: Cluster emerging roles (low-similarity cases)
        if residuals:
            residual_vecs = np.vstack(residual_vecs)
            n_clusters = min(n_clusters, len(residuals))
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(residual_vecs)

            for cluster_id in np.unique(cluster_labels):
                cluster_titles = [residuals[i] for i in range(len(residuals)) if cluster_labels[i] == cluster_id]
                centroid = kmeans.cluster_centers_[cluster_id]

                sims = util.cos_sim(
                    torch.tensor(centroid).unsqueeze(0),
                    self.model.encode(cluster_titles, normalize_embeddings=True)
                )
                rep_title = cluster_titles[int(torch.argmax(sims))]
                label = f"Emerging Role Cluster: {rep_title}"
                self.clusters[label] = centroid

                for i, title in enumerate(residuals):
                    if cluster_labels[i] == cluster_id:
                        for j in range(len(results)):
                            if results[j][0] == title:
                                results[j] = (title, label, results[j][2], results[j][3])

        # Step 3: Create results DataFrame (index-aligned)
        results_df = pd.DataFrame(results, columns=[
            title_col,
            "cleaned_title_category",
            "seniority_level",
            "similarity_score"
        ])

        # Align by index — prevents row duplication
        results_df.index = df.index

        # Step 4: Concatenate results to original DataFrame
        df_final = pd.concat([
            df,
            results_df[["cleaned_title_category", "seniority_level", "similarity_score"]]
        ], axis=1)

        return df_final

    def save_state(self):
        """Persist clusters and category embeddings."""
        joblib.dump(self.clusters, self.cluster_path)
        joblib.dump(self.category_embeddings, self.category_embeddings_path)
        print(f"✅ State saved: {len(self.clusters)} clusters tracked.")
class SkillEnhancer:
    GENERIC_SKILLS = {
        'Data Science', 'Artificial Intelligence', 'Ai', 'Analytics',
        'Data Analysis', 'Programming', 'Coding', 'Statistics', 'Ml', 'Data Engineer'
    }

    @staticmethod
    def parse_skills(skills_str: str) -> list[str]:
        """Parse a skills string into a cleaned list."""
        if not isinstance(skills_str, str) or not skills_str.strip():
            return []

        try:
            # Convert list-like strings safely
            skills = ast.literal_eval(skills_str) if skills_str.startswith('[') else skills_str.split(',')
            # Normalize and filter
            cleaned = {
                s.strip().title() for s in skills
                if s and s.strip() and s.strip().title() not in SkillEnhancer.GENERIC_SKILLS
            }
            return list(cleaned)
        except Exception:
            logger.warning(f"Could not parse skills string: {skills_str}")
            return []

    @staticmethod
    def enhance_skills_data(df: pd.DataFrame) -> pd.DataFrame:
        """Enhance DataFrame by cleaning skills and adding skill count."""
        df = df.copy()
        df['skills'] = df['skills'].apply(SkillEnhancer.parse_skills)
        df['skills_count'] = df['skills'].apply(len)
        logger.info("✅ Skills data enhanced")
        return df

