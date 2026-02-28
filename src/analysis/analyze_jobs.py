import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple 
import logging
from collections import Counter
from itertools import combinations
  
import warnings
from tqdm import tqdm 
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  


class BaseAnalyzer:
    """
    Comprehensive analyzer for data science job market trends
    """    
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        self.reports_dir = self.data_dir.parent / "reports"
        self.figures_dir = self.reports_dir / "figures"
        
        # Create directories if they don't exist
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        # --- Skill Categories ---
        self.SKILL_CATEGORIES = {
            "Programming": ["python", "r", "sql", "java", "scala", "c++", "javascript", "julia"],
            "ML Frameworks": ["tensorflow", "pytorch", "keras", "scikit-learn", "mxnet", "caffe"],
            "Big Data": ["spark", "hadoop", "hive", "kafka", "airflow", "dbt", "snowflake"],
            "Cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform"],
            "Visualization": ["tableau", "powerbi", "matplotlib", "seaborn", "plotly", "d3"],
            "Statistics": ["statistics", "hypothesis testing", "experimentation", "a/b testing"],
            "ML Techniques": ["machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning"],
        }
        self.seniority_order = ['Unspecified', 'Intern', 'Junior', 'Mid-level', 'Senior', 'Lead', 'Manager', 'Executive']           
        self.skill_to_category = {skill.lower(): cat for cat, skills in self.SKILL_CATEGORIES.items() for skill in skills}
        
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load cleaned data from interim directory"""
        interim_files = list(self.interim_dir.glob("*.csv"))
        if not interim_files:
            raise FileNotFoundError("No cleaned data found in interim directory")
        
        df = pd.read_csv(interim_files[0])
        logger.info(f"Loaded {len(df)} cleaned records")
        return df
    
    def _convert_to_list(self, x) -> List[str]:
        """Convert string representation of list to actual list"""
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        try:
            if isinstance(x, str) and x.startswith("["):
                return eval(x)
            else:
                return [s.strip() for s in str(x).split(",") if s.strip()]
        except:
            return []
    
    def create_skill_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Build a pivot-style DataFrame preserving job-level granularity.

        Each job-skill pair is expanded before aggregation.
        Groups by Month, Country, Company, Role, Seniority, etc.
        Counts distinct job mentions per skill and computes prevalence.
        """
        # --- Improved convert_to_list function ---
        def convert_to_list(x):
            # Handle if x is a Series/array (from pandas operations)
            if hasattr(x, '__len__') and not isinstance(x, (str, list)):
                try:
                    # Try to get the first value if it's a 1-element container
                    if len(x) == 1:
                        if hasattr(x, 'iloc'):
                            x = x.iloc[0]
                        else:
                            x = x[0]
                    else:
                        return []
                except:
                    return []
            
            # Handle None, NaN, etc.
            if x is None or (isinstance(x, float) and pd.isna(x)):
                return []
            
            # Handle empty string
            if isinstance(x, str) and x.strip() == '':
                return []
                
            # Already a list
            if isinstance(x, list):
                return x
                
            # String processing
            try:
                if isinstance(x, str):
                    # Check if it's a string representation of a list
                    x_clean = x.strip()
                    if x_clean.startswith("[") and x_clean.endswith("]"):
                        # Safely evaluate
                        try:
                            result = eval(x_clean)
                            if isinstance(result, list):
                                return result
                        except:
                            pass
                    
                    # Try comma separation
                    return [s.strip() for s in x.split(",") if s.strip()]
            except Exception:
                pass
                
            return []

        # --- Required columns ---
        required_cols = [
            "country", "company_name", "standardized_title", "seniority_level",
            "skills", "published", "job_type", "work_mode"
        ]
        
        # Verify all required columns exist
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # --- Preprocess ---
        df = df.copy()
        
        # Handle missing values intelligently
        for col in required_cols:
            # Convert empty strings to NaN
            df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
        
        # Only require essential columns (published and skills are truly required)
        df = df.dropna(subset=['published', 'skills'])
        
        # Fill categorical columns with defaults
        default_fills = {
            'seniority_level': 'Unspecified',
            'job_type': 'Unspecified',
            'work_mode': 'Unspecified',
            'country': 'Unknown',
            'company_name': 'Unknown',
            'standardized_title': 'Unknown'
        }
        
        for col, default in default_fills.items():
            if col in df.columns:
                df[col] = df[col].fillna(default)
        
        # Ensure a stable unique identifier for each job
        if "job_id" not in df.columns:
            df["job_id"] = df.index.astype(str)

        # Normalize and explode skills
        df["skills_list"] = df["skills"].apply(convert_to_list)
        
        # Standardize skill names (lowercase, strip)
        df["skills_list"] = df["skills_list"].apply(
            lambda lst: list(set([str(s).lower().strip() for s in lst if s and str(s).strip()]))
        )
        
        # Remove rows with empty skills_list after cleaning
        df = df[df["skills_list"].apply(len) > 0]
        
        # Explode skills
        df = df.explode("skills_list").dropna(subset=["skills_list"])

        # --- Grouping after expansion ---
        grouped = (
            df.groupby(
                ["published", "country", "company_name", "job_type", "work_mode",
                "standardized_title", "seniority_level", "skills_list"]
            )
            .agg(
                job_ids=("job_id", lambda x: list(set(x))),
                total_jobs=("job_id", "nunique")
            )
            .reset_index()
        )

        # --- Compute mentions and prevalence per skill in each group ---
        grouped["mentions"] = grouped["job_ids"].apply(len)
        grouped["prevalence"] = (
            grouped["mentions"]
            / grouped.groupby(
                ["published", "country", "company_name", "job_type", "work_mode",
                "standardized_title", "seniority_level"]
            )["mentions"].transform("sum")
        ) * 100

        # --- Rename for clarity ---
        pivot_df = grouped.rename(columns={"skills_list": "skill"})

        # Add skill categories if available
        if hasattr(self, 'SKILL_CATEGORIES'):
            skill_to_category = {
                skill.lower(): cat
                for cat, skills in self.SKILL_CATEGORIES.items()
                for skill in skills
            }
            pivot_df["skill_category"] = pivot_df["skill"].map(skill_to_category)
        else:
            pivot_df["skill_category"] = "Unknown"

        # Save if directory exists
        if hasattr(self, 'processed_dir'):
            pivot_df.to_parquet(self.processed_dir/'skill_pivot.parquet', index=False)

        return pivot_df
    
    def aggregate_pivot(self, filtered_df: pd.DataFrame, column: str = "skill", metric: str = "mentions") -> pd.DataFrame:
        """Vectorized aggregation - avoid multiple explosions"""
        if column not in filtered_df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        # Single explosion for the entire operation
        exploded = filtered_df.explode('job_ids')[['job_ids', column]].dropna()
        
        # Single aggregation pass
        agg = (exploded.groupby(column)['job_ids']
               .nunique()
               .reset_index(name='mentions'))
        
        # Calculate prevalence
        total_jobs = exploded['job_ids'].nunique()
        agg['prevalence'] = (agg['mentions'] / total_jobs) * 100
        agg['total_jobs'] = total_jobs
        
        return agg.sort_values(metric, ascending=False).reset_index(drop=True)        