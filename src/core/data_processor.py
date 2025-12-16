"""Data loading and preprocessing"""
import pandas as pd
from pathlib import Path
from typing import List
from src.core.base_analyzer import BaseAnalyzer

class DataProcessor(BaseAnalyzer):
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load cleaned data from interim directory"""
        interim_files = list(self.interim_dir.glob("*.csv"))
        if not interim_files:
            raise FileNotFoundError("No cleaned data found in interim directory")
        
        df = pd.read_csv(interim_files[0])
        self.logger.info(f"✅ Loaded {len(df)} cleaned records")
        return df
    
    def create_skill_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        """Build a pivot-style DataFrame preserving job-level granularity"""
        # Required columns check
        required_cols = [
            "country", "company", "cleaned_title_category", "seniority_level",
            "skills", "date", "job_type", "work_mode"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        df = df.copy().dropna(subset=required_cols)
        
        # Create job_id if not exists
        if "job_id" not in df.columns:
            df["job_id"] = df.index.astype(str)
        
        # Normalize and explode skills
        df["skills_list"] = df["skills"].apply(self._convert_to_list)
        df["skills_list"] = df["skills_list"].apply(
            lambda lst: list(set([s.lower().strip() for s in lst]))
        )
        df = df.explode("skills_list").dropna(subset=["skills_list"])
        
        # Grouping after expansion
        grouped = (
            df.groupby([
                "date", "country", "company", "job_type", "work_mode",
                "cleaned_title_category", "seniority_level", "skills_list"
            ])
            .agg(
                job_ids=("job_id", lambda x: list(set(x))),
                total_jobs=("job_id", "nunique")
            )
            .reset_index()
        )
        
        # Compute mentions and prevalence
        grouped["mentions"] = grouped["job_ids"].apply(len)
        grouped["prevalence"] = (
            grouped["mentions"] / grouped.groupby([
                "date", "country", "company", "job_type", "work_mode",
                "cleaned_title_category", "seniority_level"
            ])["mentions"].transform("sum")
        ) * 100
        
        pivot_df = grouped.rename(columns={"skills_list": "skill"})
        pivot_df["skill_category"] = pivot_df["skill"].map(self.skill_to_category)
        
        # Save processed data
        pivot_df.to_parquet(self.processed_dir / 'skill_pivot.parquet', index=False)
        return pivot_df
    
    def aggregate_pivot(self, filtered_df: pd.DataFrame, column: str = "skill", metric: str = "mentions") -> pd.DataFrame:
        """Vectorized aggregation - avoid multiple explosions"""
        if column not in filtered_df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        exploded = filtered_df.explode('job_ids')[['job_ids', column]].dropna()
        agg = (exploded.groupby(column)['job_ids']
               .nunique()
               .reset_index(name='mentions'))
        
        total_jobs = exploded['job_ids'].nunique()
        agg['prevalence'] = (agg['mentions'] / total_jobs) * 100
        agg['total_jobs'] = total_jobs
        
        return agg.sort_values(metric, ascending=False).reset_index(drop=True)