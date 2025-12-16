"""Skill frequency and combination analysis"""
import pandas as pd
from collections import Counter
from itertools import combinations
from tqdm import tqdm
from src.core.base_analyzer import BaseAnalyzer
from src.core.data_processor import DataProcessor

class SkillAnalyzer(DataProcessor):
    def analyze_skill_frequency(self, pivot_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Analyze overall skill frequency across the entire dataset"""
        return self.aggregate_pivot(pivot_df, column="skill", metric="mentions").head(top_n)
    
    def analyze_skill_categories(self, pivot_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Analyze skill categories frequency"""
        return self.aggregate_pivot(pivot_df, column="skill_category", metric="mentions").head(top_n)
    
    def analyze_skill_combination_prevalence(self, pivot_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Ultra-fast skill combination analysis"""
        self.logger.info("🔗 Ultra-fast Skill Combination Prevalence Analysis...")
        skill_pairs = self.prepare_skill_combinations_fast(pivot_df, min_mentions=3, top_n=top_n * 2)
        
        if skill_pairs.empty:
            return pd.DataFrame()
        
        # Pre-compute job sets for all unique skills
        all_skills = set(skill_pairs['skill_1']).union(set(skill_pairs['skill_2']))
        job_skills = pivot_df.explode('job_ids')[['job_ids', 'skill']].dropna()
        skill_jobs_map = {}
        
        for skill in tqdm(all_skills, desc="Pre-computing skill job sets"):
            skill_jobs_map[skill] = set(job_skills[job_skills['skill'] == skill]['job_ids'])
        
        # Process combinations
        results = []
        for _, pair in tqdm(skill_pairs.iterrows(), total=len(skill_pairs), desc="Processing combinations"):
            skill1, skill2 = pair['skill_1'], pair['skill_2']
            combo_type = self._categorize_fast(skill1, skill2)
            synergy = self._assess_synergy_fast(skill1, skill2, skill_jobs_map)
            
            results.append({
                'skill_1': skill1,
                'skill_2': skill2,
                'mentions': pair['mentions'],
                'prevalence': pair['prevalence'],
                'combination_type': combo_type,
                'learning_synergy': synergy,
            })
        
        return pd.DataFrame(results).head(top_n)
    
    def prepare_skill_combinations_fast(self, pivot_df: pd.DataFrame, 
                                      combination_size: int = 2,
                                      min_mentions: int = 10,
                                      top_n: int = 20) -> pd.DataFrame:
        """Ultra-fast skill combination analysis using vectorized operations"""
        # Implementation from original script...
        pass
    
    def _categorize_fast(self, skill1: str, skill2: str) -> str:
        """Ultra-fast categorization using pre-existing mapping"""
        cat1 = self.skill_to_category.get(skill1.lower(), "Other")
        cat2 = self.skill_to_category.get(skill2.lower(), "Other")
        
        if cat1 == cat2:
            return f"Same Domain: {cat1}"
        else:
            return f"Cross-Domain: {cat1} + {cat2}"
    
    def _assess_synergy_fast(self, skill1: str, skill2: str, skill_jobs_map: dict) -> str:
        """Ultra-fast synergy assessment using pre-computed job sets"""
        jobs1 = skill_jobs_map.get(skill1, set())
        jobs2 = skill_jobs_map.get(skill2, set())
        
        if not jobs1 or not jobs2:
            return "Unknown (insufficient data)"
        
        jobs_with_both = jobs1 & jobs2
        total_jobs_with_either = len(jobs1 | jobs2)
        
        if total_jobs_with_either == 0:
            return "Unknown"
        
        co_occurrence_rate = len(jobs_with_both) / total_jobs_with_either
        
        if co_occurrence_rate > 0.4:
            return "Very High Synergy"
        elif co_occurrence_rate > 0.25:
            return "High Synergy"
        elif co_occurrence_rate > 0.1:
            return "Medium Synergy"
        else:
            return "Low Synergy"