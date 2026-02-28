"""Skill ecosystem and clustering analysis"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging 
from src.analysis.role_analyzer import RoleAnalyzer
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  
class EcosystemAnalyzer(RoleAnalyzer):
    def identify_natural_skill_clusters(self, pivot_df: pd.DataFrame, top_skills_count: int = 30) -> dict:
        """Fast co-occurrence analysis using matrix operations"""
        logger.info("Fast Natural Skill Clusters Identification...")
        
        top_skills_df = self.analyze_skill_frequency(pivot_df, top_n=top_skills_count)
        top_skill_names = set(top_skills_df['skill'].tolist())
        
        job_skills = pivot_df.explode('job_ids')[['job_ids', 'skill']].dropna()
        top_skill_jobs = {}
        
        for skill in tqdm(top_skill_names, desc="Pre-calculating skill jobs"):
            skill_jobs = set(job_skills[job_skills['skill'] == skill]['job_ids'])
            top_skill_jobs[skill] = skill_jobs
        
        clusters = {}
        processed_skills = set()
        skill_list = list(top_skill_names)
        
        for i, skill in enumerate(tqdm(skill_list, desc="Building clusters")):
            if skill in processed_skills:
                continue
                
            skill_jobs = top_skill_jobs[skill]
            if not skill_jobs:
                continue
            
            co_occurrence_scores = []
            for other_skill in skill_list:
                if skill == other_skill or other_skill in processed_skills:
                    continue
                    
                other_jobs = top_skill_jobs[other_skill]
                if not other_jobs:
                    continue
                    
                co_count = len(skill_jobs.intersection(other_jobs))
                if co_count >= 10:
                    co_occurrence_scores.append((other_skill, co_count))
            
            if co_occurrence_scores:
                co_occurrence_scores.sort(key=lambda x: x[1], reverse=True)
                cluster_skills = [skill] + [s for s, _ in co_occurrence_scores[:4]]
                
                cluster_name = f"Cluster_{len(clusters) + 1}"
                clusters[cluster_name] = {
                    'core_skill': skill,
                    'skills': cluster_skills,
                    'avg_co_occurrence': np.mean([score for _, score in co_occurrence_scores[:4]]),
                    'size': len(cluster_skills)
                }
                
                processed_skills.update(cluster_skills)
        
        return clusters
    
    def analyze_skill_ecosystem(self, pivot_df: pd.DataFrame) -> dict:
        """Comprehensive analysis of the skill ecosystem based entirely on data"""
        logger.info("Analyzing Complete Skill Ecosystem...")
        
        ecosystem = {}
        ecosystem['top_combinations'] = self.analyze_skill_combination_prevalence(pivot_df, top_n=15)
        ecosystem['natural_clusters'] = self.identify_natural_skill_clusters(pivot_df)
        
        role_comparison = self.compare_role_skill_profiles(pivot_df)
        ecosystem['role_specializations'] = {}
        
        for role in role_comparison.columns:
            top_skills_for_role = role_comparison[role].sort_values(ascending=False).head(10)
            ecosystem['role_specializations'][role] = {
                'top_skills': top_skills_for_role.index.tolist(),
                'skill_intensity': top_skills_for_role.mean()
            }
        
        ecosystem['skill_universality'] = self.analyze_role_skill_concentration_data_driven(pivot_df)
        return ecosystem
    
    def analyze_role_skill_concentration_data_driven(self, pivot_df: pd.DataFrame) -> dict:
        """Data-driven analysis of skill concentration vs universality"""
        logger.info("Data-Driven Skill Concentration Analysis...")
        
        skills_by_role = self.analyze_skills_by_role(pivot_df, top_n=20)
        skill_universality = {}
        total_roles = len(skills_by_role)
        
        for role, skills_df in skills_by_role.items():
            for _, skill_row in skills_df.iterrows():
                skill = skill_row['skill']
                if skill not in skill_universality:
                    skill_universality[skill] = {
                        'roles': [],
                        'total_prevalence': 0,
                        'avg_prevalence': 0
                    }
                skill_universality[skill]['roles'].append(role)
                skill_universality[skill]['total_prevalence'] += skill_row['prevalence']
        
        universal_skills = []
        specialized_skills = []
        
        for skill, data in skill_universality.items():
            role_count = len(data['roles'])
            avg_prevalence = data['total_prevalence'] / role_count if role_count > 0 else 0
            
            if role_count >= total_roles * 0.6 and avg_prevalence > 15:
                universal_skills.append({
                    'skill': skill,
                    'role_coverage': f"{role_count}/{total_roles} roles",
                    'avg_prevalence': avg_prevalence
                })
            elif role_count == 1 and avg_prevalence > 25:
                specialized_skills.append({
                    'skill': skill,
                    'primary_role': data['roles'][0],
                    'prevalence': avg_prevalence
                })
        
        return {
            'universal_skills': sorted(universal_skills, key=lambda x: x['avg_prevalence'], reverse=True)[:10],
            'specialized_skills': sorted(specialized_skills, key=lambda x: x['prevalence'], reverse=True)[:10]
        }