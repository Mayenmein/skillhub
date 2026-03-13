"""Seniority-specific skill pattern analysis"""
import pandas as pd
import numpy as np
from tqdm import tqdm
import logging
from src.analysis.skill_analyzer import SkillAnalyzer
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  
class SeniorityAnalyzer(SkillAnalyzer): 
        
    def analyze_seniority_specific_patterns(self, pivot_df: pd.DataFrame, progression_df: pd.DataFrame, 
                                      importance_threshold: float = 1.2, top_n_skills: int = 5, 
                                      max_specific_skills: int = 3) -> dict:
        """Analyze seniority-specific skill patterns"""
        logger.info("Identifying seniority-specific skill patterns...")
        seniority_patterns = {}
        
        for level in tqdm(self.seniority_order, desc="Analyzing seniority patterns"):
            if level in pivot_df['seniority_level'].unique():
                level_data = pivot_df[pivot_df['seniority_level'] == level]
                
                if not level_data.empty:
                    level_skills = self.aggregate_pivot(level_data, column="skill", metric="prevalence")
                    
                    if not level_skills.empty:
                        level_specific_skills = self._identify_level_specific_skills(
                            level_skills, progression_df, importance_threshold, max_specific_skills
                        )
                        
                        skill_diversity = self._calculate_skill_diversity_metrics(level_skills)
                        
                        seniority_patterns[level] = {
                            'top_skills': level_skills.head(top_n_skills)[['skill', 'prevalence']].to_dict('records'),
                            'level_specific_skills': level_specific_skills,
                            'skill_diversity': skill_diversity,
                            'total_jobs': level_data['job_ids'].explode().nunique(),
                            'total_skills': len(level_skills)
                        }
        
        self._generate_seniority_insights(seniority_patterns)
        return seniority_patterns
    
    def _identify_level_specific_skills(self, level_skills: pd.DataFrame, progression_df: pd.DataFrame, 
                                    threshold: float, max_skills: int) -> list:
        """Identify skills that are particularly important for a specific seniority level"""
        level_specific_skills = []
        
        for _, skill_row in level_skills.iterrows():
            skill = skill_row['skill']
            level_prevalence = skill_row['prevalence']
            
            skill_overall_data = progression_df[progression_df['skill'] == skill]
            if not skill_overall_data.empty:
                overall_avg_prevalence = skill_overall_data['prevalence'].mean()
                
                if overall_avg_prevalence > 0:
                    importance_ratio = level_prevalence / overall_avg_prevalence
                    
                    if importance_ratio >= threshold:
                        level_specific_skills.append({
                            'skill': skill,
                            'level_prevalence': level_prevalence,
                            'overall_avg_prevalence': overall_avg_prevalence,
                            'importance_ratio': importance_ratio,
                            'prevalence_difference': level_prevalence - overall_avg_prevalence
                        })
        
        return sorted(level_specific_skills, key=lambda x: x['importance_ratio'], reverse=True)[:max_skills]
    
    def _calculate_skill_diversity_metrics(self, level_skills: pd.DataFrame) -> dict:
        """Calculate skill diversity and concentration metrics for a seniority level"""
        if level_skills.empty:
            return {}
        
        total_mentions = level_skills['mentions'].sum()
        total_prevalence = level_skills['prevalence'].sum()
        sorted_prevalence = level_skills['prevalence'].sort_values(ascending=False).values
        n_skills = len(sorted_prevalence)
        
        if n_skills > 1:
            cumulative_share = np.cumsum(sorted_prevalence) / total_prevalence
            perfect_equality = np.linspace(0, 1, n_skills)
            concentration_index = np.sum(cumulative_share - perfect_equality) / n_skills
        else:
            concentration_index = 0
        
        top_skill_dominance = sorted_prevalence[0] / total_prevalence if total_prevalence > 0 else 0
        
        return {
            'concentration_index': concentration_index,
            'top_skill_dominance': top_skill_dominance,
            'skills_per_job': total_mentions / level_skills['total_jobs'].iloc[0] if level_skills['total_jobs'].iloc[0] > 0 else 0,
            'high_prevalence_skills': len(level_skills[level_skills['prevalence'] > 10])
        }
    
    def _generate_seniority_insights(self, seniority_patterns: dict):
        """Generate insights from seniority-specific patterns"""
        logger.info("\n" + "="*70)
        logger.info("SENIORITY-SPECIFIC SKILL PATTERNS INSIGHTS")
        logger.info("="*70)
        
        for level, patterns in seniority_patterns.items():
            logger.info(f"\n{level.upper()} LEVEL:")
            logger.info(f"   • Total jobs: {patterns['total_jobs']:,}")
            logger.info(f"   • Unique skills: {patterns['total_skills']}")
            logger.info(f"   • Skill concentration: {patterns['skill_diversity']['concentration_index']:.3f}")
            
            top_skills_str = ", ".join([f"{s['skill']} ({s['prevalence']:.1f}%)" 
                                    for s in patterns['top_skills'][:3]])
            logger.info(f"   • Top skills: {top_skills_str}")
            
            if patterns['level_specific_skills']:
                specific_skills_str = ", ".join([
                    f"{s['skill']} ({s['importance_ratio']:.1f}x)" 
                    for s in patterns['level_specific_skills']
                ])
                logger.info(f"   • Level-specific: {specific_skills_str}")
            else:
                logger.info("   • Level-specific: No strongly specific skills identified")