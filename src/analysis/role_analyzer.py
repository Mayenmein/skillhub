"""Role-based skill analysis"""
import pandas as pd
from src.analysis.analyze_jobs import BaseAnalyzer
from src.analysis.skill_analyzer import SkillAnalyzer

class RoleAnalyzer(SkillAnalyzer):
    def analyze_by_group(self, pivot_df: pd.DataFrame, group_column: str, top_n: int = 15) -> dict:
        """Vectorized group analysis using single aggregation"""
        if group_column not in pivot_df.columns:
            raise ValueError(f"Group column '{group_column}' not found")
        
        exploded = pivot_df.explode('job_ids')[['job_ids', 'skill', group_column]].dropna()
        skill_counts = (exploded.groupby([group_column, 'skill'])['job_ids']
                          .nunique()
                          .reset_index(name='mentions'))
        
        total_jobs_per_group = (exploded.groupby(group_column)['job_ids']
                               .nunique()
                               .reset_index(name='total_jobs'))
        
        merged = skill_counts.merge(total_jobs_per_group, on=group_column)
        merged['prevalence'] = (merged['mentions'] / merged['total_jobs']) * 100
        
        results = {}
        for group in merged[group_column].unique():
            group_df = merged[merged[group_column] == group]
            results[group] = (group_df.sort_values('mentions', ascending=False)
                             .head(top_n)
                             .reset_index(drop=True))
        
        return results
    
    def analyze_skills_by_role(self, pivot_df: pd.DataFrame, top_n: int = 15) -> dict:
        """Analyze top skills for each role category"""
        return self.analyze_by_group(pivot_df, 'standardized_title', top_n)
    
    def analyze_skills_by_seniority(self, pivot_df: pd.DataFrame, top_n: int = 15) -> dict:
        """Analyze top skills for each seniority level"""
        return self.analyze_by_group(pivot_df, 'seniority_level', top_n)
    
    def analyze_regional_skill_demand(self, pivot_df: pd.DataFrame, top_n: int = 10) -> dict:
        """Analyze top skills by country/region"""
        return self.analyze_by_group(pivot_df, 'country', top_n)
    
    def compare_role_skill_profiles(self, pivot_df: pd.DataFrame, roles: list = None) -> pd.DataFrame:
        """Compare skill profiles across different roles"""
        role_skills = self.analyze_skills_by_role(pivot_df, top_n=20)
        
        if roles:
            role_skills = {k: v for k, v in role_skills.items() if k in roles}
        
        comparison_data = []
        for role, skills_df in role_skills.items():
            for _, row in skills_df.head(10).iterrows():
                comparison_data.append({
                    'skill': row['skill'], 'role': role,
                    'prevalence': row['prevalence'], 'mentions': row['mentions']
                })
        
        comparison_df = pd.DataFrame(comparison_data)
        wide_df = comparison_df.pivot_table(
            index='skill', columns='role', values='prevalence', aggfunc='first'
        ).fillna(0)
        
        return wide_df
    
    def analyze_skill_progression_data(self, pivot_df, skills_list):
        """Maximum performance vectorized version"""
        if pivot_df.empty or not skills_list:
            return pd.DataFrame()
        
        skills_set = set(skills_list)
        result_df = (pivot_df[pivot_df['skill'].isin(skills_set)]
                    .groupby(['skill', 'seniority_level'])
                    .size()
                    .reset_index(name='mentions')
                    .merge(
                        pivot_df.explode('job_ids')
                        .groupby('seniority_level')['job_ids']
                        .nunique()
                        .reset_index(name='total_jobs'),
                        on='seniority_level'
                    )
                    .assign(prevalence=lambda x: (x['mentions'] / x['total_jobs']) * 100)
                    [['skill', 'seniority_level', 'prevalence', 'mentions', 'total_jobs']]
                    )
        
        return result_df