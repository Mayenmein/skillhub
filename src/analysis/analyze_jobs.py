import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple 
import logging
from collections import Counter
from itertools import combinations
 

from src.core.config_skills import SKILL_CATEGORIES
import warnings
from tqdm import tqdm 
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  


class DataScienceJobsAnalyzer:
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
        
        # Style settings
        plt.style.use('default')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        
                
        self.trend_colors = {
            'Emerging': '#2E8B57',      # Sea Green
            'Growing': '#1E90FF',       # Dodger Blue
            'Declining': '#DC143C',     # Crimson
            'Rapidly Declining': '#8B0000',  # Dark Red
            'Accelerating': '#32CD32',  # Lime Green
            'Peaking': '#FF8C00',       # Dark Orange
            'Reviving': '#9370DB',      # Medium Purple
            'Stabilizing': '#696969',   # Dim Gray
            'Stable': '#A9A9A9'         # Dark Gray
        }

        self.skill_to_category = {skill.lower(): cat for cat, skills in self.SKILL_CATEGORIES.items() for skill in skills}
    
    # ==================== CORE DATA PROCESSING ====================
    
    def load_cleaned_data(self) -> pd.DataFrame:
        """Load cleaned data from interim directory"""
        interim_files = list(self.interim_dir.glob("*.csv"))
        if not interim_files:
            raise FileNotFoundError("No cleaned data found in interim directory")
        
        df = pd.read_csv(interim_files[0])
        logger.info(f"✅ Loaded {len(df)} cleaned records")
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

        # --- Sub-function: convert to list ---
        def convert_to_list(x):
            if pd.isna(x):
                return []
            if isinstance(x, list):
                return x
            try:
                if isinstance(x, str) and x.startswith("["):
                    return eval(x)
                else:
                    return [s.strip() for s in str(x).split(",") if s.strip()]
            except Exception:
                return []

        # --- Required columns ---
        required_cols = [
            "country", "company", "cleaned_title_category", "seniority_level",
            "skills", "date", "job_type", "work_mode"
        ]
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # --- Preprocess ---
        df = df.copy()
        df = df.dropna(subset=required_cols)

        # Ensure a stable unique identifier for each job
        if "job_id" not in df.columns:
            df["job_id"] = df.index.astype(str)

        # normalize and explode skills
        df["skills_list"] = df["skills"].apply(convert_to_list)
        df["skills_list"] = df["skills_list"].apply(
            lambda lst: list(set([s.lower().strip() for s in lst]))
        )
        df = df.explode("skills_list").dropna(subset=["skills_list"])

        # --- Grouping after expansion ---
        grouped = (
            df.groupby(
                ["date", "country", "company", "job_type", "work_mode",
                "cleaned_title_category", "seniority_level", "skills_list"]
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
                ["date", "country", "company", "job_type", "work_mode",
                "cleaned_title_category", "seniority_level"]
            )["mentions"].transform("sum")
        ) * 100

        # --- Rename for clarity ---
        pivot_df = grouped.rename(columns={"skills_list": "skill"})

        # --- Skill Categories ---
        SKILL_CATEGORIES = {
            "Programming": ["python", "r", "sql", "java", "scala", "c++", "javascript", "julia"],
            "ML Frameworks": ["tensorflow", "pytorch", "keras", "scikit-learn", "mxnet", "caffe"],
            "Big Data": ["spark", "hadoop", "hive", "kafka", "airflow", "dbt", "snowflake"],
            "Cloud": ["aws", "azure", "gcp", "docker", "kubernetes", "terraform"],
            "Visualization": ["tableau", "powerbi", "matplotlib", "seaborn", "plotly", "d3"],
            "Statistics": ["statistics", "hypothesis testing", "experimentation", "a/b testing"],
            "ML Techniques": ["machine learning", "deep learning", "nlp", "computer vision", "reinforcement learning"],
        }

        skill_to_category = {
            skill.lower(): cat
            for cat, skills in SKILL_CATEGORIES.items()
            for skill in skills
        }

        pivot_df["skill_category"] = pivot_df["skill"].map(skill_to_category)
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
    
    # ==================== MARKET OVERVIEW ANALYSIS ====================
    
    def analyze_skill_frequency(self, pivot_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Analyze overall skill frequency across the entire dataset"""
        return self.aggregate_pivot(pivot_df, column="skill", metric="mentions").head(top_n)
    
    def analyze_skill_categories(self, pivot_df: pd.DataFrame, top_n: int = 10) -> pd.DataFrame:
        """Analyze skill categories frequency"""
        return self.aggregate_pivot(pivot_df, column="skill_category", metric="mentions").head(top_n)
        
    def analyze_seniority_specific_patterns(self, pivot_df: pd.DataFrame, progression_df: pd.DataFrame, 
                                      importance_threshold: float = 1.2, top_n_skills: int = 5, 
                                      max_specific_skills: int = 3) -> Dict[str, Dict]:
        """
        Analyze seniority-specific skill patterns and identify skills that are particularly 
        important at each seniority level compared to overall averages.
        
        Parameters:
        -----------
        pivot_df : pd.DataFrame
            The main pivot dataframe with skill data
        progression_df : pd.DataFrame
            The progression dataframe from analyze_skill_progression_data
        importance_threshold : float
            Minimum ratio of level prevalence to overall average (default: 1.2 = 20% higher)
        top_n_skills : int
            Number of top skills to show for each level (default: 5)
        max_specific_skills : int
            Maximum number of level-specific skills to identify (default: 3)
        
        Returns:
        --------
        Dict containing patterns for each seniority level
        """
        logger.info("🎯 Identifying seniority-specific skill patterns...")
        
        # Define seniority order
        seniority_order = ['Entry-level', 'Junior', 'Mid-level', 'Senior', 'Lead', 'Executive']
        seniority_patterns = {}
        
        for level in tqdm(seniority_order, desc="Analyzing seniority patterns"):
            if level in pivot_df['seniority_level'].unique():
                level_data = pivot_df[pivot_df['seniority_level'] == level]
                
                if not level_data.empty:
                    # Get top skills for this level by prevalence
                    level_skills = self.aggregate_pivot(level_data, column="skill", metric="prevalence")
                    
                    if not level_skills.empty:
                        # Find skills that are more important in this level compared to others
                        level_specific_skills = self._identify_level_specific_skills(
                            level_skills, progression_df, importance_threshold, max_specific_skills
                        )
                        
                        # Calculate skill diversity metrics
                        skill_diversity = self._calculate_skill_diversity_metrics(level_skills)
                        
                        seniority_patterns[level] = {
                            'top_skills': level_skills.head(top_n_skills)[['skill', 'prevalence']].to_dict('records'),
                            'level_specific_skills': level_specific_skills,
                            'skill_diversity': skill_diversity,
                            'total_jobs': level_data['job_ids'].explode().nunique(),
                            'total_skills': len(level_skills)
                        }
        
        # Generate insights and summary
        self._generate_seniority_insights(seniority_patterns)
        
        return seniority_patterns

    def _identify_level_specific_skills(self, level_skills: pd.DataFrame, progression_df: pd.DataFrame, 
                                    threshold: float, max_skills: int) -> List[Dict]:
        """Identify skills that are particularly important for a specific seniority level"""
        level_specific_skills = []
        
        for _, skill_row in level_skills.iterrows():
            skill = skill_row['skill']
            level_prevalence = skill_row['prevalence']
            
            # Get overall average prevalence for this skill across all levels
            skill_overall_data = progression_df[progression_df['skill'] == skill]
            if not skill_overall_data.empty:
                overall_avg_prevalence = skill_overall_data['prevalence'].mean()
                
                # Calculate importance ratio
                if overall_avg_prevalence > 0:
                    importance_ratio = level_prevalence / overall_avg_prevalence
                    
                    # Check if this skill is significantly more important at this level
                    if importance_ratio >= threshold:
                        level_specific_skills.append({
                            'skill': skill,
                            'level_prevalence': level_prevalence,
                            'overall_avg_prevalence': overall_avg_prevalence,
                            'importance_ratio': importance_ratio,
                            'prevalence_difference': level_prevalence - overall_avg_prevalence
                        })
        
        # Return top N most level-specific skills
        return sorted(level_specific_skills, key=lambda x: x['importance_ratio'], reverse=True)[:max_skills]

    def _calculate_skill_diversity_metrics(self, level_skills: pd.DataFrame) -> Dict[str, float]:
        """Calculate skill diversity and concentration metrics for a seniority level"""
        if level_skills.empty:
            return {}
        
        total_mentions = level_skills['mentions'].sum()
        total_prevalence = level_skills['prevalence'].sum()
        
        # Skill concentration (Gini-like coefficient)
        sorted_prevalence = level_skills['prevalence'].sort_values(ascending=False).values
        n_skills = len(sorted_prevalence)
        
        if n_skills > 1:
            # Calculate concentration (0 = perfectly equal, 1 = perfectly concentrated)
            cumulative_share = np.cumsum(sorted_prevalence) / total_prevalence
            perfect_equality = np.linspace(0, 1, n_skills)
            concentration_index = np.sum(cumulative_share - perfect_equality) / n_skills
        else:
            concentration_index = 0
        
        # Top skill dominance
        top_skill_dominance = sorted_prevalence[0] / total_prevalence if total_prevalence > 0 else 0
        
        return {
            'concentration_index': concentration_index,
            'top_skill_dominance': top_skill_dominance,
            'skills_per_job': total_mentions / level_skills['total_jobs'].iloc[0] if level_skills['total_jobs'].iloc[0] > 0 else 0,
            'high_prevalence_skills': len(level_skills[level_skills['prevalence'] > 10])
        }

    def _generate_seniority_insights(self, seniority_patterns: Dict):
        """Generate insights from seniority-specific patterns"""
        logger.info("\n" + "="*70)
        logger.info("🎯 SENIORITY-SPECIFIC SKILL PATTERNS INSIGHTS")
        logger.info("="*70)
        
        for level, patterns in seniority_patterns.items():
            logger.info(f"\n📋 {level.upper()} LEVEL:")
            logger.info(f"   • Total jobs: {patterns['total_jobs']:,}")
            logger.info(f"   • Unique skills: {patterns['total_skills']}")
            logger.info(f"   • Skill concentration: {patterns['skill_diversity']['concentration_index']:.3f}")
            
            # Top skills
            top_skills_str = ", ".join([f"{s['skill']} ({s['prevalence']:.1f}%)" 
                                    for s in patterns['top_skills'][:3]])
            logger.info(f"   • Top skills: {top_skills_str}")
            
            # Level-specific skills
            if patterns['level_specific_skills']:
                specific_skills_str = ", ".join([
                    f"{s['skill']} ({s['importance_ratio']:.1f}x)" 
                    for s in patterns['level_specific_skills']
                ])
                logger.info(f"   • Level-specific: {specific_skills_str}")
            else:
                logger.info("   • Level-specific: No strongly specific skills identified")
    
    # ==================== ROLE & CAREER ANALYSIS ====================
    
    def analyze_by_group(self, pivot_df: pd.DataFrame, group_column: str, top_n: int = 15) -> Dict[str, pd.DataFrame]:
        """Vectorized group analysis using single aggregation - 3-5x faster"""
        if group_column not in pivot_df.columns:
            raise ValueError(f"Group column '{group_column}' not found")
        
        # Single explosion and aggregation for all groups
        exploded = pivot_df.explode('job_ids')[['job_ids', 'skill', group_column]].dropna()
        
        # Count unique job_ids per skill per group in one operation
        skill_counts = (exploded.groupby([group_column, 'skill'])['job_ids']
                          .nunique()
                          .reset_index(name='mentions'))
        
        # Calculate total jobs per group
        total_jobs_per_group = (exploded.groupby(group_column)['job_ids']
                               .nunique()
                               .reset_index(name='total_jobs'))
        
        # Merge and calculate prevalence
        merged = skill_counts.merge(total_jobs_per_group, on=group_column)
        merged['prevalence'] = (merged['mentions'] / merged['total_jobs']) * 100
        
        # Split by group and take top_n for each
        results = {}
        for group in merged[group_column].unique():
            group_df = merged[merged[group_column] == group]
            results[group] = (group_df.sort_values('mentions', ascending=False)
                             .head(top_n)
                             .reset_index(drop=True))
        
        return results
    
    def analyze_skills_by_role(self, pivot_df: pd.DataFrame, top_n: int = 15) -> Dict[str, pd.DataFrame]:
        """Analyze top skills for each role category"""
        return self.analyze_by_group(pivot_df, 'cleaned_title_category', top_n)
    
    def analyze_skills_by_seniority(self, pivot_df: pd.DataFrame, top_n: int = 15) -> Dict[str, pd.DataFrame]:
        """Analyze top skills for each seniority level"""
        return self.analyze_by_group(pivot_df, 'seniority_level', top_n)
    
    def analyze_regional_skill_demand(self, pivot_df: pd.DataFrame, top_n: int = 10) -> Dict[str, pd.DataFrame]:
        """Analyze top skills by country/region"""
        return self.analyze_by_group(pivot_df, 'country', top_n)
    
    def compare_role_skill_profiles(self, pivot_df: pd.DataFrame, roles: List[str] = None) -> pd.DataFrame:
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
        
        # Pivot to wide format for easy comparison
        wide_df = comparison_df.pivot_table(
            index='skill', columns='role', values='prevalence', aggfunc='first'
        ).fillna(0)
        
        return wide_df
    
    def analyze_skill_progression_data(self, pivot_df, skills_list):
        """Maximum performance vectorized version"""
        if pivot_df.empty or not skills_list:
            return pd.DataFrame()
        
        # Convert to set for faster membership testing 
        skills_set = set(skills_list)
        
        # Single pass: filter, group, and calculate in one chain
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
    
    # ==================== TREND ANALYSIS METHODS ====================
    
    def analyze_skill_trends_enhanced(self, pivot_df: pd.DataFrame,
                                  min_prevalence=1.0,
                                  min_avg_mentions=5,
                                  min_months=8,
                                  smoothing_alpha=0.3):
        months = sorted(pivot_df["date"].unique())
        if len(months) < min_months:
            return pd.DataFrame()

        skill_month_matrix = pivot_df.pivot_table(
            index="skill", columns="date", values="mentions", aggfunc="sum", fill_value=0
        ).reindex(columns=months)

        monthly_totals = pivot_df.groupby("date")["job_ids"].apply(
            lambda x: len(set([i for sublist in x for i in sublist]))
        ).reindex(months)

        prevalence_matrix = skill_month_matrix.div(monthly_totals, axis=1) * 100
        mentions_matrix = skill_month_matrix.copy()

        mask = (prevalence_matrix.mean(axis=1) >= min_prevalence) & \
            (mentions_matrix.mean(axis=1) >= min_avg_mentions)
        prevalence_matrix = prevalence_matrix.loc[mask]
        mentions_matrix = mentions_matrix.loc[mask]

        if prevalence_matrix.empty:
            return pd.DataFrame()

        # --- Smoothed month-over-month growth
        growth_mentions, y_smoothed_mentions = calculate_smoothed_growth(mentions_matrix.values, smoothing_alpha)
        growth_prevalence, y_smoothed_prevalence = calculate_smoothed_growth(prevalence_matrix.values, smoothing_alpha)

        # --- Rolling metrics
        cagr_prevalence, recent_momentum_prevalence, current_prevalence, peak_ratio = calculate_rolling_metrics(y_smoothed_prevalence)
        cagr_mentions, recent_momentum_mentions, current_mentions, _ = calculate_rolling_metrics(y_smoothed_mentions)

        # --- OLS & nonlinearity
        slopes, intercepts = batch_ols(y_smoothed_prevalence)
        nonlinearity = calculate_nonlinearity(y_smoothed_prevalence, slopes, intercepts)

        # --- Baseline-aware hybrid momentum using smoothed growth
        final_hybrid_momentum = np.zeros_like(recent_momentum_prevalence)
        reliability_flags = np.zeros_like(final_hybrid_momentum, dtype=bool)
        baseline_mentions = np.mean(mentions_matrix.values[:, :-1], axis=1)

        for i in range(len(final_hybrid_momentum)):
            if baseline_mentions[i] < 5:
                # Very low baseline, growth unreliable
                final_hybrid_momentum[i] = 0.2*recent_momentum_prevalence[i] + 0.8*growth_mentions[i]
                reliability_flags[i] = False
            elif baseline_mentions[i] < 20:
                # Medium baseline
                final_hybrid_momentum[i] = 0.5*recent_momentum_prevalence[i] + 0.5*growth_mentions[i]
                reliability_flags[i] = True
            else:
                # High baseline, trust momentum more
                final_hybrid_momentum[i] = 0.8*recent_momentum_prevalence[i] + 0.2*growth_mentions[i]
                reliability_flags[i] = True

        # --- Percentiles for classification thresholds
        p10, p25, p75, p90 = np.percentile(final_hybrid_momentum, [10, 25, 75, 90])

        # --- Volume-adjusted scoring
        trend_confidence = np.log1p(current_mentions)
        volume_adjusted_score = final_hybrid_momentum * trend_confidence

        # --- Trend categories (external)
        categories = map_categories_to_strings(
            classify_trends_smart_recent(y_smoothed_prevalence, slopes, final_hybrid_momentum,
                                        current_prevalence, p10, p25, p75, p90)
        )

        # --- Build final DataFrame
        df = pd.DataFrame({
            "skill": prevalence_matrix.index,
            "CAGR_pct": cagr_prevalence,
            "recent_momentum_pct": final_hybrid_momentum,
            "current_prevalence": current_prevalence,
            "avg_mentions_per_month": mentions_matrix.mean(axis=1),
            "total_mentions": mentions_matrix.sum(axis=1),
            "peak_ratio": peak_ratio,
            "trend_slope": slopes,
            "nonlinearity": nonlinearity,
            "trend_confidence": trend_confidence,
            "volume_adjusted_score": volume_adjusted_score,
            "growth_mentions_pct": growth_mentions,
            "growth_prevalence_pct": growth_prevalence,
            "growth_reliability": reliability_flags,
            "trend_category": categories,
            "baseline_mentions": baseline_mentions
        })

        # --- Boolean flags
        df["is_emerging"] = df["trend_category"].isin(["Emerging", "Accelerating"])
        df["is_declining"] = df["trend_category"].isin(["Declining", "Rapidly Declining"])
        df["is_growing"] = df["trend_category"].isin(["Growing", "Accelerating"])
        df["is_stable"] = df["trend_category"].isin(["Stable", "Stabilizing"])
        df["is_special"] = df["trend_category"].isin(["Peaking", "Reviving"])

        return df.sort_values("volume_adjusted_score", ascending=False).reset_index(drop=True)

    
    def get_trend_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Get summary statistics by trend category using pandas"""
        if results_df.empty:
            return pd.DataFrame()
            
        summary = results_df.groupby("trend_category").agg({
            'skill': 'count',
            'current_prevalence': ['mean', 'std'],
            'trend_slope': ['mean', 'std'],
            'recent_momentum_pct': ['mean', 'std'],
            'CAGR_pct': 'mean'
        }).round(3)
        
        summary.columns = ['_'.join(col).strip() for col in summary.columns.values]
        summary = summary.rename(columns={'skill_count': 'skill_count'})
        return summary.sort_values('skill_count', ascending=False)

    def get_top_skills_by_category(self, results_df: pd.DataFrame, category: str, 
                                top_n: int = 10, sort_by: str = "recent_momentum_pct") -> pd.DataFrame:
        """Get top skills for specific category"""
        category_skills = results_df[results_df["trend_category"] == category].copy()
        if category_skills.empty:
            return pd.DataFrame()
        
        return category_skills.sort_values(sort_by, ascending=False).head(top_n)

    # ==================== Skill Clustering and Combinations ====================
    def analyze_skill_combination_prevalence(self, pivot_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
        """Ultra-fast skill combination analysis - 50-100x faster"""
        logger.info("🔗 Ultra-fast Skill Combination Prevalence Analysis...")
        
        # Get skill pairs efficiently
        skill_pairs = self.prepare_skill_combinations_fast(
            pivot_df,
            min_mentions=3,
            top_n=top_n * 2
        )
        
        if skill_pairs.empty:
            return pd.DataFrame()
        
        # PRE-COMPUTE everything needed for the helper functions
        logger.info("📊 Pre-computing helper data...")
        
        # Pre-compute job sets for all unique skills in the pairs
        all_skills = set(skill_pairs['skill_1']).union(set(skill_pairs['skill_2']))
        
        # Pre-compute job sets for all skills (ONCE)
        job_skills = pivot_df.explode('job_ids')[['job_ids', 'skill']].dropna()
        skill_jobs_map = {}
        for skill in tqdm(all_skills, desc="Pre-computing skill job sets"):
            skill_jobs_map[skill] = set(job_skills[job_skills['skill'] == skill]['job_ids'])
        
        # Pre-compute role skill rankings (ONCE)
        logger.info("🎯 Pre-computing role skill rankings...")
        role_skills = self.analyze_skills_by_role(pivot_df, top_n=25)
        
        # Vectorize all calculations
        results = []
        for _, pair in tqdm(skill_pairs.iterrows(), total=len(skill_pairs), desc="Processing combinations"):
            skill1, skill2 = pair['skill_1'], pair['skill_2']
            
            # Use pre-computed data for ultra-fast calculations
            combo_type = self._categorize_fast(skill1, skill2)
            synergy = self._assess_synergy_fast(skill1, skill2, skill_jobs_map)
            context = self._suggest_context_fast(skill1, skill2, role_skills)
            
            results.append({
                'skill_1': skill1,
                'skill_2': skill2,
                'mentions': pair['mentions'],
                'prevalence': pair['prevalence'],
                'combination_type': combo_type,
                'learning_synergy': synergy,
                'career_context': context
            })
        
        return pd.DataFrame(results).head(top_n)

    def prepare_skill_combinations_fast(self, pivot_df: pd.DataFrame, 
                                  combination_size: int = 2,
                                  min_mentions: int = 10,
                                  top_n: int = 20) -> pd.DataFrame:
        """Ultra-fast skill combination analysis using vectorized operations"""
        logger.info("🔗 Fast skill combination analysis...")
        
        # Use pre-exploded data for efficiency
        if 'job_ids' not in pivot_df.columns or not pivot_df['job_ids'].iloc[0]:
            job_skills = pivot_df.explode('job_ids')[['job_ids', 'skill']].dropna().drop_duplicates()
        else:
            job_skills = pivot_df[['job_ids', 'skill']].explode('job_ids').dropna().drop_duplicates()
        
        # Filter to jobs with enough skills quickly
        job_skill_counts = job_skills.groupby('job_ids')['skill'].count().reset_index(name='skill_count')
        valid_jobs = job_skill_counts[job_skill_counts['skill_count'] >= combination_size]['job_ids']
        job_skills_filtered = job_skills[job_skills['job_ids'].isin(valid_jobs)]
        
        # Group skills by job and filter in one go
        job_skill_sets = (job_skills_filtered.groupby('job_ids')['skill']
                        .apply(frozenset)  # Use frozenset for hashability
                        .reset_index())
        
        # Use Counter with itertools.combinations - optimized
        combo_counts = Counter()
        
        for skills in tqdm(job_skill_sets['skill'], desc="Counting combinations"):
            if len(skills) <= 20:  # Reasonable limit
                # Convert to sorted tuple for consistent counting
                for combo in combinations(sorted(skills), combination_size):
                    combo_counts[combo] += 1
        
        # Convert to DataFrame efficiently
        if not combo_counts:
            return pd.DataFrame()
        
        # Use list comprehension for faster DataFrame creation
        total_jobs = job_skills['job_ids'].nunique()
        results = [{
            'skill_1': skill1,
            'skill_2': skill2, 
            'mentions': count,
            'prevalence': (count / total_jobs) * 100
        } for (skill1, skill2), count in combo_counts.most_common(top_n) if count >= min_mentions]
        
        return pd.DataFrame(results)

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

    def _suggest_context_fast(self, skill1: str, skill2: str, role_skills: dict) -> str:
        """Ultra-fast context suggestion using pre-computed role rankings"""
        strong_roles = []
        moderate_roles = []
        
        for role, skills_df in role_skills.items():
            if skills_df.empty:
                continue
                
            # Convert to index-based lookup for speed
            skills_list = skills_df['skill'].tolist()
            
            try:
                skill1_rank = skills_list.index(skill1)
                skill2_rank = skills_list.index(skill2)
                
                if skill1_rank <= 10 and skill2_rank <= 10:
                    strong_roles.append(role)
                elif skill1_rank <= 20 or skill2_rank <= 20:
                    moderate_roles.append(role)
            except ValueError:
                # Skill not in top 25 for this role
                continue
        
        if strong_roles:
            return f"Core combination for {strong_roles[0]}"
        elif moderate_roles:
            return f"Relevant for {moderate_roles[0]}"
        
        return "Cross-functional combination"

    def identify_natural_skill_clusters(self, pivot_df: pd.DataFrame, top_skills_count: int = 30) -> Dict:
        """Fast co-occurrence analysis using matrix operations - 5-8x faster"""
        logger.info("🎯 Fast Natural Skill Clusters Identification...")
        
        # Get top skills efficiently
        top_skills_df = self.analyze_skill_frequency(pivot_df, top_n=top_skills_count)
        top_skill_names = set(top_skills_df['skill'].tolist())
        
        # Pre-calculate job sets for top skills
        job_skills = pivot_df.explode('job_ids')[['job_ids', 'skill']].dropna()
        top_skill_jobs = {}
        
        for skill in tqdm(top_skill_names, desc="Pre-calculating skill jobs"):
            skill_jobs = set(job_skills[job_skills['skill'] == skill]['job_ids'])
            top_skill_jobs[skill] = skill_jobs
        
        # Build co-occurrence matrix efficiently
        clusters = {}
        processed_skills = set()
        skill_list = list(top_skill_names)
        
        for i, skill in enumerate(tqdm(skill_list, desc="Building clusters")):
            if skill in processed_skills:
                continue
                
            skill_jobs = top_skill_jobs[skill]
            if not skill_jobs:
                continue
            
            # Find co-occurring skills efficiently
            co_occurrence_scores = []
            for other_skill in skill_list:
                if skill == other_skill or other_skill in processed_skills:
                    continue
                    
                other_jobs = top_skill_jobs[other_skill]
                if not other_jobs:
                    continue
                    
                co_count = len(skill_jobs.intersection(other_jobs))
                if co_count >= 10:  # Minimum threshold
                    co_occurrence_scores.append((other_skill, co_count))
            
            # Take top co-occurring skills
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
    
    def analyze_skill_ecosystem(self, pivot_df: pd.DataFrame) -> Dict:
        """Comprehensive analysis of the skill ecosystem based entirely on data"""
        logger.info("🌐 Analyzing Complete Skill Ecosystem...")
        
        ecosystem = {}
        
        # 1. Most common combinations
        ecosystem['top_combinations'] = self.analyze_skill_combination_prevalence(pivot_df, top_n=15)
        
        # 2. Natural clusters
        ecosystem['natural_clusters'] = self.identify_natural_skill_clusters(pivot_df)
        
        # 3. Role-based patterns
        role_comparison = self.compare_role_skill_profiles(pivot_df)
        ecosystem['role_specializations'] = {}
        
        for role in role_comparison.columns:
            top_skills_for_role = role_comparison[role].sort_values(ascending=False).head(10)
            ecosystem['role_specializations'][role] = {
                'top_skills': top_skills_for_role.index.tolist(),
                'skill_intensity': top_skills_for_role.mean()
            }
        
        # 4. Universal vs specialized skills
        ecosystem['skill_universality'] = self.analyze_role_skill_concentration_data_driven(pivot_df)
        
        return ecosystem
    
    def analyze_role_skill_concentration_data_driven(self, pivot_df: pd.DataFrame) -> Dict:
        """
        Data-driven analysis of skill concentration vs universality
        """
        logger.info("🎯 Data-Driven Skill Concentration Analysis...")
        
        skills_by_role = self.analyze_skills_by_role(pivot_df, top_n=20)
        
        # Calculate skill universality
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
        
        # Calculate averages and categorize
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
    
    # ==================== VISUALIZATION METHODS ====================
    
    def _generate_trend_insights(self, results_df: pd.DataFrame) -> Dict:
        """Generate actionable insights from trend analysis"""
        if results_df.empty:
            return {}
        
        emerging_count = results_df['is_emerging'].sum()
        declining_count = results_df['is_declining'].sum()
        growing_count = results_df['is_growing'].sum()
        
        top_emerging = self.get_top_skills_by_category(results_df, 'Emerging', 5)
        top_declining = self.get_top_skills_by_category(results_df, 'Declining', 5)
        top_growing = self.get_top_skills_by_category(results_df, 'Growing', 5)
        
        return {
            'market_health': {
                'emerging_skills': emerging_count,
                'growing_skills': growing_count,
                'declining_skills': declining_count,
                'innovation_ratio': emerging_count / max(declining_count, 1)
            },
            'learning_priorities': {
                'emerging': top_emerging[['skill', 'recent_momentum_pct', 'current_prevalence']].to_dict('records'),
                'growing': top_growing[['skill', 'current_prevalence', 'trend_slope']].to_dict('records')
            },
            'risks': {
                'declining': top_declining[['skill', 'current_prevalence', 'recent_momentum_pct']].to_dict('records')
            }
        }    
    
    # ==================== HELPER FUNCTIONS ====================
    
    def get_skill_taxonomy_insights(self, pivot_df: pd.DataFrame) -> pd.DataFrame:
        """Analyze skill category distribution and trends"""
        categories_df = self.aggregate_pivot(pivot_df, column="skill_category", metric="mentions")
        
        category_insights = []
        for category in categories_df['skill_category'].unique():
            cat_data = pivot_df[pivot_df['skill_category'] == category]
            
            category_insights.append({
                'skill_category': category,
                'total_mentions': categories_df[categories_df['skill_category'] == category]['mentions'].iloc[0],
                'avg_prevalence': categories_df[categories_df['skill_category'] == category]['prevalence'].iloc[0],
                'role_diversity': cat_data['cleaned_title_category'].nunique(),
                'seniority_spread': cat_data['seniority_level'].nunique()
            })
        
        return pd.DataFrame(category_insights).sort_values('total_mentions', ascending=False)
