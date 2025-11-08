import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from collections import Counter
from itertools import combinations

from scipy import stats
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_absolute_error

import warnings
from tqdm import tqdm 
warnings.filterwarnings('ignore')

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__) 
from numba import njit, prange
from statsmodels.api import OLS, add_constant
from scipy import stats

# -------------------------
# 1. Exponential smoothing
# -------------------------
@njit(fastmath=True, parallel=True)
def exp_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    n_rows, n_cols = y.shape
    out = np.empty_like(y)
    for i in prange(n_rows):
        current = y[i, 0]
        out[i, 0] = current
        for j in range(1, n_cols):
            current = alpha * y[i, j] + (1 - alpha) * current
            out[i, j] = current
    return out

# -------------------------
# 2. Metrics (CAGR, momentum, prevalence, peak ratio)
# -------------------------
@njit(fastmath=True, parallel=True)
def calculate_rolling_metrics(y_smoothed: np.ndarray, window: int = 3):
    n_skills, n_time = y_smoothed.shape
    cagr = np.zeros(n_skills)
    recent_momentum = np.zeros(n_skills)
    current_prevalence = np.zeros(n_skills)
    peak_ratio = np.zeros(n_skills)

    for i in prange(n_skills):
        start = y_smoothed[i, 0]
        end = y_smoothed[i, -1]
        n_periods = n_time - 1
        cagr[i] = ((end / start) ** (12 / n_periods) - 1) * 100 if start > 0 else np.nan

        # Recent momentum
        if n_time >= 2*window:
            recent_avg = 0.0
            prev_avg = 0.0
            for j in range(n_time - window, n_time):
                recent_avg += y_smoothed[i, j]
            recent_avg /= window
            for j in range(n_time - 2*window, n_time - window):
                prev_avg += y_smoothed[i, j]
            prev_avg /= window
            recent_momentum[i] = ((recent_avg - prev_avg)/prev_avg*100) if prev_avg > 0 else 0
        else:
            recent_momentum[i] = 0

        current_prevalence[i] = y_smoothed[i, -1]

        # Peak prevalence
        peak = y_smoothed[i, 0]
        for j in range(1, n_time):
            if y_smoothed[i, j] > peak:
                peak = y_smoothed[i, j]
        peak_ratio[i] = current_prevalence[i] / (peak + 1e-12)

    return cagr, recent_momentum, current_prevalence, peak_ratio

# -------------------------
# 3. OLS slopes and intercepts
# -------------------------
@njit(fastmath=True, parallel=True)
def batch_ols(y_smoothed: np.ndarray):
    n_skills, n_time = y_smoothed.shape
    slopes = np.zeros(n_skills)
    intercepts = np.zeros(n_skills)

    x = np.arange(n_time)
    X = np.empty((n_time, 2))
    X[:, 0] = 1.0
    X[:, 1] = x

    XT_X = X.T @ X
    XT_X_inv = np.linalg.inv(XT_X)
    for i in prange(n_skills):
        beta = XT_X_inv @ X.T @ y_smoothed[i, :]
        intercepts[i] = beta[0]
        slopes[i] = beta[1]

    return slopes, intercepts

# -------------------------
# 4. Nonlinearity
# -------------------------
@njit(fastmath=True, parallel=True)
def calculate_nonlinearity(y_smoothed: np.ndarray, slopes: np.ndarray, intercepts: np.ndarray):
    n_skills, n_time = y_smoothed.shape
    nonlinearity = np.zeros(n_skills)
    for i in prange(n_skills):
        linear_pred = intercepts[i] + slopes[i] * np.arange(n_time)
        residuals = y_smoothed[i, :] - linear_pred
        mean_abs = 0.0
        var_sum = 0.0
        n = residuals.shape[0]
        for j in range(n):
            mean_abs += abs(residuals[j])
        mean_abs /= n
        if mean_abs > 0:
            mean_val = 0.0
            for j in range(n):
                mean_val += residuals[j]
            mean_val /= n
            for j in range(n):
                var_sum += (residuals[j] - mean_val)**2
            nonlinearity[i] = np.sqrt(var_sum / (n-1)) / mean_abs
        else:
            nonlinearity[i] = 0
    return nonlinearity

# -------------------------
# 5. Skill-wise percentile helpers (Numba)
# -------------------------
@njit(fastmath=True)
def percentile_1d(arr: np.ndarray, q: float) -> float:
    """Compute percentile manually for 1D array in Numba"""
    sorted_arr = np.sort(arr)
    idx = int(q * (len(arr) - 1))
    return sorted_arr[idx]

# -------------------------
# 6. Skill-wise classification
# -------------------------

@njit
def calculate_recent_percentile(y_smoothed: np.ndarray, window: int = 6) -> np.ndarray:
    """
    Calculate what percentile the latest month is within the recent window
    Returns array where each skill has value 0-1 representing percentile in recent history
    """
    n_skills, n_time = y_smoothed.shape
    recent_percentiles = np.zeros(n_skills)
    
    for i in range(n_skills):
        if n_time >= window:
            # Get the last 'window' months including current month
            recent_data = y_smoothed[i, -window:]
            current_val = recent_data[-1]
            
            # Calculate percentile within recent window
            count_below = 0
            for j in range(window):
                if recent_data[j] <= current_val:
                    count_below += 1
            recent_percentiles[i] = count_below / window
        else:
            # Not enough data, use 0.5 as neutral
            recent_percentiles[i] = 0.5
            
    return recent_percentiles

# Mapping integers to trend strings
TREND_MAP = ["Stable", "Emerging", "Growing", "Declining", "Mature",
             "Peaking", "Reviving", "Accelerating", "Stabilizing", "Rapidly Declining"]

@njit
def classify_trends_smart_recent(y_smoothed: np.ndarray, slopes: np.ndarray,
                               recent_momentum: np.ndarray, current_prevalence: np.ndarray) -> np.ndarray:
    """
    Balanced trend classification using recent 6-month context and adjusted thresholds.
    """
    n_skills, n_time = y_smoothed.shape
    categories = np.zeros(n_skills, dtype=np.int8)
    
    recent_percentiles = calculate_recent_percentile(y_smoothed, window=6)
    
    for i in range(n_skills):
        current_val = current_prevalence[i]
        slope = slopes[i]
        momentum = recent_momentum[i]
        recent_pct = recent_percentiles[i]
        
        # High prevalence skills (>25%)
        if current_val > 25.0:
            if slope < -0.3:
                categories[i] = 3  # Declining
            elif slope < -0.1 and recent_pct < 0.4:
                categories[i] = 4  # Mature 
            elif slope > 0.3 and recent_pct > 0.7:
                categories[i] = 2  # Growing
            elif slope > 0.2 and recent_pct > 0.8:
                categories[i] = 5  # Peaking
            else:
                categories[i] = 0  # Stable
        
        # Medium prevalence skills (10-25%)
        elif current_val > 10.0:
            if slope > 0.8 and momentum > 5.0:
                categories[i] = 7  # Accelerating
            elif slope > 0.3 and recent_pct > 0.6:
                categories[i] = 2  # Growing
            elif slope < -0.3 and momentum < -3.0:
                categories[i] = 9  # Rapidly Declining
            elif slope < -0.2:
                categories[i] = 3  # Declining
            elif slope > 0.2 and recent_pct > 0.6:
                categories[i] = 6  # Reviving
            elif abs(slope) < 0.1 and recent_pct > 0.5:
                categories[i] = 8  # Stabilizing
            else:
                categories[i] = 0  # Stable
        
        # Low prevalence skills (<10%)
        else:
            if slope > 0.5 and momentum > 8.0:
                categories[i] = 1  # Emerging
            elif slope > 0.3 and recent_pct > 0.6:
                categories[i] = 2  # Growing
            elif slope < -0.3 and momentum < -5.0:
                categories[i] = 9  # Rapidly Declining
            elif slope < -0.2:
                categories[i] = 3  # Declining
            elif slope > 0.2 and recent_pct > 0.7:
                categories[i] = 6  # Reviving
            elif abs(slope) < 0.1 and recent_pct < 0.4:
                categories[i] = 8  # Stabilizing
            else:
                categories[i] = 0  # Stable
                
    return categories

# Map integer codes back to strings after Numba
def map_categories_to_strings(categories_int: np.ndarray) -> np.ndarray:
    n = len(categories_int)
    categories_str = np.empty(n, dtype=object)
    for i in range(n):
        categories_str[i] = TREND_MAP[categories_int[i]]
    return categories_str

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
        
        # Skill categories mapping
        self.SKILL_CATEGORIES = {
            'Programming': ['python', 'r', 'sql', 'java', 'scala', 'c++', 'javascript', 'julia'],
            'ML Frameworks': ['tensorflow', 'pytorch', 'keras', 'scikit-learn', 'mxnet', 'caffe'],
            'Big Data': ['spark', 'hadoop', 'hive', 'kafka', 'airflow', 'dbt', 'snowflake'],
            'Cloud': ['aws', 'azure', 'gcp', 'docker', 'kubernetes', 'terraform'],
            'Visualization': ['tableau', 'powerbi', 'matplotlib', 'seaborn', 'plotly', 'd3'],
            'Statistics': ['statistics', 'hypothesis testing', 'experimentation', 'a/b testing'],
            'ML Techniques': ['machine learning', 'deep learning', 'nlp', 'computer vision', 'reinforcement learning']
        }
        
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
    
    def plot_skills_bar(self, agg_df: pd.DataFrame, column = 'skill', top_n: int = 15, metric: str = "mentions", title: str = None):
        """Plot top skills from an aggregated DataFrame"""
        if agg_df.empty:
            logger.warning("⚠️ No data available for plotting.")
            return
        
        plot_df = agg_df.sort_values(by=metric, ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 6))
        ax = sns.barplot(data=plot_df, y=column, x=metric, palette="viridis")
        
        column = 'Skill' if column == 'skill' else 'Skill Category'
        title = title or f"Top {top_n} {column}s by {metric.capitalize()}"
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.set_xlabel("Number of Mentions" if metric == "mentions" else "Prevalence (% of Jobs)")
        ax.set_ylabel(column)
        
        # Annotate values on bars
        for bar in ax.patches:
            width = bar.get_width()
            y = bar.get_y() + bar.get_height() / 2
            ax.text(width + plot_df[metric].max()*0.01, y,
                    f"{width:.1f}" + ("%" if metric == "prevalence" else ""),
                    va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
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

    def plot_seniority_patterns(self, seniority_patterns: Dict, figsize: tuple = (15, 10)):
        """Visualize seniority-specific skill patterns"""
        if not seniority_patterns:
            logger.warning("No seniority patterns data available for plotting")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
        
        # Plot 1: Top skills by seniority level
        self._plot_top_skills_by_level(ax1, seniority_patterns)
        
        # Plot 2: Level-specific skill importance
        self._plot_level_specific_skills(ax2, seniority_patterns)
        
        # Plot 3: Skill diversity metrics
        self._plot_skill_diversity(ax3, seniority_patterns)
        
        # Plot 4: Seniority level comparison
        self._plot_seniority_comparison(ax4, seniority_patterns)
        
        plt.tight_layout()
        plt.show()

    def _plot_top_skills_by_level(self, ax, seniority_patterns: Dict):
        """Plot top skills for each seniority level"""
        levels = list(seniority_patterns.keys())
        skills_data = {}
        
        for level in levels:
            top_skills = seniority_patterns[level]['top_skills']
            for skill_data in top_skills[:3]:  # Top 3 skills per level
                skill = skill_data['skill']
                prevalence = skill_data['prevalence']
                if skill not in skills_data:
                    skills_data[skill] = {level: prevalence}
                else:
                    skills_data[skill][level] = prevalence
        
        # Create stacked bar chart
        skill_names = list(skills_data.keys())
        level_positions = np.arange(len(levels))
        bar_width = 0.8 / len(skill_names)
        
        for i, skill in enumerate(skill_names):
            values = [skills_data[skill].get(level, 0) for level in levels]
            ax.bar(level_positions + i * bar_width, values, bar_width, label=skill, alpha=0.8)
        
        ax.set_xticks(level_positions + bar_width * len(skill_names) / 2)
        ax.set_xticklabels(levels, rotation=45)
        ax.set_ylabel('Prevalence (%)')
        ax.set_title('Top Skills by Seniority Level', fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.grid(True, alpha=0.3)

    def _plot_level_specific_skills(self, ax, seniority_patterns: Dict):
        """Plot level-specific skill importance ratios"""
        levels = list(seniority_patterns.keys())
        
        for level in levels:
            specific_skills = seniority_patterns[level]['level_specific_skills']
            if specific_skills:
                skills = [s['skill'] for s in specific_skills]
                ratios = [s['importance_ratio'] for s in specific_skills]
                
                y_pos = np.arange(len(skills))
                bars = ax.barh(y_pos, ratios, alpha=0.7, label=level)
                
                # Add value labels
                for bar, ratio in zip(bars, ratios):
                    ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2, 
                        f'{ratio:.1f}x', va='center', fontweight='bold')
        
        ax.set_xlabel('Importance Ratio (Level vs Overall Average)')
        ax.set_title('Level-Specific Skill Importance', fontweight='bold')
        ax.legend()

    def _plot_skill_diversity(self, ax, seniority_patterns: Dict):
        """Plot skill diversity metrics across seniority levels"""
        levels = list(seniority_patterns.keys())
        concentration = [p['skill_diversity']['concentration_index'] for p in seniority_patterns.values()]
        dominance = [p['skill_diversity']['top_skill_dominance'] for p in seniority_patterns.values()]
        
        x = np.arange(len(levels))
        width = 0.35
        
        ax.bar(x - width/2, concentration, width, label='Concentration Index', alpha=0.7)
        ax.bar(x + width/2, dominance, width, label='Top Skill Dominance', alpha=0.7)
        
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=45)
        ax.set_ylabel('Metric Value')
        ax.set_title('Skill Diversity Across Seniority Levels', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)

    def _plot_seniority_comparison(self, ax, seniority_patterns: Dict):
        """Plot comprehensive seniority level comparison"""
        levels = list(seniority_patterns.keys())
        total_jobs = [p['total_jobs'] for p in seniority_patterns.values()]
        total_skills = [p['total_skills'] for p in seniority_patterns.values()]
        skills_per_job = [p['skill_diversity']['skills_per_job'] for p in seniority_patterns.values()]
        
        x = np.arange(len(levels))
        
        # Normalize for comparison
        jobs_norm = [j/max(total_jobs) for j in total_jobs]
        skills_norm = [s/max(total_skills) for s in total_skills]
        skills_per_job_norm = [spj/max(skills_per_job) for spj in skills_per_job]
        
        ax.plot(x, jobs_norm, 'o-', label='Job Volume (normalized)', linewidth=2)
        ax.plot(x, skills_norm, 's-', label='Skill Variety (normalized)', linewidth=2)
        ax.plot(x, skills_per_job_norm, '^-', label='Skills per Job (normalized)', linewidth=2)
        
        ax.set_xticks(x)
        ax.set_xticklabels(levels, rotation=45)
        ax.set_ylabel('Normalized Values')
        ax.set_title('Seniority Level Characteristics', fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
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
    
    def plot_skill_progression_organized(self, progression_df: pd.DataFrame, key_skills: List[str] = None, max_skills: int = 6):
        """
        Organized plot of skill progression across seniority levels
        Handles data from analyze_skill_progression_data output
        """
        if progression_df.empty:
            logger.warning("No progression data available for plotting")
            return
        
        # Define proper seniority order
        seniority_order = ['Entry-level', 'Junior', 'Mid-level', 'Senior', 'Lead', 'Executive']
        
        # Clean and prepare data
        progression_clean = progression_df.dropna(subset=['seniority_level', 'prevalence'])
        
        # Convert seniority_level to categorical with proper order
        progression_clean['seniority_level'] = pd.Categorical(
            progression_clean['seniority_level'], 
            categories=[level for level in seniority_order if level in progression_clean['seniority_level'].unique()],
            ordered=True
        )
        
        # Determine which skills to plot
        if key_skills is None:
            # Get top skills by average prevalence across all levels
            skill_avg_prevalence = (progression_clean.groupby('skill')['prevalence']
                                .mean()
                                .sort_values(ascending=False))
            key_skills = skill_avg_prevalence.head(max_skills).index.tolist()
        else:
            key_skills = [skill for skill in key_skills if skill in progression_clean['skill'].unique()]
            key_skills = key_skills[:max_skills]
        
        if not key_skills:
            logger.warning("No valid skills found for plotting")
            return
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: Line plot for progression trends
        self._plot_progression_lines(ax1, progression_clean, key_skills, seniority_order)
        
        # Plot 2: Heatmap for comprehensive view
        self._plot_progression_heatmap(ax2, progression_clean, key_skills, seniority_order)
        
        plt.tight_layout()
        plt.show()
        
        # Print data summary
        self._print_progression_summary(progression_clean, key_skills)

    def _plot_progression_lines(self, ax, data: pd.DataFrame, key_skills: List[str], seniority_order: List[str]):
        """Plot skill progression as line chart"""
        
        # Filter to available seniority levels in the data
        available_levels = [level for level in seniority_order if level in data['seniority_level'].cat.categories]
        
        # Define a color palette
        colors = plt.cm.Set3(np.linspace(0, 1, len(key_skills)))
        
        for i, skill in enumerate(key_skills):
            skill_data = data[data['skill'] == skill].sort_values('seniority_level')
            
            if not skill_data.empty:
                # Get prevalence values in correct order
                prevalence_values = []
                for level in available_levels:
                    level_data = skill_data[skill_data['seniority_level'] == level]
                    if not level_data.empty:
                        prevalence_values.append(level_data['prevalence'].iloc[0])
                    else:
                        prevalence_values.append(0)  # Fill missing with 0
                
                # Plot line
                line = ax.plot(range(len(available_levels)), prevalence_values, 
                            marker='o', linewidth=2.5, markersize=8, 
                            label=skill, color=colors[i], alpha=0.8)
                
                # Add data labels for significant values
                for j, prevalence in enumerate(prevalence_values):
                    if prevalence >= 1.0:  # Only label values above 1% for clarity
                        ax.annotate(f'{prevalence:.1f}%', 
                                (j, prevalence), 
                                textcoords="offset points", 
                                xytext=(0, 8), 
                                ha='center', 
                                fontsize=9,
                                fontweight='bold' if prevalence > 10 else 'normal')
        
        ax.set_xticks(range(len(available_levels)))
        ax.set_xticklabels(available_levels, rotation=45, ha='right')
        ax.set_title('Skill Prevalence Across Seniority Levels\n(Line View)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Seniority Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Prevalence (% of Jobs)', fontsize=12, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', frameon=True)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_ylim(bottom=0)  # Start from 0 for better visualization

    def _plot_progression_heatmap(self, ax, data: pd.DataFrame, key_skills: List[str], seniority_order: List[str]):
        """Plot skill progression as heatmap"""
        
        # Create pivot table for heatmap
        heatmap_data = data.pivot_table(
            index='skill', 
            columns='seniority_level', 
            values='prevalence', 
            aggfunc='first'
        ).fillna(0)
        
        # Reindex to ensure proper order
        available_levels = [level for level in seniority_order if level in heatmap_data.columns]
        heatmap_data = heatmap_data[available_levels]
        
        # Filter to key skills and sort by average prevalence
        heatmap_data = heatmap_data.loc[key_skills]
        skill_order = heatmap_data.mean(axis=1).sort_values(ascending=False).index
        heatmap_data = heatmap_data.loc[skill_order]
        
        # Create heatmap
        im = ax.imshow(heatmap_data, cmap='YlOrRd', aspect='auto', interpolation='nearest')
        
        # Set ticks and labels
        ax.set_xticks(range(len(available_levels)))
        ax.set_xticklabels(available_levels, rotation=45, ha='right')
        ax.set_yticks(range(len(heatmap_data.index)))
        ax.set_yticklabels(heatmap_data.index)
        
        # Add value annotations
        for i in range(len(heatmap_data.index)):
            for j in range(len(available_levels)):
                value = heatmap_data.iloc[i, j]
                if value > 0:  # Only show non-zero values
                    text_color = 'white' if value > heatmap_data.values.max() * 0.6 else 'black'
                    ax.text(j, i, f'{value:.1f}%', 
                        ha='center', va='center', 
                        fontsize=9, fontweight='bold',
                        color=text_color)
        
        ax.set_title('Skill Prevalence Heatmap\n(Color Intensity View)', 
                    fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Seniority Level', fontsize=12, fontweight='bold')
        ax.set_ylabel('Skill', fontsize=12, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Prevalence (%)', fontsize=10, fontweight='bold')

    def _print_progression_summary(self, data: pd.DataFrame, key_skills: List[str]):
        """Print summary statistics for the plotted skills"""
        print("\n" + "="*60)
        print("📊 SKILL PROGRESSION SUMMARY")
        print("="*60)
        
        for skill in key_skills:
            skill_data = data[data['skill'] == skill]
            if not skill_data.empty:
                max_level = skill_data.loc[skill_data['prevalence'].idxmax(), 'seniority_level']
                max_value = skill_data['prevalence'].max()
                min_value = skill_data['prevalence'].min()
                avg_value = skill_data['prevalence'].mean()
                
                print(f"\n🔹 {skill}:")
                print(f"   • Peak: {max_value:.1f}% ({max_level})")
                print(f"   • Range: {min_value:.1f}% - {max_value:.1f}%")
                print(f"   • Average: {avg_value:.1f}%")
                
                # Identify progression pattern
                if max_level in ['Entry-level', 'Junior']:
                    print(f"   • Pattern: 📉 Early specialization (declines with seniority)")
                elif max_level in ['Lead', 'Executive']:
                    print(f"   • Pattern: 📈 Leadership-focused (grows with seniority)")
                else:
                    print(f"   • Pattern: 📊 Core competency (stable across levels)")

    # Simplified plotting function for quick use
    def plot_skill_progression_simple(self, progression_df: pd.DataFrame, skills_to_plot: List[str] = None):
        """
        Simple version of progression plot for quick visualization
        """
        if progression_df.empty:
            return

        # Use provided skills or top 6 by average prevalence
        if skills_to_plot is None:
            avg_prevalence = progression_df.groupby('skill')['prevalence'].mean()
            skills_to_plot = avg_prevalence.nlargest(6).index.tolist()
        
        seniority_order = ['Entry-level', 'Junior', 'Mid-level', 'Senior', 'Lead', 'Executive']
        
        # Clean and order data
        clean_data = progression_df.dropna(subset=['seniority_level', 'prevalence']).copy()
        clean_data['seniority_level'] = pd.Categorical(
            clean_data['seniority_level'], 
            categories=[level for level in seniority_order if level in clean_data['seniority_level'].unique()],
            ordered=True
        )
        
        plt.figure(figsize=(12, 8))
        
        # Plot each skill
        for skill in skills_to_plot:
            skill_data = clean_data[clean_data['skill'] == skill].sort_values('seniority_level')
            
            if not skill_data.empty:
                # Ensure we have data for all levels in correct order
                x_positions = []
                y_values = []
                
                for level in clean_data['seniority_level'].cat.categories:
                    level_data = skill_data[skill_data['seniority_level'] == level]
                    if not level_data.empty:
                        x_positions.append(level)
                        y_values.append(level_data['prevalence'].iloc[0])
                
                if x_positions:  # Only plot if we have data
                    plt.plot(range(len(x_positions)), y_values, 
                            marker='o', linewidth=2, markersize=6, label=skill)
        
        # Format plot
        available_levels = clean_data['seniority_level'].cat.categories.tolist()
        plt.xticks(range(len(available_levels)), available_levels, rotation=45)
        plt.title('Skill Progression Across Seniority Levels', fontsize=14, fontweight='bold')
        plt.xlabel('Seniority Level', fontsize=12)
        plt.ylabel('Prevalence (% of Jobs)', fontsize=12)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
    
    # ==================== TREND ANALYSIS METHODS ====================
    
    def analyze_skill_trends_full(self, pivot_df: pd.DataFrame, min_prevalence=1.0, min_months=8, smoothing_alpha=0.3):
        months = sorted(pivot_df["date"].unique())
        if len(months) < min_months:
            return pd.DataFrame()

        skill_month_matrix = pivot_df.pivot_table(
            index="skill", columns="date", values="mentions", aggfunc="sum", fill_value=0
        ).reindex(columns=months)

        monthly_totals = pivot_df.groupby("date")["job_ids"].apply(
            lambda x: len(set([i for sublist in x for i in sublist]))
        ).reindex(months)

        prevalence = skill_month_matrix.div(monthly_totals, axis=1) * 100
        prevalence = prevalence.loc[prevalence.mean(axis=1) >= min_prevalence]
        if prevalence.empty:
            return pd.DataFrame()

        y = prevalence.values.astype(np.float64)

        # Exponential smoothing
        y_smoothed = exp_smooth(y, smoothing_alpha)

        # Metrics
        cagr, recent_momentum, current_prevalence, peak_ratio = calculate_rolling_metrics(y_smoothed)
        slopes, intercepts = batch_ols(y_smoothed)
        nonlinearity = calculate_nonlinearity(y_smoothed, slopes, intercepts)

        # Skill-wise classification using percentiles
        categories = map_categories_to_strings(
    classify_trends_smart_recent(y_smoothed, slopes, recent_momentum, current_prevalence))

        # Build final DataFrame
        df = pd.DataFrame({
            "skill": prevalence.index,
            "CAGR_pct": cagr,
            "recent_momentum_pct": recent_momentum,
            "current_prevalence": current_prevalence,
            "peak_ratio": peak_ratio,
            "trend_slope": slopes,
            "nonlinearity": nonlinearity,
            "trend_category": categories
        })

        # Boolean flags
        df["is_emerging"] = df["trend_category"].isin(["Emerging", "Accelerating"])
        df["is_declining"] = df["trend_category"].isin(["Declining", "Rapidly Declining"])
        df["is_growing"] = df["trend_category"].isin(["Growing", "Accelerating"])
        df["is_stable"] = df["trend_category"].isin(["Stable", "Stabilizing"])
        df["is_special"] = df["trend_category"].isin(["Peaking", "Reviving"])

        return df.sort_values("trend_slope", ascending=False).reset_index(drop=True)
    
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

    def plot_trend_categories_distribution(self, results_df, figsize=(12, 8)):
        """Plot distribution of skills across trend categories"""
        if results_df.empty:
            print("No data to plot")
            return
            
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Count plot by category
        category_counts = results_df['trend_category'].value_counts()
        colors = [self.trend_colors.get(cat, '#999999') for cat in category_counts.index]
        
        ax1.bar(category_counts.index, category_counts.values, color=colors, alpha=0.8)
        ax1.set_title('Distribution of Skills by Trend Category', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Number of Skills')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add percentage labels
        total = len(results_df)
        for i, (category, count) in enumerate(category_counts.items()):
            ax1.text(i, count + 0.5, f'{count}\n({count/total:.1%})', 
                    ha='center', va='bottom', fontsize=9)
        
        # Pie chart
        ax2.pie(category_counts.values, labels=category_counts.index, 
                colors=colors, autopct='%1.1f%%', startangle=90)
        ax2.set_title('Trend Categories Proportion', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig

    def plot_trend_scatter_matrix(self, results_df, figsize=(14, 10)):
        """Scatter matrix showing relationships between key metrics"""
        if results_df.empty:
            print("No data to plot")
            return
            
        # Select key metrics for scatter plot
        metrics_df = results_df[['current_prevalence', 'trend_slope', 
                               'recent_momentum_pct', 'CAGR_pct', 'trend_category']].copy()
        
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.ravel()
        
        # Plot 1: Prevalence vs Trend Slope
        for category in metrics_df['trend_category'].unique():
            mask = metrics_df['trend_category'] == category
            axes[0].scatter(metrics_df.loc[mask, 'current_prevalence'], 
                          metrics_df.loc[mask, 'trend_slope'],
                          c=self.trend_colors.get(category, '#999999'),
                          label=category, alpha=0.7, s=60)
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[0].set_xlabel('Current Prevalence (%)')
        axes[0].set_ylabel('Trend Slope')
        axes[0].set_title('Prevalence vs Trend Slope')
        axes[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Plot 2: CAGR vs Recent Momentum
        for category in metrics_df['trend_category'].unique():
            mask = metrics_df['trend_category'] == category
            axes[1].scatter(metrics_df.loc[mask, 'CAGR_pct'], 
                          metrics_df.loc[mask, 'recent_momentum_pct'],
                          c=self.trend_colors.get(category, '#999999'),
                          label=category, alpha=0.7, s=60)
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        axes[1].set_xlabel('CAGR (%)')
        axes[1].set_ylabel('Recent Momentum (%)')
        axes[1].set_title('CAGR vs Recent Momentum')
        
        # Plot 3: Prevalence Distribution by Category
        category_data = []
        categories = []
        for category in metrics_df['trend_category'].unique():
            category_data.append(metrics_df[metrics_df['trend_category'] == category]['current_prevalence'])
            categories.append(category)
        
        box_plot = axes[2].boxplot(category_data, labels=categories, patch_artist=True)
        for patch, category in zip(box_plot['boxes'], categories):
            patch.set_facecolor(self.trend_colors.get(category, '#999999'))
        axes[2].tick_params(axis='x', rotation=45)
        axes[2].set_ylabel('Current Prevalence (%)')
        axes[2].set_title('Prevalence Distribution by Category')
        
        # Plot 4: Momentum Distribution by Category
        category_data = []
        for category in metrics_df['trend_category'].unique():
            category_data.append(metrics_df[metrics_df['trend_category'] == category]['recent_momentum_pct'])
        
        box_plot = axes[3].boxplot(category_data, labels=categories, patch_artist=True)
        for patch, category in zip(box_plot['boxes'], categories):
            patch.set_facecolor(self.trend_colors.get(category, '#999999'))
        axes[3].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        axes[3].tick_params(axis='x', rotation=45)
        axes[3].set_ylabel('Recent Momentum (%)')
        axes[3].set_title('Momentum Distribution by Category')
        
        plt.tight_layout()
        return fig

    def plot_skill_trend_timeline(self, pivot_df, skill_names, figsize=(15, 8)):
        """Plot timeline for specific skills"""
        if not skill_names:
            print("No skills specified")
            return
            
        fig, ax = plt.subplots(figsize=figsize)
        
        # Prepare data
        months = sorted(pivot_df["date"].unique())
        skill_month_matrix = pivot_df.pivot_table(
            index="skill", columns="date", values="mentions", aggfunc="sum", fill_value=0
        ).reindex(columns=months)
        
        monthly_totals = pivot_df.groupby("date")["job_ids"].apply(
            lambda x: len(set([i for sublist in x for i in sublist]))
        ).reindex(months)
        
        prevalence = skill_month_matrix.div(monthly_totals, axis=1) * 100
        
        # Plot each skill
        for i, skill in enumerate(skill_names):
            if skill in prevalence.index:
                skill_data = prevalence.loc[skill]
                ax.plot(months, skill_data.values, 
                       marker='o', linewidth=2, markersize=4, 
                       label=skill, alpha=0.8)
        
        ax.set_xlabel('Date')
        ax.set_ylabel('Prevalence (%)')
        ax.set_title('Skill Prevalence Over Time', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig

    def plot_emerging_vs_declining_skills(self, results_df, top_n=15, figsize=(14, 10)):
        """Compare top emerging vs declining skills"""
        if results_df.empty:
            print("No data to plot")
            return
            
        emerging_skills = results_df[results_df['is_emerging']].nlargest(top_n, 'recent_momentum_pct')
        declining_skills = results_df[results_df['is_declining']].nsmallest(top_n, 'recent_momentum_pct')
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
        
        # Emerging skills
        if not emerging_skills.empty:
            y_pos = np.arange(len(emerging_skills))
            ax1.barh(y_pos, emerging_skills['recent_momentum_pct'], 
                    color=self.trend_colors['Emerging'], alpha=0.7)
            ax1.set_yticks(y_pos)
            ax1.set_yticklabels(emerging_skills['skill'])
            ax1.set_xlabel('Recent Momentum (%)')
            ax1.set_title(f'Top {len(emerging_skills)} Emerging Skills', fontsize=12, fontweight='bold')
            ax1.axvline(x=0, color='black', linestyle='-', alpha=0.8)
            
            # Add prevalence as text
            for i, (_, row) in enumerate(emerging_skills.iterrows()):
                ax1.text(row['recent_momentum_pct'] + 0.5, i, 
                        f"{row['current_prevalence']:.1f}%", va='center', fontsize=8)
        
        # Declining skills
        if not declining_skills.empty:
            y_pos = np.arange(len(declining_skills))
            ax2.barh(y_pos, declining_skills['recent_momentum_pct'], 
                    color=self.trend_colors['Declining'], alpha=0.7)
            ax2.set_yticks(y_pos)
            ax2.set_yticklabels(declining_skills['skill'])
            ax2.set_xlabel('Recent Momentum (%)')
            ax2.set_title(f'Top {len(declining_skills)} Declining Skills', fontsize=12, fontweight='bold')
            ax2.axvline(x=0, color='black', linestyle='-', alpha=0.8)
            
            # Add prevalence as text
            for i, (_, row) in enumerate(declining_skills.iterrows()):
                ax2.text(row['recent_momentum_pct'] - 1, i, 
                        f"{row['current_prevalence']:.1f}%", va='center', 
                        ha='right', fontsize=8, color='white')
        
        plt.tight_layout()
        return fig

    def plot_trend_radar_chart(self, results_df, figsize=(10, 8)):
        """Radar chart showing average metrics by trend category"""
        if results_df.empty:
            print("No data to plot")
            return
            
        # Calculate average metrics by category
        metrics = ['current_prevalence', 'trend_slope', 'recent_momentum_pct', 'CAGR_pct', 'peak_ratio']
        categories = results_df['trend_category'].unique()
        
        # Normalize metrics for radar chart
        normalized_data = []
        for category in categories:
            category_data = results_df[results_df['trend_category'] == category]
            if len(category_data) > 0:
                avg_metrics = category_data[metrics].mean()
                # Normalize to 0-1 scale
                normalized = (avg_metrics - results_df[metrics].min()) / (results_df[metrics].max() - results_df[metrics].min())
                normalized_data.append(normalized)
        
        if not normalized_data:
            return
            
        # Create radar chart
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(projection='polar'))
        
        angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        for i, category in enumerate(categories):
            if i < len(normalized_data):
                values = normalized_data[i].tolist()
                values += values[:1]  # Complete the circle
                ax.plot(angles, values, 'o-', linewidth=2, 
                       label=category, color=self.trend_colors.get(category, '#999999'))
                ax.fill(angles, values, alpha=0.1, color=self.trend_colors.get(category, '#999999'))
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics)
        ax.set_yticklabels([])
        ax.set_title('Average Metrics by Trend Category (Normalized)', fontsize=14, fontweight='bold')
        ax.legend(bbox_to_anchor=(1.1, 1), loc='upper left')
        
        return fig

    def plot_heatmap_correlation(self, results_df, figsize=(10, 8)):
        """Heatmap of correlations between metrics"""
        if results_df.empty:
            print("No data to plot")
            return
            
        # Select numerical columns for correlation
        numerical_cols = ['current_prevalence', 'trend_slope', 'recent_momentum_pct', 
                         'CAGR_pct', 'peak_ratio', 'trend_r2', 'nonlinearity']
        
        corr_matrix = results_df[numerical_cols].corr()
        
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(corr_matrix, cmap='coolwarm', aspect='auto', vmin=-1, vmax=1)
        
        # Add correlation values as text
        for i in range(len(corr_matrix)):
            for j in range(len(corr_matrix)):
                ax.text(j, i, f'{corr_matrix.iloc[i, j]:.2f}', 
                       ha='center', va='center', fontsize=10,
                       color='white' if abs(corr_matrix.iloc[i, j]) > 0.5 else 'black')
        
        ax.set_xticks(range(len(corr_matrix)))
        ax.set_yticks(range(len(corr_matrix)))
        ax.set_xticklabels([col.replace('_', ' ').title() for col in corr_matrix.columns])
        ax.set_yticklabels([col.replace('_', ' ').title() for col in corr_matrix.columns])
        ax.tick_params(axis='x', rotation=45)
        
        plt.colorbar(im, ax=ax, shrink=0.6)
        ax.set_title('Correlation Matrix of Skill Metrics', fontsize=14, fontweight='bold')
        
        plt.tight_layout()
        return fig

    def create_interactive_trend_dashboard(self, results_df, pivot_df):
        """Create an interactive Plotly dashboard"""
        if results_df.empty:
            print("No data to plot")
            return
            
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Trend Categories Distribution', 'Prevalence vs Trend Slope',
                          'Emerging vs Declining Skills', 'Metrics by Category'),
            specs=[[{"type": "pie"}, {"type": "scatter"}],
                   [{"type": "bar"}, {"type": "box"}]]
        )
        
        # Pie chart - Trend categories
        category_counts = results_df['trend_category'].value_counts()
        colors = [self.trend_colors.get(cat, '#999999') for cat in category_counts.index]
        
        fig.add_trace(
            go.Pie(labels=category_counts.index, values=category_counts.values,
                  marker=dict(colors=colors), name="Categories"),
            row=1, col=1
        )
        
        # Scatter plot - Prevalence vs Trend Slope
        for category in results_df['trend_category'].unique():
            mask = results_df['trend_category'] == category
            fig.add_trace(
                go.Scatter(x=results_df.loc[mask, 'current_prevalence'],
                          y=results_df.loc[mask, 'trend_slope'],
                          mode='markers',
                          name=category,
                          marker=dict(color=self.trend_colors.get(category, '#999999'),
                                     size=8, opacity=0.7),
                          text=results_df.loc[mask, 'skill'],
                          hovertemplate='<b>%{text}</b><br>Prevalence: %{x:.1f}%<br>Slope: %{y:.3f}'),
                row=1, col=2
            )
        
        # Bar chart - Top emerging and declining
        emerging = results_df[results_df['is_emerging']].nlargest(10, 'recent_momentum_pct')
        declining = results_df[results_df['is_declining']].nsmallest(10, 'recent_momentum_pct')
        
        if not emerging.empty:
            fig.add_trace(
                go.Bar(x=emerging['recent_momentum_pct'],
                      y=emerging['skill'],
                      orientation='h',
                      name='Emerging',
                      marker_color=self.trend_colors['Emerging'],
                      text=emerging['current_prevalence'].round(1),
                      texttemplate='%{text}%',
                      hovertemplate='<b>%{y}</b><br>Momentum: %{x:.1f}%<br>Prevalence: %{text}%'),
                row=2, col=1
            )
        
        if not declining.empty:
            fig.add_trace(
                go.Bar(x=declining['recent_momentum_pct'],
                      y=declining['skill'],
                      orientation='h',
                      name='Declining',
                      marker_color=self.trend_colors['Declining'],
                      text=declining['current_prevalence'].round(1),
                      texttemplate='%{text}%',
                      hovertemplate='<b>%{y}</b><br>Momentum: %{x:.1f}%<br>Prevalence: %{text}%'),
                row=2, col=1
            )
        
        # Box plot - Momentum by category
        for category in results_df['trend_category'].unique():
            category_data = results_df[results_df['trend_category'] == category]['recent_momentum_pct']
            fig.add_trace(
                go.Box(y=category_data,
                      name=category,
                      marker_color=self.trend_colors.get(category, '#999999'),
                      showlegend=False),
                row=2, col=2
            )
        
        fig.update_layout(height=800, title_text="Skill Trends Analysis Dashboard", 
                         showlegend=True, template="plotly_white")
        
        return fig
    
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
    
    
    def run_full_analysis(self) -> Tuple[Dict, pd.DataFrame]:
        """Execute the complete analysis pipeline"""
        logger.info("🚀 Starting comprehensive skills analysis...")
        
        df = self.load_cleaned_data()
        pivot_df = self.create_skill_pivot(df)
        pivot_df.to_parquet(self.processed_dir / "monthly_skill_pivot.parquet", index=False)
        
        report = self.generate_comprehensive_report(pivot_df)
        
        logger.info("🎉 Analysis completed successfully!")
        return report, pivot_df
    
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
