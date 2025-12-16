# market_analysis_plot.py
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple 
import numpy as np
import logging
from pathlib import Path
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  

class MarketAnalysisPlot: 
    
    def __init__(self, reports_dir: str="../reports"):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12)) 
        self.figures_dir = Path(reports_dir) / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # AUTO-SAVE: Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        column_name = 'skills' if column == 'Skill' else 'skill_categories'
        save_path = self.figures_dir / f"{column_name}_bar_{metric}_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved bar plot to: {save_path}")
        
        plt.show()
    
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
        
        # AUTO-SAVE: Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.figures_dir / f"seniority_patterns_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved seniority patterns plot to: {save_path}")
        
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