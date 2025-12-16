# trends_plot.py
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple 
import numpy as np
from pathlib import Path
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  

TREND_COLORS = {
    'Emerging': '#2E8B57',
    'Growing': '#1E90FF', 
    'Declining': '#DC143C',
    'Rapidly Declining': '#8B0000',
    'Accelerating': '#32CD32',
    'Peaking': '#FF8C00',
    'Reviving': '#9370DB',
    'Stabilizing': '#696969',
    'Stable': '#A9A9A9'
}

class TrendsAnalysisPlot: 

    def __init__(self, reports_dir: str="../reports"):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12)) 
        self.figures_dir = Path(reports_dir) / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.trend_colors = TREND_COLORS
        
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
        
        # AUTO-SAVE: Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.figures_dir / f"trend_categories_distribution_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved trend categories plot to: {save_path}")
        
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
        
        # AUTO-SAVE: Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.figures_dir / f"trend_scatter_matrix_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved scatter matrix plot to: {save_path}")
        
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
        
        # AUTO-SAVE: Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        skill_str = "_".join(skill_names[:3]).replace(" ", "_")
        save_path = self.figures_dir / f"skill_trend_timeline_{skill_str}_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved timeline plot to: {save_path}")
        
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
        
        # AUTO-SAVE: Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.figures_dir / f"emerging_vs_declining_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved emerging vs declining plot to: {save_path}")
        
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
        
        # AUTO-SAVE: Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.figures_dir / f"trend_radar_chart_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved radar chart to: {save_path}")
        
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
        
        # AUTO-SAVE: Save the figure
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.figures_dir / f"correlation_heatmap_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved correlation heatmap to: {save_path}")
        
        return fig