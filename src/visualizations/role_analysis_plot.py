# role_analysis_plot.py
import pandas as pd 
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple 
import logging
from datetime import datetime
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)  

class RoleAnalysisPlot: 

    def __init__(self, reports_dir: str="../reports"):
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12)) 
        self.figures_dir = Path(reports_dir) / "figures"
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # AUTO-SAVE: Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.figures_dir / f"skill_progression_organized_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved plot to: {save_path}")
        
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
        
        # AUTO-SAVE: Save plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_path = self.figures_dir / f"skill_progression_simple_{timestamp}.jpg"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved simple progression plot to: {save_path}")
        
        plt.show()