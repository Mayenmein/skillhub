"""Skill trend analysis over time"""
import pandas as pd
import numpy as np
from src.core.base_analyzer import BaseAnalyzer
from src.utils.calculations import (
    calculate_smoothed_growth, calculate_rolling_metrics, 
    batch_ols, calculate_nonlinearity, classify_trends_smart_recent, 
    map_categories_to_strings
)

class TrendAnalyzer(BaseAnalyzer):
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
    
    def _generate_trend_insights(self, results_df: pd.DataFrame) -> dict:
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