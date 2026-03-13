# dashboard/pages/4_💰_Salary_Analysis.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from app import get_dashboard_manager
from utils.dashboard_utils import render_filter_summary
from src.analysis.salary_analyzer import SalarySkillRegressionAnalyzer

def main():
    st.title("💰 Salary Insights")
    st.markdown("""
    **Understand your earning potential and make informed career decisions.**
    """)
    
    # Get dashboard manager
    dashboard_manager = get_dashboard_manager()
    
    # Setup sidebar filters
    filters = dashboard_manager.setup_sidebar_filters()
    
    # Apply filters
    main_df, filtered_df = dashboard_manager.apply_filters(filters)
    
    # Show filter summary
    render_filter_summary(filters, main_df, dashboard_manager.df)
    
    if main_df.empty:
        st.warning("No data available with current filters")
        return
    
    # Calculate salary as average of min/max
    if 'salary_min' in main_df.columns and 'salary_max' in main_df.columns:
        main_df['salary'] = (main_df['salary_min'] + main_df['salary_max']) / 2
        # Filter out unrealistic salaries
        main_df = main_df[(main_df['salary'] > 1000) & (main_df['salary'] < 500000)].copy()
        main_df['job_id'] = main_df.index.astype(str)

    # Story-driven tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Your Earning Potential", "📊 Market Context", "🧠 Skill-Salary Insights"])

    with tab1:
        render_earning_potential(main_df)

    with tab2:
        render_market_context(main_df)
    
    with tab3:
        render_skill_salary_insights(main_df, dashboard_manager)

def render_earning_potential(df):
    """Personalized salary insights for career planning"""
    st.header("🎯 Your Earning Potential")
    
    if 'salary' not in df.columns or df['salary'].isna().all():
        st.warning("⚠️ Salary data not available")
        return
    
    salary_data = df[df['salary'].notna()].copy()
    
    if salary_data.empty:
        st.warning("⚠️ No valid salary data to analyze")
        return

    # Quick reality check
    st.subheader("💰 What Can You Expect to Earn?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        median_salary = salary_data['salary'].median()
        st.metric("Typical Salary", f"${median_salary:,.0f}")
        st.caption("What most people actually earn")
    
    with col2:
        avg_salary = salary_data['salary'].mean()
        st.metric("Average Salary", f"${avg_salary:,.0f}")
        st.caption("Including very high/low earners")
    
    with col3:
        salary_range = f"${salary_data['salary'].min():,.0f} - ${salary_data['salary'].max():,.0f}"
        st.metric("Salary Range", salary_range)
        st.caption("From entry to top earners")

    # Role-based earning potential
    if "standardized_title" in salary_data.columns:
        st.subheader("💼 Choose Your Path: Role Comparison")
        
        # Show top paying roles (with sufficient samples)
        role_stats = salary_data.groupby("standardized_title").agg({
            "salary": ["median", "count", "std"]
        }).round(0)
        role_stats.columns = ['median_salary', 'count', 'std']
        role_stats = role_stats.reset_index()
        
        # Filter for roles with at least 3 samples
        role_stats = role_stats[role_stats['count'] >= 3]
        
        if not role_stats.empty:
            top_roles = role_stats.nlargest(8, 'median_salary')
            
            fig_roles = px.bar(
                top_roles,
                x='median_salary',
                y='standardized_title',
                orientation='h',
                title="Highest Paying Roles (with sufficient data)",
                labels={'median_salary': 'Median Salary (USD)', 'standardized_title': 'Role'},
                color='median_salary',
                color_continuous_scale='Viridis',
                error_x='std'  # Show variability
            )
            st.plotly_chart(fig_roles, use_container_width=True)
            
            # Role selection for detailed comparison
            available_roles = role_stats['standardized_title'].tolist()
            
            col1, col2 = st.columns(2)
            with col1:
                selected_role = st.selectbox("Explore salary for:", available_roles)
            
            if selected_role:
                role_data = salary_data[salary_data['standardized_title'] == selected_role]
                
                with col2:
                    role_median = role_data['salary'].median()
                    role_count = len(role_data)
                    st.metric(f"Median for {selected_role}", f"${role_median:,.0f}")
                    st.caption(f"Based on {role_count} jobs")

    # Career progression insights
    if "seniority_level" in salary_data.columns:
        st.subheader("📈 Grow Your Earnings")
        
        seniority_data = salary_data[salary_data['seniority_level'].notna()]
        if not seniority_data.empty and len(seniority_data) >= 10:
            # Show salary progression
            seniority_stats = seniority_data.groupby('seniority_level')['salary'].agg(['median', 'count'])
            seniority_stats = seniority_stats[seniority_stats['count'] >= 3]  # Min samples
            
            if not seniority_stats.empty:
                # Define logical order
                seniority_order = ['Entry-level', 'Junior', 'Mid-level', 'Senior', 'Lead', 'Executive']
                available_levels = [level for level in seniority_order if level in seniority_stats.index]
                
                if len(available_levels) > 1:
                    progression_data = pd.DataFrame({
                        'Level': available_levels,
                        'Salary': [seniority_stats.loc[level, 'median'] for level in available_levels],
                        'Jobs': [seniority_stats.loc[level, 'count'] for level in available_levels]
                    })
                    
                    fig_progression = px.line(
                        progression_data,
                        x='Level',
                        y='Salary',
                        markers=True,
                        title='How Salary Grows with Experience',
                        labels={'Salary': 'Median Salary (USD)', 'Level': 'Career Level'},
                        text='Jobs'  # Show sample size
                    )
                    fig_progression.update_traces(textposition="top center")
                    st.plotly_chart(fig_progression, use_container_width=True)
                    
                    # Progression insight
                    entry_salary = seniority_stats.loc[available_levels[0], 'median']
                    senior_salary = seniority_stats.loc[available_levels[-1], 'median']
                    growth = senior_salary - entry_salary
                    growth_pct = (growth / entry_salary) * 100
                    
                    st.success(f"**Career growth potential**: From ${entry_salary:,.0f} to ${senior_salary:,.0f} (+{growth_pct:.0f}%)")

    # Work mode impact
    if "work_mode" in salary_data.columns:
        st.subheader("🏠 Work Location & Salary")
        
        work_mode_data = salary_data[salary_data['work_mode'].notna()]
        if not work_mode_data.empty and len(work_mode_data) >= 10:
            work_mode_stats = work_mode_data.groupby('work_mode')['salary'].agg(['median', 'count'])
            work_mode_stats = work_mode_stats[work_mode_stats['count'] >= 3]
            
            if not work_mode_stats.empty:
                work_mode_stats = work_mode_stats.sort_values('median', ascending=False)
                
                fig_work = px.bar(
                    x=work_mode_stats['median'].values,
                    y=work_mode_stats.index,
                    orientation='h',
                    title='Salary by Work Arrangement',
                    labels={'x': 'Median Salary (USD)', 'y': 'Work Mode'},
                    color=work_mode_stats['median'].values,
                    color_continuous_scale='Blues',
                    text=work_mode_stats['count'].values
                )
                fig_work.update_traces(texttemplate='%{text} jobs', textposition='outside')
                st.plotly_chart(fig_work, use_container_width=True)
                
                if len(work_mode_stats) > 1:
                    best_mode = work_mode_stats.index[0]
                    best_salary = work_mode_stats.iloc[0]['median']
                    st.info(f"**{best_mode}** roles tend to pay the most (${best_salary:,.0f})")

def render_market_context(df):
    """Broader market trends and data quality"""
    st.header("📊 Salary Market Context")
    
    if 'salary' not in df.columns or df['salary'].isna().all():
        st.warning("⚠️ Salary data not available")
        return
    
    salary_data = df[df['salary'] > 0].copy()
    
    if salary_data.empty:
        st.warning("⚠️ No valid salary data to analyze")
        return

    # Market overview
    st.subheader("📈 Salary Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create salary categories if not present
        if 'salary_category' not in salary_data.columns:
            bins = [0, 30000, 50000, 70000, 90000, 120000, 150000, 200000, float('inf')]
            labels = ['<30k', '30-50k', '50-70k', '70-90k', '90-120k', '120-150k', '150-200k', '200k+']
            salary_data['salary_category'] = pd.cut(salary_data['salary'], bins=bins, labels=labels)
        
        category_counts = salary_data['salary_category'].value_counts().sort_index()
        
        fig_ranges = px.bar(
            x=category_counts.index,
            y=category_counts.values,
            title="Jobs by Salary Range",
            labels={"x": "Salary Range", "y": "Number of Jobs"}
        )
        st.plotly_chart(fig_ranges, use_container_width=True)
    
    with col2:
        # Detailed distribution with box plot
        fig_detailed = px.box(
            salary_data,
            y="salary",
            title="Salary Distribution Overview",
            labels={"salary": "Salary (USD)"},
            points="all"  # Show all points for transparency
        )
        # Add mean marker
        mean_val = salary_data['salary'].mean()
        fig_detailed.add_hline(y=mean_val, line_dash="dash", line_color="red", 
                             annotation_text=f"Mean: ${mean_val:,.0f}")
        st.plotly_chart(fig_detailed, use_container_width=True)

    # Geographic insights
    if 'country' in salary_data.columns:
        st.subheader("🌍 Geographic Salary Variations")
        
        country_stats = salary_data.groupby('country')['salary'].agg(['median', 'count', 'mean'])
        country_stats = country_stats[country_stats['count'] >= 5].sort_values('median', ascending=False).head(10)
        
        if not country_stats.empty:
            fig_country = px.bar(
                country_stats.reset_index(),
                x='country',
                y='median',
                title="Top 10 Countries by Median Salary",
                labels={'country': 'Country', 'median': 'Median Salary (USD)'},
                color='median',
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig_country, use_container_width=True)

    # Data quality and limitations
    st.subheader("🔍 Understanding the Data")
    
    salary_jobs = len(salary_data)
    total_jobs = len(df)
    salary_coverage = (salary_jobs / total_jobs) * 100
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.metric("Jobs with Salary Data", f"{salary_jobs:,}")
        st.metric("Data Coverage", f"{salary_coverage:.1f}%")
    
    with col2:
        st.write("**What this means:**")
        if salary_coverage < 30:
            st.warning("⚠️ Limited salary data - consider these insights as directional")
        elif salary_coverage < 50:
            st.info("📊 Moderate salary data coverage - insights are reasonably reliable")
        else:
            st.success("✅ Good salary data coverage - insights are reliable")
        
        st.caption("Salaries are estimated based on job postings and may vary by location, company, and experience.")

def render_skill_salary_insights(df, dashboard_manager):
    """Skill-based salary insights using regression analysis qualitatively"""
    st.header("🧠 Skills That Drive Higher Salaries")
    
    if 'salary' not in df.columns or df['salary'].isna().all():
        st.warning("⚠️ Salary data not available for skill analysis")
        return
    
    # Check if we have skill data
    if not hasattr(dashboard_manager, 'skill_pivot_df') or dashboard_manager.skill_pivot_df is None:
        st.info("📊 Building skill matrix for analysis...")
        # Try to get or create skill pivot
        try:
            from src.features.skill_pivot_builder import create_skill_pivot
            dashboard_manager.skill_pivot_df = create_skill_pivot(df, job_id_column='job_id', skill_column='skills')
        except:
            st.warning("⚠️ Unable to build skill matrix")
            return
    
    salary_data = df[df['salary'].notna()].copy()
    
    if len(salary_data) < 20:  # Need minimum samples
        st.warning("⚠️ Insufficient data for skill-salary analysis (need at least 20 jobs with salary)")
        return
    
    st.markdown("""
    This analysis identifies which skills and skill combinations are associated with higher salaries.
    While individual skill impact varies, certain patterns emerge from the data.
    """)
    
    try:
        # Initialize analyzer
        salary_analyzer = SalarySkillRegressionAnalyzer(
            salary_column='salary',
            location_column='country' if 'country' in salary_data.columns else None,
            min_samples_per_skill=3  # Lower threshold for insights
        )
        
        # Run analysis
        with st.spinner("Analyzing skill-salary relationships..."):
            results = salary_analyzer.analyze_from_pivot(
                original_df=salary_data,
                skill_pivot_df=dashboard_manager.skill_pivot_df,
                adjust_for_location='country' in salary_data.columns,
                include_interactions=True
            )
        
        # Model performance (with context about low R²)
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.metric("Model R²", f"{results['model_performance'].get('r2', 0):.3f}")
            st.caption("Explains {:.1f}% of salary variation".format(results['model_performance'].get('r2', 0) * 100))
        
        with col2:
            st.info("""
            **Note on interpretation**: The model explains a portion of salary variation, but many factors 
            (company, specific role, negotiation, etc.) aren't captured. Use these insights as directional 
            guidance, not precise predictions.
            """)
        
        # Top skill premiums
        if 'skill_premiums' in results and not results['skill_premiums'].empty:
            st.subheader("💰 Skills Associated with Higher Salaries")
            
            premiums = results['skill_premiums'].copy()
            premiums = premiums[premiums['skill'].notna()].head(12)
            
            # Create horizontal bar chart
            fig_premiums = px.bar(
                premiums,
                x='coefficient',
                y='skill',
                orientation='h',
                title="Estimated Salary Impact by Skill",
                labels={'coefficient': 'Salary Impact (USD)', 'skill': 'Skill'},
                color='coefficient',
                color_continuous_scale='RdYlGn',
                text=premiums['coefficient'].round(0).astype(int).astype(str) + '$'
            )
            fig_premiums.update_traces(textposition='outside')
            st.plotly_chart(fig_premiums, use_container_width=True)
            
            # Add context about confidence
            st.caption("""
            *Values represent estimated salary difference when skill is mentioned. 
            Higher values suggest stronger association with higher salaries, but individual results vary.*
            """)
        
        # Skill combinations with synergy
        if 'interaction_effects' in results and not results['interaction_effects'].empty:
            st.subheader("🔗 Powerful Skill Combinations")
            
            interactions = results['interaction_effects'].copy()
            interactions = interactions[interactions['skill1'].notna() & interactions['skill2'].notna()]
            
            if not interactions.empty:
                # Show top synergistic combinations
                top_synergies = interactions.nlargest(6, 'synergy')
                
                # Create combination labels
                top_synergies['combination'] = top_synergies['skill1'] + ' + ' + top_synergies['skill2']
                
                fig_synergy = px.bar(
                    top_synergies,
                    x='synergy',
                    y='combination',
                    orientation='h',
                    title="Skill Combinations with Highest Synergy",
                    labels={'synergy': 'Synergy Effect (USD)', 'combination': 'Skill Combination'},
                    color='synergy',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig_synergy, use_container_width=True)
        
        # Recommendations
        if 'recommendations' in results:
            st.subheader("💡 Skill Development Recommendations")
            
            recs = results['recommendations']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if 'skill_development' in recs and recs['skill_development']:
                    st.markdown("**Priority Skills to Develop**")
                    for i, rec in enumerate(recs['skill_development'][:4]):
                        priority_icon = "🔴" if rec.get('priority') == 'high' else "🟡" if rec.get('priority') == 'medium' else "🟢"
                        st.markdown(f"{priority_icon} **{rec['skill']}**")
                        st.caption(f"*{rec.get('reason', '')}*")
            
            with col2:
                if 'skill_combinations' in recs and recs['skill_combinations']:
                    st.markdown("**Valuable Skill Combinations**")
                    for i, rec in enumerate(recs['skill_combinations'][:4]):
                        synergy_icon = "🚀" if rec.get('synergy') == 'high' else "📈"
                        st.markdown(f"{synergy_icon} **{rec.get('combination', '')}**")
                        st.caption(f"*{rec.get('reason', '')}*")
        
        # Alternative simple analysis if model fails
        if results['model_performance'].get('r2', 0) < 0.05:
            with st.expander("📊 View Simple Skill Frequency Analysis"):
                st.markdown("""
                **Alternative perspective**: Looking at skill frequency in high vs low salary jobs
                """)
                
                # Simple percentile-based analysis
                high_salary_threshold = salary_data['salary'].quantile(0.75)
                low_salary_threshold = salary_data['salary'].quantile(0.25)
                
                high_salary_jobs = salary_data[salary_data['salary'] >= high_salary_threshold]
                low_salary_jobs = salary_data[salary_data['salary'] <= low_salary_threshold]
                
                if not high_salary_jobs.empty and not low_salary_jobs.empty:
                    # Get skill frequencies
                    high_skills = []
                    low_skills = []
                    
                    for _, row in high_salary_jobs.iterrows():
                        if 'skills' in row and isinstance(row['skills'], list):
                            high_skills.extend(row['skills'])
                    
                    for _, row in low_salary_jobs.iterrows():
                        if 'skills' in row and isinstance(row['skills'], list):
                            low_skills.extend(row['skills'])
                    
                    if high_skills and low_skills:
                        from collections import Counter
                        high_skill_counts = Counter(high_skills)
                        low_skill_counts = Counter(low_skills)
                        
                        # Calculate ratio
                        all_skills = set(high_skill_counts.keys()) | set(low_skill_counts.keys())
                        skill_ratios = []
                        
                        for skill in all_skills:
                            high_count = high_skill_counts.get(skill, 0)
                            low_count = low_skill_counts.get(skill, 0)
                            if high_count + low_count >= 5:  # Min occurrences
                                ratio = (high_count + 1) / (low_count + 1)  # Add smoothing
                                skill_ratios.append({
                                    'skill': skill,
                                    'high_salary_count': high_count,
                                    'low_salary_count': low_count,
                                    'ratio': ratio
                                })
                        
                        if skill_ratios:
                            ratio_df = pd.DataFrame(skill_ratios)
                            ratio_df = ratio_df.sort_values('ratio', ascending=False).head(10)
                            
                            fig_ratio = px.bar(
                                ratio_df,
                                x='ratio',
                                y='skill',
                                orientation='h',
                                title="Skills More Common in High-Paying Jobs",
                                labels={'ratio': 'High/Low Salary Ratio', 'skill': 'Skill'},
                                color='ratio',
                                color_continuous_scale='RdYlGn'
                            )
                            st.plotly_chart(fig_ratio, use_container_width=True)
                            
                            st.caption("""
                            *Ratio > 1 means skill appears more frequently in high-salary jobs.
                            Ratio < 1 means it's more common in lower-salary jobs.*
                            """)
    
    except Exception as e:
        st.error(f"Error in skill-salary analysis: {str(e)}")
        st.info("Showing basic salary analysis instead")
        
        # Fallback to simple analysis
        if 'skills' in df.columns:
            st.subheader("📊 Skills in High-Paying Jobs")
            
            # Simple approach: show skills from top 25% of jobs
            high_salary_threshold = salary_data['salary'].quantile(0.75)
            top_jobs = salary_data[salary_data['salary'] >= high_salary_threshold]
            
            all_skills = []
            for _, row in top_jobs.iterrows():
                if isinstance(row.get('skills'), list):
                    all_skills.extend(row['skills'])
            
            if all_skills:
                from collections import Counter
                skill_counts = Counter(all_skills)
                skill_df = pd.DataFrame(skill_counts.most_common(15), columns=['Skill', 'Count'])
                
                fig = px.bar(
                    skill_df,
                    x='Count',
                    y='Skill',
                    orientation='h',
                    title="Most Common Skills in Top 25% of Jobs",
                    color='Count',
                    color_continuous_scale='Viridis'
                )
                st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()