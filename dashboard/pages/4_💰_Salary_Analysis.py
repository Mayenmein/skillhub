# dashboard/pages/4_💰_Salary_Analysis.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from app import get_dashboard_manager
from utils.dashboard_utils import render_filter_summary

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

    # Story-driven tabs
    tab1, tab2 = st.tabs(["🎯 Your Earning Potential", "📊 Market Context"])

    with tab1:
        render_earning_potential(main_df)

    with tab2:
        render_market_context(main_df)

def render_earning_potential(df):
    """Personalized salary insights for career planning"""
    st.header("🎯 Your Earning Potential")
    
    if 'salary_category' not in df.columns or df['salary_category'].isna().all():
        st.warning("⚠️ Salary data not available")
        return
    
    df['avg_salary_usd'] = df['avg_salary_usd'].replace(0, np.NaN)
    salary_data = df[df['salary_category'] != 'Unknown']
    
    if salary_data.empty:
        st.warning("⚠️ No valid salary data to analyze")
        return

    # Quick reality check
    st.subheader("💰 What Can You Expect to Earn?")
    
    if not salary_data.empty and 'avg_salary_usd' in salary_data.columns:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            median_salary = salary_data['avg_salary_usd'].median()
            st.metric("Typical Salary", f"${median_salary:,.0f}")
            st.caption("What most people actually earn")
        
        with col2:
            avg_salary = salary_data['avg_salary_usd'].mean()
            st.metric("Average Salary", f"${avg_salary:,.0f}")
            st.caption("Including very high/low earners")
        
        with col3:
            salary_range = f"${salary_data['avg_salary_usd'].min():,.0f} - ${salary_data['avg_salary_usd'].max():,.0f}"
            st.metric("Salary Range", salary_range)
            st.caption("From entry to top earners")

    # Role-based earning potential
    st.subheader("💼 Choose Your Path: Role Comparison")
    
    if "standardized_title" in salary_data.columns:
        # Show top paying roles
        role_salaries = salary_data.groupby("standardized_title")["avg_salary_usd"].median().sort_values(ascending=False).head(8)
        
        fig_roles = px.bar(
            x=role_salaries.values,
            y=role_salaries.index,
            orientation='h',
            title="Highest Paying Roles",
            labels={'x': 'Median Salary (USD)', 'y': 'Role'},
            color=role_salaries.values,
            color_continuous_scale='Viridis'
        )
        st.plotly_chart(fig_roles, use_container_width=True)
        
        # Role selection for detailed comparison
        available_roles = role_salaries.index.tolist()
        
        col1, col2 = st.columns(2)
        with col1:
            selected_role = st.selectbox("Explore salary for:", available_roles)
        
        if selected_role:
            role_data = salary_data[salary_data['standardized_title'] == selected_role]
            
            with col2:
                role_median = role_data['avg_salary_usd'].median()
                st.metric(f"Median for {selected_role}", f"${role_median:,.0f}")

    # Career progression insights
    st.subheader("📈 Grow Your Earnings")
    
    if "seniority_level" in salary_data.columns:
        seniority_data = salary_data[salary_data['seniority_level'].notna()]
        if not seniority_data.empty:
            # Show salary progression
            seniority_medians = seniority_data.groupby('seniority_level')['avg_salary_usd'].median()
            
            # Define logical order
            seniority_order = ['Entry-level', 'Junior', 'Mid-level', 'Senior', 'Lead', 'Executive']
            available_levels = [level for level in seniority_order if level in seniority_medians.index]
            
            if len(available_levels) > 1:
                progression_data = pd.DataFrame({
                    'Level': available_levels,
                    'Salary': [seniority_medians[level] for level in available_levels]
                })
                
                fig_progression = px.line(
                    progression_data,
                    x='Level',
                    y='Salary',
                    markers=True,
                    title='How Salary Grows with Experience',
                    labels={'Salary': 'Median Salary (USD)', 'Level': 'Career Level'}
                )
                st.plotly_chart(fig_progression, use_container_width=True)
                
                # Progression insight
                entry_salary = seniority_medians[available_levels[0]]
                senior_salary = seniority_medians[available_levels[-1]]
                growth = senior_salary - entry_salary
                growth_pct = (growth / entry_salary) * 100
                
                st.success(f"**Career growth potential**: From ${entry_salary:,.0f} to ${senior_salary:,.0f} (+{growth_pct:.0f}%)")

    # Work mode impact
    st.subheader("🏠 Work Location & Salary")
    
    if "work_mode" in salary_data.columns:
        work_mode_data = salary_data[salary_data['work_mode'].notna()]
        if not work_mode_data.empty:
            work_mode_medians = work_mode_data.groupby('work_mode')['avg_salary_usd'].median().sort_values(ascending=False)
            
            fig_work = px.bar(
                x=work_mode_medians.values,
                y=work_mode_medians.index,
                orientation='h',
                title='Salary by Work Arrangement',
                labels={'x': 'Median Salary (USD)', 'y': 'Work Mode'},
                color=work_mode_medians.values,
                color_continuous_scale='Blues'
            )
            st.plotly_chart(fig_work, use_container_width=True)
            
            if len(work_mode_medians) > 1:
                best_mode = work_mode_medians.index[0]
                best_salary = work_mode_medians.iloc[0]
                st.info(f"**{best_mode}** roles tend to pay the most (${best_salary:,.0f})")

def render_market_context(df):
    """Broader market trends and data quality"""
    st.header("📊 Salary Market Context")
    
    if 'salary_category' not in df.columns or df['salary_category'].isna().all():
        st.warning("⚠️ Salary data not available")
        return
    
    df['avg_salary_usd'] = df['avg_salary_usd'].replace(0, np.NaN)
    salary_data = df[df['salary_category'] != 'Unknown']
    
    if salary_data.empty:
        st.warning("⚠️ No valid salary data to analyze")
        return

    # Market overview
    st.subheader("📈 Salary Distribution")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Salary ranges
        fig_ranges = px.histogram(
            salary_data, 
            x="salary_category",
            title="Jobs by Salary Range",
            labels={"salary_category": "Salary Range", "count": "Number of Jobs"}
        )
        st.plotly_chart(fig_ranges, use_container_width=True)
    
    with col2:
        # Detailed distribution
        if not salary_data.empty and 'avg_salary_usd' in salary_data.columns:
            fig_detailed = px.histogram(
                salary_data,
                x="avg_salary_usd",
                nbins=20,
                title="Detailed Salary Distribution",
                labels={"avg_salary_usd": "Salary (USD)"},
            )
            # Add median line
            median_val = salary_data['avg_salary_usd'].median()
            fig_detailed.add_vline(x=median_val, line_dash="dash", line_color="red", 
                                 annotation_text=f"Median: ${median_val:,.0f}")
            st.plotly_chart(fig_detailed, use_container_width=True)

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
        if salary_coverage < 50:
            st.warning("Limited salary data - consider this when making decisions")
        else:
            st.success("Good salary data coverage")
        
        st.write("Salaries are estimates based on job postings and may vary by location, company, and experience.")

if __name__ == "__main__":
    main()