# dashboard/utils/dashboard_utils.py

import streamlit as st
import pandas as pd
from typing import Dict

def render_filter_summary(filters: Dict, filtered_df: pd.DataFrame, total_df: pd.DataFrame):
    """Show current filter summary and active filters"""
    total = len(total_df)
    shown = len(filtered_df)

    # Build a readable list of active filters
    active_parts = []
    if filters:
        # Date range
        date_range = filters.get('date_range')
        if date_range and isinstance(date_range, (list, tuple)) and len(date_range) == 2 and date_range[0] and date_range[1]:
            try:
                start = pd.to_datetime(date_range[0]).strftime('%Y-%m-%d')
                end = pd.to_datetime(date_range[1]).strftime('%Y-%m-%d')
                active_parts.append(f"Date: {start} → {end}")
            except Exception:
                pass

        # Common categorical filters
        for key in ['country', 'company_name', 'primary_job_type', 'standardized_title', 'seniority_level']:
            val = filters.get(key)
            if val and val != 'All':
                pretty = key.replace('_', ' ').title()
                active_parts.append(f"{pretty}: {val}")

        # Skills (if present)
        skills = filters.get('skills')
        if skills:
            try:
                if isinstance(skills, (list, tuple)):
                    skills_str = ', '.join(map(str, skills))
                else:
                    skills_str = str(skills)
                active_parts.append(f"Skills: {skills_str}")
            except Exception:
                active_parts.append(f"Skills: {skills}")

    # Display summary
    if shown < total:
        reduction_pct = ((total - shown) / total) * 100 if total > 0 else 0
        if active_parts:
            st.success(
                f"""
                **Filters Applied:** Showing {shown:,} of {total:,} jobs ({reduction_pct:.1f}% reduction)

                **Active Filters:** {', '.join(active_parts)}
                """
            )
        else:
            st.success(f"**Filters Applied:** Showing {shown:,} of {total:,} jobs ({reduction_pct:.1f}% reduction)")
    else:
        if active_parts:
            st.info(f"**Active Filters (no row reduction):** {', '.join(active_parts)}")
        else:
            st.info("**No filters applied:** Showing all available jobs")

def render_metrics(df: pd.DataFrame):
    """Render metrics cards"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Total Jobs", f"{len(df):,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        companies = df['company_name'].nunique()
        st.metric("Companies", f"{companies:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        avg_skills = df['skills_count'].mean() if 'skills_count' in df.columns else 0
        st.metric("Avg Skills/Job", f"{avg_skills:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if 'avg_salary_usd' in df.columns:
            avg_salary = df['avg_salary_usd'].mean()
            st.metric(f"Avg Salary (From {df['avg_salary_usd'].notna().sum()} jobs)", 
                     f"${avg_salary:,.0f} " if not pd.isna(avg_salary) else "N/A")
        else:
            st.metric("Data Freshness", "Live")
        st.markdown('</div>', unsafe_allow_html=True)