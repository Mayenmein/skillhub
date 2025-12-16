# dashboard/pages/1_📊_Overview.py

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
from app import get_dashboard_manager
from utils.dashboard_utils import render_filter_summary 

def main():
    st.title("📊 Dashboard Overview")
    st.markdown("""
    **Comprehensive overview of the data science job market with key metrics, trends, and insights.**
    Use the filters in the sidebar to customize your view.
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
         
    # Market insights
    st.markdown("---")
    st.subheader("💡 Market Insights")
     
    # Key visualizations in columns
    st.markdown("---")
    st.subheader("📈 Key Market Visualizations")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Top companies
        if 'company' in main_df.columns:
            top_companies = main_df['company'].value_counts().head(8)
            fig = px.bar(
                x=top_companies.values,
                y=top_companies.index,
                orientation='h',
                title='Top Companies by Job Count',
                labels={'x': 'Number of Jobs', 'y': 'Company'},
                color=top_companies.values,
                color_continuous_scale='Blues'
            )
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Role distribution
        if 'cleaned_title_category' in main_df.columns:
            role_dist = main_df['cleaned_title_category'].value_counts().head(8)
            fig = px.pie(
                values=role_dist.values,
                names=role_dist.index,
                title='Top Role Categories',
                hole=0.4
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Geographic and seniority distribution
    col3, col4 = st.columns(2)
    
    with col3:
        # Country distribution
        if 'country' in main_df.columns:
            country_dist = main_df['country'].value_counts().head(10)
            fig = px.bar(
                x=country_dist.values,
                y=country_dist.index,
                orientation='h',
                title='Top Countries by Job Count',
                labels={'x': 'Number of Jobs', 'y': 'Country'},
                color=country_dist.values,
                color_continuous_scale='Greens'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col4:
        # Seniority distribution
        if 'seniority_level' in main_df.columns:
            seniority_dist = main_df['seniority_level'].value_counts()
            fig = px.bar(
                x=seniority_dist.index,
                y=seniority_dist.values,
                title='Seniority Level Distribution',
                labels={'x': 'Seniority Level', 'y': 'Count'},
                color=seniority_dist.values,
                color_continuous_scale='Viridis'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Skills overview
    st.markdown("---")
    st.subheader("🛠️ Skills Overview")
    
    if not filtered_df.empty:
        # Aggregate skills data using DataProcessor
        skill_df = dashboard_manager.data_processor.aggregate_pivot(filtered_df, metric='prevalence')
        skill_df['prevalence'] = skill_df['prevalence'].round(1)
        category_df = dashboard_manager.data_processor.aggregate_pivot(filtered_df, column='skill_category')
        
        col5, col6 = st.columns(2)
        
        with col5:
            # Top 10 Skills
            top_skills = skill_df.head(10)
            fig = px.bar(
                top_skills,
                x='prevalence',
                y='skill',
                orientation='h',
                title='Top 10 Skills by Prevalence',
                labels={'prevalence': 'Prevalence (% of Jobs)', 'skill': 'Skill'},
                color='prevalence',
                color_continuous_scale='Blues'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
        
        with col6:
            # Skill Categories
            top_categories = category_df.head(8)
            fig = px.bar(
                top_categories,
                x='prevalence',
                y='skill_category',
                orientation='h',
                title='Top Skill Categories by Prevalence',
                labels={'prevalence': 'Prevalence (% of Jobs)', 'skill_category': 'Category'},
                color='prevalence',
                color_continuous_scale='Viridis'
            )
            fig.update_layout(yaxis={'categoryorder': 'total ascending'})
            st.plotly_chart(fig, use_container_width=True)
    
    # Temporal trends
    st.markdown("---")
    st.subheader("📅 Market Trends Over Time")
    
    if 'date' in main_df.columns and not main_df.empty:
        try:
            # Convert date and aggregate by month
            main_df['date_dt'] = pd.to_datetime(main_df['date'], format='%Y.0_%m.0')
            monthly_trends = main_df.groupby(main_df['date_dt'].dt.to_period('M')).size().reset_index()
            monthly_trends['date_dt'] = monthly_trends['date_dt'].dt.to_timestamp()
            monthly_trends.columns = ['date', 'job_count']
            
            fig = px.line(
                monthly_trends,
                x='date',
                y='job_count',
                title='Monthly Job Postings Trend',
                labels={'date': 'Month', 'job_count': 'Number of Jobs'},
                markers=True
            )
            fig.update_layout(
                xaxis=dict(tickformat="%b %Y"),
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as e:
            st.info("Could not generate temporal trends with current date format")
    
    # Market health indicators
    st.markdown("---")
    st.subheader("📊 Market Health Indicators")
    
    col7, col8, col9, col10 = st.columns(4)
    
    with col7:
        if not filtered_df.empty:
            skills_per_job = dashboard_manager.df['skills_count'].mean()
            st.metric("Avg Skills per Job", f"{skills_per_job:.1f}")
    
    with col8:
        if 'company' in main_df.columns:
            jobs_per_company = len(main_df) / main_df['company'].nunique()
            st.metric("Avg Jobs per Company", f"{jobs_per_company:.1f}")
    
    with col9:
        if 'cleaned_title_category' in main_df.columns:
            role_diversity = main_df['cleaned_title_category'].nunique()
            st.metric("Role Categories", f"{role_diversity}")
    
    with col10:
        if 'date' in main_df.columns:
            try:
                date_range = pd.to_datetime(main_df['date'], format='%Y.0_%m.0')
                date_span = (date_range.max() - date_range.min()).days
                st.metric("Data Time Span", f"{date_span} days")
            except:
                st.metric("Data Time Span", "N/A")
    
    # Data quality indicators
    st.markdown("---")
    st.subheader("🔍 Data Quality Summary")
    
    col11, col12, col13, col14 = st.columns(4)
    
    with col11:
        completeness = (1 - main_df.isnull().sum().sum() / (main_df.shape[0] * main_df.shape[1])) * 100
        st.metric("Data Completeness", f"{completeness:.1f}%")
    
    with col12:
        duplicate_rate = (main_df.duplicated().sum() / len(main_df)) * 100
        st.metric("Duplicate Rate", f"{duplicate_rate:.1f}%")
    
    with col13:
        if not filtered_df.empty:
            skills_coverage = (filtered_df['skill'].notna().sum() / len(filtered_df)) * 100
            st.metric("Skills Coverage", f"{skills_coverage:.1f}%")
    
    with col14:
        if 'avg_salary_usd' in main_df.columns:
            salary_completeness = (main_df['avg_salary_usd'].notna().sum() / len(main_df)) * 100
            st.metric("Salary Data", f"{salary_completeness:.1f}%")
        else:
            st.metric("Salary Data", "0%")
    
    # Footer with data freshness
    st.markdown("---")
    st.caption(f"Data last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} | Total records: {len(main_df):,}")

if __name__ == "__main__":
    main()