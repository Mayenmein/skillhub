# dashboard/app.py

import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from pathlib import Path 
import sys
import networkx as nx
from typing import Dict, List
from datetime import datetime

from functools import lru_cache

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import refactored analysis modules
from src.main import DataScienceJobsAnalyzer
from src.core.data_processor import DataProcessor
from src.analysis.skill_analyzer import SkillAnalyzer
from src.analysis.trend_analyzer import TrendAnalyzer
from src.analysis.seniority_analyzer import SeniorityAnalyzer
from src.analysis.role_analyzer import RoleAnalyzer
from src.analysis.ecosystem_analyzer import EcosystemAnalyzer

from src.utils.calculations import *

st.set_option("client.showErrorDetails", True)

# Configure the page
st.set_page_config(
    page_title="Data Science Jobs Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .section-header {
        border-bottom: 2px solid #1f77b4;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

class DashboardManager:
    """
    Central dashboard manager for data loading and filter management
    Updated for refactored modular code
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the dashboard manager with refactored components"""
        self.data_dir = data_dir
        
        # Initialize individual analyzers for specific functionality
        self.data_processor = DataProcessor(data_dir)
        self.skill_analyzer = SkillAnalyzer(data_dir)
        self.trend_analyzer = TrendAnalyzer(data_dir)
        self.seniority_analyzer = SeniorityAnalyzer(data_dir)
        self.role_analyzer = RoleAnalyzer(data_dir)
        self.ecosystem_analyzer = EcosystemAnalyzer(data_dir)
        
        # For full analysis pipeline
        self.full_analyzer = DataScienceJobsAnalyzer(data_dir)
        
        # Load data
        SKILLS_DATA_LOCATION = Path(f'{data_dir}/processed/skill_pivot.parquet')
        DATA_INTERIM_PATH = Path(f'{data_dir}/interim/cleaned_jobs_data.csv')

        try:
            self.df = pd.read_csv(DATA_INTERIM_PATH)
            self.skills_df = pd.read_parquet(SKILLS_DATA_LOCATION)
        except FileNotFoundError as e:
            st.error(f"Data files not found: {e}")
            # Create empty dataframes to prevent crashes
            self.df = pd.DataFrame()
            self.skills_df = pd.DataFrame()
    
    def load_and_process_data(self):
        """Load and process data if not already available"""
        if self.df.empty:
            try:
                self.df = self.data_processor.load_cleaned_data()
                self.skills_df = self.data_processor.create_skill_pivot(self.df)
            except Exception as e:
                st.error(f"Error loading data: {e}")
             
    def setup_sidebar_filters(self):
        """Create enhanced sidebar with additional filters and info"""
        with st.sidebar:
            st.title("🌐 Dashboard Controls")
            st.markdown("---")
            
            # Data info card
            st.subheader("📁 Dataset Info")
            st.info(f"""
            **Total Jobs:** {len(self.df):,}
            **Companies:** {self.df['company_name'].nunique():,}  
            **Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
            """)
            
            st.markdown("---")
            
            # Enhanced filters
            filters = {}
            
            # Start with base dataset for filtering
            filtered_df = self.df.copy()
            
            # Date Range Filter
            st.subheader("📅 Date Range Filter")
            if 'published' in filtered_df.columns and not filtered_df.empty:
                try:
                    min_date = pd.to_datetime(filtered_df['published'], format='%Y.0_%m.0').min()
                    max_date = pd.to_datetime(filtered_df['published'], format='%Y.0_%m.0').max()
                    
                    date_range = st.date_input(
                        "Select Date Range",
                        [min_date, max_date],
                        min_value=min_date,
                        max_value=max_date,
                        help="Filter jobs by date range"
                    )
                    filters['date_range'] = date_range
                    
                    # Apply date filter to get available options for subsequent filters
                    if len(date_range) == 2 and date_range[0] and date_range[1]:
                        start_date, end_date = date_range
                        filtered_df = filtered_df[
                            (pd.to_datetime(filtered_df['published'], format='%Y.0_%m.0') >= pd.to_datetime(start_date)) & 
                            (pd.to_datetime(filtered_df['published'], format='%Y.0_%m.0') <= pd.to_datetime(end_date))
                        ]
                except Exception as e:
                    st.warning(f"Date filtering may not work properly: {e}")
            
            st.markdown("---")
            
            # Country Filter
            if 'country' in filtered_df.columns and not filtered_df.empty:
                st.subheader("🌍 Country Filter")
                country_counts = filtered_df['country'].value_counts()
                valid_countries = country_counts[country_counts > 10]
                if len(valid_countries) > 0:
                    countries = ['All'] + valid_countries.index.tolist()
                    selected_country = st.selectbox(
                        "Select Country",
                        countries,
                        index=0,
                        help="Filter jobs by country (only showing countries with 10+ jobs)"
                    )
                    filters['country'] = selected_country
                    
                    if selected_country != 'All':
                        filtered_df = filtered_df.query(f"country == '{selected_country}'")
                else:
                    st.info("No countries with 10+ jobs in current selection")
                    filters['country'] = 'All'
            
            # Company Filter
            if 'company_name' in filtered_df.columns and not filtered_df.empty:
                st.subheader("🏢 Company Filter")
                company_counts = filtered_df['company_name'].value_counts()
                valid_companies = company_counts[company_counts > 10]
                if len(valid_companies) > 0:
                    companies = ['All'] + valid_companies.index.tolist()
                    selected_company = st.selectbox(
                        "Select Company",
                        companies,
                        index=0,
                        help="Filter jobs by company (only showing companies with 10+ jobs)"
                    )
                    filters['company_name'] = selected_company
                    
                    if selected_company != 'All':
                        filtered_df = filtered_df.query(f"company == '{selected_company}'")
                else:
                    st.info("No companies with 10+ jobs in current selection")
                    filters['company_name'] = 'All'
            
            # Primary Job Type Filter
            if 'primary_job_type' in filtered_df.columns and not filtered_df.empty:
                st.subheader("💼 Job Type Filter")
                job_type_counts = filtered_df['primary_job_type'].value_counts()
                valid_job_types = job_type_counts[job_type_counts > 10]
                if len(valid_job_types) > 0:
                    job_types = ['All'] + valid_job_types.index.tolist()
                    selected_job_type = st.selectbox(
                        "Select Job Type",
                        job_types,
                        index=0,
                        help="Filter jobs by primary job type (only showing types with 10+ jobs)"
                    )
                    filters['primary_job_type'] = selected_job_type
                    
                    if selected_job_type != 'All':
                        filtered_df = filtered_df.query(f"primary_job_type == '{selected_job_type}'")
                else:
                    st.info("No job types with 10+ jobs in current selection")
                    filters['primary_job_type'] = 'All'
            
            # Role Category Filter
            if 'standardized_title' in filtered_df.columns and not filtered_df.empty:
                st.subheader("🎯 Role Category Filter")
                role_counts = filtered_df['standardized_title'].value_counts()
                valid_roles = role_counts[role_counts > 10]
                if len(valid_roles) > 0:
                    roles = ['All'] + valid_roles.index.tolist()
                    selected_role = st.selectbox(
                        "Select Role Category",
                        roles,
                        index=0,
                        help="Filter jobs by role category (only showing roles with 10+ jobs)"
                    )
                    filters['standardized_title'] = selected_role
                    
                    if selected_role != 'All':
                        filtered_df = filtered_df.query(f"standardized_title == '{selected_role}'")
                else:
                    st.info("No roles with 10+ jobs in current selection")
                    filters['standardized_title'] = 'All'
            
            # Seniority Level Filter
            if 'seniority_level' in filtered_df.columns and not filtered_df.empty:
                st.subheader("📊 Seniority Level Filter")
                seniority_counts = filtered_df['seniority_level'].value_counts()
                valid_seniorities = seniority_counts[seniority_counts > 10]
                if len(valid_seniorities) > 0:
                    seniorities = ['All'] + valid_seniorities.index.tolist()
                    selected_seniority = st.selectbox(
                        "Select Seniority Level",
                        seniorities,
                        index=0,
                        help="Filter jobs by seniority level (only showing levels with 10+ jobs)"
                    )
                    filters['seniority_level'] = selected_seniority
                else:
                    st.info("No seniority levels with 10+ jobs in current selection")
                    filters['seniority_level'] = 'All'

            st.markdown("---")
            
            return filters
    
    def apply_filters(self, filters: Dict) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Apply filters to the dataset"""
        if self.skills_df.empty or self.df.empty:
            return pd.DataFrame(), pd.DataFrame()
            
        df_filtered = self.skills_df.copy()
        main_df = self.df.copy()
        
        # Date filter first
        if filters.get('date_range') and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            if start_date and end_date:
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                df_filtered = df_filtered[
                    (pd.to_datetime(df_filtered['published'], format='%Y.0_%m.0') >= start_date) &  
                    (pd.to_datetime(df_filtered['published'], format='%Y.0_%m.0') <= end_date)
                ]
                main_df = main_df[
                    (pd.to_datetime(main_df['published'], format='%Y.0_%m.0') >= start_date) & 
                    (pd.to_datetime(main_df['published'], format='%Y.0_%m.0') <= end_date)
                ]
        
        # Country filter
        if filters.get('country') and filters['country'] != 'All':
            df_filtered = df_filtered.query(f"country == '{filters['country']}'")  
            main_df = main_df.query(f"country == '{filters['country']}'") 
        
        # Company filter
        if filters.get('company_name') and filters['company_name'] != 'All':
            df_filtered = df_filtered.query(f"company == '{filters['company_name']}'")  
            main_df = main_df.query(f"company == '{filters['company_name']}'") 
        
        # Primary job type filter
        if filters.get('primary_job_type') and filters['primary_job_type'] != 'All':
            df_filtered = df_filtered.query(f"primary_job_type == '{filters['primary_job_type']}'")  
            main_df = main_df.query(f"primary_job_type == '{filters['primary_job_type']}'") 
        
        # Role filter
        if filters.get('standardized_title') and filters['standardized_title'] != 'All':
            df_filtered = df_filtered.query(f"standardized_title == '{filters['standardized_title']}'") 
            main_df = main_df.query(f"standardized_title == '{filters['standardized_title']}'") 
        
        # Seniority filter
        if filters.get('seniority_level') and filters['seniority_level'] != 'All':
            df_filtered = df_filtered.query(f"seniority_level == '{filters['seniority_level']}'")  
            main_df = main_df.query(f"seniority_level == '{filters['seniority_level']}'") 
                
        # Skills filter
        if filters.get('skills'):
            def has_skills(skills_str, target_skills):
                if pd.isna(skills_str):
                    return False
                try:
                    skills = eval(skills_str) if isinstance(skills_str, str) else skills_str
                    return any(skill in skills for skill in target_skills)
                except:
                    return False
            
            mask = df_filtered['skill'].apply(
                lambda x: has_skills(x, filters['skills'])
            )
            df_filtered = df_filtered[mask] 
            
        return main_df, df_filtered

# Initialize dashboard manager
@st.cache_resource
def get_dashboard_manager():
    return DashboardManager()

def main():
    """Main function to run the dashboard"""
    st.markdown('<h1 class="main-header">Data Science Job Market Analytics</h1>', 
               unsafe_allow_html=True)
    
    st.markdown("""
    **Interactive dashboard for analyzing data science job market trends, skill requirements, 
    and geographic distribution. Use the navigation on the left to explore different aspects of the data.**
     
    """)
    
    # Initialize dashboard manager
    dashboard_manager = get_dashboard_manager()
    
    # Load data if not already loaded
    dashboard_manager.load_and_process_data()
    
    # Show overview metrics
    st.markdown("---")
    st.subheader("📈 Quick Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        total_jobs = len(dashboard_manager.df) if not dashboard_manager.df.empty else 0
        st.metric("Total Jobs", f"{total_jobs:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        companies = dashboard_manager.df['company_name'].nunique() if not dashboard_manager.df.empty else 0
        st.metric("Companies", f"{companies:,}")
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if not dashboard_manager.df.empty and 'skills_count' in dashboard_manager.df.columns:
            avg_skills = dashboard_manager.df['skills_count'].mean()
        else:
            avg_skills = 0
        st.metric("Avg Skills/Job", f"{avg_skills:.1f}")
        st.markdown('</div>', unsafe_allow_html=True)
            
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        if not dashboard_manager.df.empty and 'avg_salary_usd' in dashboard_manager.df.columns:
            avg_salary = dashboard_manager.df['avg_salary_usd'].mean()
            st.metric(f"Avg Salary", f"${avg_salary:,.0f} " if not pd.isna(avg_salary) else "N/A")
        else:
            st.metric("Analysis Modules", "6 Available")
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown("---")
        # Page descriptions
    st.subheader("📖 Dashboard Sections")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **📊 Overview**  
        High-level metrics and dataset summary
        """)
        
        st.info("""
        **🛠️ Skills Analysis**  
        Detailed skill demand and relationships
        *Uses: SkillAnalyzer*
        """)
        
        st.info("""
        **📈 Trends**  
        Temporal analysis of skill popularity
        *Uses: TrendAnalyzer*
        """)
    
    with col2:
        st.info("""
        **🎯 Role Comparison**  
        Compare skills across different job roles
        *Uses: RoleAnalyzer*
        """)
        
        st.info("""
        **👨‍💼 Seniority Patterns**  
        Skill progression across career levels
        *Uses: SeniorityAnalyzer*
        """)
        
        st.info("""
        **🌐 Skill Ecosystem**  
        Complete skill network and clusters
        *Uses: EcosystemAnalyzer*
        """)
    
  
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Data Science Job Market Dashboard • Built with Streamlit • Modular Architecture</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()