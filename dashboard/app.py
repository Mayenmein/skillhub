# dashboard/app.py

import streamlit as st
import pandas as pd 
import numpy as np
import plotly.express as px
import plotly.graph_objects as go 
from pathlib import Path 
import sys
import networkx as nx
from typing import Dict,List
from datetime import datetime

from functools import lru_cache
# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

# Import existing analysis modules
#from scr.visualization.dashboard import DataScienceJobsDashboard
from src.analyze_jobs import DataScienceJobsAnalyzer 

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

class EnhancedDataScienceJobsDashboard:
    """
    Enhanced dashboard with additional features and improved UI
    """
    
    def __init__(self, data_dir: str = "data"):
        """Initialize the enhanced dashboard""" 

        self.analyzer = DataScienceJobsAnalyzer(data_dir)

        SKILLS_DATA_LOCATION = Path(f'{data_dir}/processed/skill_pivot.parquet')
        DATA_INTERIM_PATH = Path(f'{data_dir}/interim/cleaned_jobs_data.csv')

        self.df = pd.read_csv(DATA_INTERIM_PATH)
        self.skills_df = pd.read_parquet(SKILLS_DATA_LOCATION) 

        self.analyzer = DataScienceJobsAnalyzer('data')
    
    def render_enhanced_dashboard(self):
        """Render the enhanced dashboard with additional features"""
        
        # Header section
        st.markdown('<h1 class="main-header">Data Science Job Market Analytics</h1>', 
                   unsafe_allow_html=True)
        st.markdown("""
        **Interactive dashboard for analyzing data science job market trends, skill requirements, 
        and geographic distribution. Use the filters in the sidebar to explore the data.**
        """)
        
        # Sidebar with enhanced filters
        filters = self.setup_enhanced_sidebar()
        
        # Apply filters and get filtered data
        main_df, filtered_df = self._apply_filters(filters)
        
        # Show filter summary
        self._render_filter_summary(filters, main_df)
        
        # Summary metrics with enhanced design
        self._render_enhanced_metrics(main_df)
        
        # Main content tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📊 Skills Analysis",  
            "📅 Temporal Trends", 
            "🎯 Role Comparison", 
            "📈 Salary Analysis"
        ])
        
        with tab1:
            self._render_enhanced_skills_analysis(filtered_df)
        
        with tab2:
            self.plot_skill_trends_dashboard(filtered_df) 
        
        with tab3:
            self._render_enhanced_role_comparison(filtered_df) 
        
        with tab4:
            self._render_enhanced_salary_analysis(main_df) 
        
        # Footer
        self._render_footer()
    
    def _apply_filters(self, filters: Dict) -> tuple[pd.DataFrame,pd.DataFrame]:
        """Apply filters to the dataset"""
        df_filtered = self.skills_df.copy()
        main_df = self.df.copy()
        
        # Date filter first
        if filters.get('date_range') and len(filters['date_range']) == 2:
            start_date, end_date = filters['date_range']
            if start_date and end_date:
                # Convert to datetime for comparison
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                
                df_filtered = df_filtered[
                    (pd.to_datetime(df_filtered['date'], format='%Y.0_%m.0') >= start_date) & 
                    (pd.to_datetime(df_filtered['date'], format='%Y.0_%m.0') <= end_date)
                ]
                main_df = main_df[
                    (pd.to_datetime(main_df['date'], format='%Y.0_%m.0') >= start_date) & 
                    (pd.to_datetime(main_df['date'], format='%Y.0_%m.0') <= end_date)
                ]
        
        # Country filter
        if filters.get('country') and filters['country'] != 'All':
            df_filtered = df_filtered.query(f"country == '{filters['country']}'")  
            main_df = main_df.query(f"country == '{filters['country']}'") 
        
        # Company filter
        if filters.get('company') and filters['company'] != 'All':
            df_filtered = df_filtered.query(f"company == '{filters['company']}'")  
            main_df = main_df.query(f"company == '{filters['company']}'") 
        
        # Primary job type filter
        if filters.get('primary_job_type') and filters['primary_job_type'] != 'All':
            df_filtered = df_filtered.query(f"primary_job_type == '{filters['primary_job_type']}'")  
            main_df = main_df.query(f"primary_job_type == '{filters['primary_job_type']}'") 
        
        # Role filter
        if filters.get('cleaned_title_category') and filters['cleaned_title_category'] != 'All':
            df_filtered = df_filtered.query(f"cleaned_title_category == '{filters['cleaned_title_category']}'") 
            main_df = main_df.query(f"cleaned_title_category == '{filters['cleaned_title_category']}'") 
        
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

    def setup_enhanced_sidebar(self):
        """Create enhanced sidebar with additional filters and info"""
        with st.sidebar:
            st.title("Dashboard Controls")
            st.markdown("---")
            
            # Data info card
            st.subheader("📁 Dataset Info")
            st.info(f"""
            **Total Jobs:** {len(self.df):,}
            **Companies:** {self.df['company'].nunique():,}  
            **Last Updated:** {datetime.now().strftime('%Y-%m-%d')}
            """)
            
            st.markdown("---")
            
            # Enhanced filters
            filters = {}
            
            # Start with base dataset for filtering
            filtered_df = self.df.copy()
            
            # Date Range Filter
            st.subheader("📅 Date Range Filter")
            if 'date' in filtered_df.columns:
                # Convert date column to datetime for proper filtering
                try:
                    min_date = pd.to_datetime(filtered_df['date'], format='%Y.0_%m.0').min()
                    max_date = pd.to_datetime(filtered_df['date'], format='%Y.0_%m.0').max()
                    
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
                            (pd.to_datetime(filtered_df['date'], format='%Y.0_%m.0') >= pd.to_datetime(start_date)) & 
                            (pd.to_datetime(filtered_df['date'], format='%Y.0_%m.0') <= pd.to_datetime(end_date))
                        ]
                except Exception as e:
                    st.warning(f"Date filtering may not work properly: {e}")
            
            st.markdown("---")
            
            # Country Filter - sorted by count descending, only include with >10 records
            if 'country' in filtered_df.columns:
                st.subheader("🌍 Country Filter")
                country_counts = filtered_df['country'].value_counts()
                # Filter countries with more than 10 records and sort by count descending
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
                    
                    # Apply country filter to get available companies
                    if selected_country != 'All':
                        filtered_df = filtered_df.query(f"country == '{selected_country}'")
                else:
                    st.info("No countries with 10+ jobs in current selection")
                    filters['country'] = 'All'
            
            # Company Filter - sorted by count descending, only include with >10 records
            if 'company' in filtered_df.columns:
                st.subheader("🏢 Company Filter")
                company_counts = filtered_df['company'].value_counts()
                # Filter companies with more than 10 records and sort by count descending
                valid_companies = company_counts[company_counts > 10]
                if len(valid_companies) > 0:
                    companies = ['All'] + valid_companies.index.tolist()
                    selected_company = st.selectbox(
                        "Select Company",
                        companies,
                        index=0,
                        help="Filter jobs by company (only showing companies with 10+ jobs)"
                    )
                    filters['company'] = selected_company
                    
                    # Apply company filter to get available job types
                    if selected_company != 'All':
                        filtered_df = filtered_df.query(f"company == '{selected_company}'")
                else:
                    st.info("No companies with 10+ jobs in current selection")
                    filters['company'] = 'All'
            
            # Primary Job Type Filter - sorted by count descending, only include with >10 records
            if 'primary_job_type' in filtered_df.columns:
                st.subheader("💼 Job Type Filter")
                job_type_counts = filtered_df['primary_job_type'].value_counts()
                # Filter job types with more than 10 records and sort by count descending
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
                    
                    # Apply job type filter to get available roles
                    if selected_job_type != 'All':
                        filtered_df = filtered_df.query(f"primary_job_type == '{selected_job_type}'")
                else:
                    st.info("No job types with 10+ jobs in current selection")
                    filters['primary_job_type'] = 'All'
            
            # Role Category Filter - sorted by count descending, only include with >10 records
            if 'cleaned_title_category' in filtered_df.columns:
                st.subheader("🎯 Role Category Filter")
                role_counts = filtered_df['cleaned_title_category'].value_counts()
                # Filter roles with more than 10 records and sort by count descending
                valid_roles = role_counts[role_counts > 10]
                if len(valid_roles) > 0:
                    roles = ['All'] + valid_roles.index.tolist()
                    selected_role = st.selectbox(
                        "Select Role Category",
                        roles,
                        index=0,
                        help="Filter jobs by role category (only showing roles with 10+ jobs)"
                    )
                    filters['cleaned_title_category'] = selected_role
                    
                    # Apply role filter to get available seniority levels
                    if selected_role != 'All':
                        filtered_df = filtered_df.query(f"cleaned_title_category == '{selected_role}'")
                else:
                    st.info("No roles with 10+ jobs in current selection")
                    filters['cleaned_title_category'] = 'All'
            
            # Seniority Level Filter - sorted by count descending, only include with >10 records
            if 'seniority_level' in filtered_df.columns:
                st.subheader("📊 Seniority Level Filter")
                seniority_counts = filtered_df['seniority_level'].value_counts()
                # Filter seniority levels with more than 10 records and sort by count descending
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
    
    def _extract_unique_skills(self) -> List[str]:
        """Extract unique skills from existing data"""
        skills_set = set(self.skills_df['skill'].unique())
        return sorted(list(skills_set))

    def _render_filter_summary(self, filters: Dict, filtered_df: pd.DataFrame):
        """Show current filter summary and active filters"""
        total = len(self.df)
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
                    # If parsing fails, skip date formatting
                    pass

            # Common categorical filters
            for key in ['country', 'company', 'primary_job_type', 'cleaned_title_category', 'seniority_level']:
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
            # No row reduction; still show active filters if any
            if active_parts:
                st.info(f"**Active Filters (no row reduction):** {', '.join(active_parts)}")
            else:
                st.info("**No filters applied:** Showing all available jobs")
    
    def _render_enhanced_metrics(self, df: pd.DataFrame):
        """Render enhanced metrics cards"""
        st.markdown("---")
        
        # Create columns for metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            st.metric("Total Jobs", f"{len(df):,}")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="metric-card">', unsafe_allow_html=True)
            companies = df['company'].nunique()
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
                st.metric(f"Avg Salary (From {df['avg_salary_usd'].notna().sum()} jobs)", f"${avg_salary:,.0f} " if not pd.isna(avg_salary) else "N/A")
            else:
                st.metric("Data Freshness", " days")
            st.markdown('</div>', unsafe_allow_html=True)
        
    def _render_enhanced_skills_analysis(self, filtered_df: pd.DataFrame):
        """Enhanced skills analysis with interactive co-occurrence insights"""
        st.header("Skills Analysis") 

        if filtered_df.empty:
            st.warning("No data available with current filters")
            return

        # --- Aggregate skills and categories ---
        skill_df = self.analyzer.aggregate_pivot(filtered_df,metric='prevalence')
        skill_df['prevalence'] = skill_df['prevalence'].round(1)
        category_df = self.analyzer.aggregate_pivot(filtered_df, column='skill_category')       

        # --- Top skills ---
        top_skills = skill_df[['skill', 'mentions', 'prevalence']].iloc[:15]

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Top Skills by Prevalence")
            fig = px.bar(
                top_skills,
                x="mentions",
                y="skill",
                orientation="h",
                title="Most In-Demand Skills",
                labels={"mentions": "Mentions", "skill": "Skill"},
                color="prevalence",
                color_continuous_scale="Blues"
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, config={"responsive": True})
            
            # Responsive caption
            top_skill = top_skills.iloc[0]
            st.caption(f"{top_skill['skill']} leads with {int(top_skill['prevalence'])} % across job postings, "
                    f"showing strongest market demand in the current dataset.")

        with col2:
            st.subheader("Skill Importance Matrix")
            fig = px.scatter(
                top_skills,
                x="prevalence",
                y="mentions",
                size=top_skills["mentions"] * top_skills["prevalence"],
                hover_name="skill",
                title="Mentions vs Prevalence",
                labels={'prevalence': 'Prevalence (% of Jobs)', 'mentions': 'Mentions'},
                size_max=30,
                color="mentions",
                color_continuous_scale="Viridis"
            )
            st.plotly_chart(fig, config={"responsive": True})
            
            # Responsive caption
            high_prevalence = top_skills.loc[top_skills['prevalence'].idxmax()]
            balanced_skill = top_skills.loc[(top_skills['mentions'] / top_skills['mentions'].max() + 
                                        top_skills['prevalence'] / top_skills['prevalence'].max()).idxmax()]
            
            st.caption(f"{high_prevalence['skill']} appears in {high_prevalence['prevalence']:.1f}% of jobs, "
                    f"while {balanced_skill['skill']} shows balanced high demand and wide adoption across roles.")
                        # --- Skills Network ---# --- Skills Network Analysis ---
        st.subheader("Skill Relationship Mapping")

        # Data processing
        skill_jobs = filtered_df[["job_ids", "skill"]].dropna().copy()
        skill_jobs_exploded = skill_jobs.explode('job_ids')
        skill_jobs_clean = skill_jobs_exploded.dropna(subset=["job_ids", "skill"])
        skill_jobs_clean["job_id_str"] = skill_jobs_clean["job_ids"].astype(str)

        if not skill_jobs_clean.empty:
            # Get top skills
            skill_counts = skill_jobs_clean['skill'].value_counts()
            top_skills = skill_counts.head(15).index.tolist()
            
            if len(top_skills) > 1:
                # Create pivot and co-occurrence matrix
                skill_jobs_top = skill_jobs_clean[skill_jobs_clean['skill'].isin(top_skills)]
                pivot = pd.crosstab(skill_jobs_top["job_id_str"], skill_jobs_top["skill"])
                co_matrix = pivot.T.dot(pivot)
                np.fill_diagonal(co_matrix.values, 0)
                 
                G = nx.Graph()
                for skill in top_skills:
                    G.add_node(skill, frequency=skill_counts[skill])
                
                # Add edges for strong co-occurrences
                for i, skill1 in enumerate(top_skills):
                    for j, skill2 in enumerate(top_skills):
                        if i < j:
                            weight = co_matrix.at[skill1, skill2]
                            if weight >= 10:
                                G.add_edge(skill1, skill2, weight=weight)
                
                if G.number_of_edges() > 0:
                    # Calculate metrics for responsive insights
                    degrees = dict(G.degree())
                    weighted_degrees = dict(G.degree(weight='weight'))
                    
                    # Find most important skill dynamically
                    most_central_skill = max(degrees.items(), key=lambda x: x[1])
                    strongest_pair = max(G.edges(data=True), key=lambda x: x[2]['weight'])
                    strongest_pair_names = f"{strongest_pair[0]}-{strongest_pair[1]}"
                    
                    # Two column layout
                    col1, col2 = st.columns([1.2, 1])
                    
                    with col1:
                        # Network Visualization
                        pos = nx.spring_layout(G, k=1.5, iterations=50, seed=42)
                        
                        # Edges
                        edge_x, edge_y, edge_text = [], [], []
                        max_weight = max([d['weight'] for u, v, d in G.edges(data=True)])
                        
                        for u, v, d in G.edges(data=True):
                            x0, y0 = pos[u]
                            x1, y1 = pos[v]
                            edge_x += [x0, x1, None]
                            edge_y += [y0, y1, None]
                            width = 0.5 + (d['weight'] / max_weight) * 2
                            edge_text.append(f"{u} + {v}: {d['weight']} jobs")
                        
                        edge_trace = go.Scatter(
                            x=edge_x, y=edge_y,
                            line=dict(width=1.5, color='rgba(120, 120, 120, 0.4)'),
                            hoverinfo='text',
                            hovertext=edge_text,
                            mode='lines'
                        )
                        
                        # Nodes
                        node_x, node_y, node_text, node_sizes = [], [], [], []
                        max_freq = max([skill_counts[skill] for skill in G.nodes()])
                        
                        for node in G.nodes():
                            x, y = pos[node]
                            node_x.append(x)
                            node_y.append(y)
                            
                            node_text.append(
                                f"{node}<br>"
                                f"Jobs: {skill_counts[node]}<br>"
                                f"Connections: {degrees[node]}"
                            )
                            
                            # Size based on job frequency
                            node_sizes.append(20 + (skill_counts[node] / max_freq) * 25)
                        
                        node_trace = go.Scatter(
                            x=node_x, y=node_y,
                            mode='markers+text',
                            text=[n for n in G.nodes()],
                            textposition="middle center",
                            hoverinfo='text',
                            hovertext=node_text,
                            marker=dict(
                                size=node_sizes,
                                color=list(degrees.values()),
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Connections"),
                                line=dict(width=1, color='darkgray')
                            )
                        )
                        
                        fig_network = go.Figure(data=[edge_trace, node_trace],
                            layout=go.Layout(
                                showlegend=False,
                                hovermode='closest',
                                margin=dict(b=0, l=0, r=0, t=0),
                                xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                                height=400
                            )
                        )
                        st.plotly_chart(fig_network, use_container_width=True)
                        
                        # Responsive caption
                        st.caption(f"Network shows {most_central_skill[0]} as the most connected skill with {most_central_skill[1]} relationships. "
                                f"Size indicates job market demand, color shows network influence.")
                    
                    with col2:
                        # Heatmap with enhanced hover
                        hover_text = []
                        for i, skill1 in enumerate(co_matrix.index):
                            row = []
                            for j, skill2 in enumerate(co_matrix.columns):
                                count = co_matrix.iloc[i, j]
                                row.append(f"{skill1} + {skill2}<br>Shared Jobs: {count}")
                            hover_text.append(row)

                        fig_heatmap = go.Figure(data=go.Heatmap(
                            z=co_matrix.values,
                            x=co_matrix.columns,
                            y=co_matrix.index,
                            colorscale='Blues',
                            hoverinfo='text',
                            text=hover_text,
                            hovertemplate='%{text}<extra></extra>'
                        ))
                        
                        fig_heatmap.update_layout(
                            margin=dict(b=0, l=0, r=0, t=0),
                            xaxis=dict(tickangle=45, tickfont=dict(size=9)),
                            yaxis=dict(tickfont=dict(size=9)),
                            height=400
                        )
                        st.plotly_chart(fig_heatmap, use_container_width=True)
                        
                        # Responsive caption
                        st.caption(f"Strongest skill partnership: {strongest_pair_names} with {strongest_pair[2]['weight']} shared job requirements. "
                                f"Darker cells indicate more frequent combinations.")
        # --- Skill categories ---
        st.subheader("Skill Categories")

        col3, col4 = st.columns(2)
        with col3:
            top_mentions = category_df.sort_values("mentions", ascending=False).head(10)
            fig = px.bar(
                top_mentions,
                x="mentions",
                y="skill_category",
                orientation="h",
                title="Top Categories by Mentions",
                labels={"mentions": "Number of Mentions", "skill_category": "Category"},
                color="mentions",
                color_continuous_scale="Blues"
            )
            fig.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig, config={"responsive": True})
            
            # Responsive caption
            top_category_mentions = top_mentions.iloc[0]
            st.caption(f"{top_category_mentions['skill_category']} category dominates with {int(top_category_mentions['mentions'])} total skill mentions, "
                    f"indicating deep specialization demand within this domain.")

        with col4:
            top_prevalence = category_df.sort_values("prevalence", ascending=False).head(10)
            fig_prevalence = px.bar(
                top_prevalence,
                x="prevalence",
                y="skill_category",
                orientation="h",
                title="Top Categories by Prevalence",
                labels={"prevalence": "Prevalence (% of Jobs)", "skill_category": "Category"},
                color="prevalence",
                color_continuous_scale="Greens"
            )
            fig_prevalence.update_layout(yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_prevalence, use_container_width=True)
            
            # Responsive caption
            top_category_prevalence = top_prevalence.iloc[0]
            st.caption(f"{top_category_prevalence['skill_category']} appears in {top_category_prevalence['prevalence']:.1f}% of all jobs, "
                    f"showing this category's fundamental importance across diverse roles and positions.")
                 
    def plot_skill_trends_dashboard(self, pivot_df: pd.DataFrame):
        """
        Interactive dashboard for analyzing skill/category trends.
        User can choose skill vs category, select top N, and view side-by-side or combined dual-axis plots.
        """
        if pivot_df.empty:
            st.warning("⚠️ Pivot table is empty, cannot plot trends.")
            return

        df = pivot_df.copy()
        df["date"] = pd.to_datetime(df["date"], format="%Y.0_%m.0")

        # --- Column selector / view mode ---
        view_mode = st.selectbox(
            "Choose view:",
            options=["By Skill", "By Skill Category", "By Trend Category"],
            index=0
        )

        # Normalize column_choice for backward compatibility
        if view_mode == "By Skill":
            column_choice = "skill"
        elif view_mode == "By Skill Category":
            column_choice = "skill_category"
        else:
            column_choice = "skill"  # we'll aggregate by trend category after computing skill trends

        # --- Explode job_ids ---
        exploded = df.explode("job_ids").dropna(subset=["job_ids"]).copy()
        exploded["job_ids"] = exploded["job_ids"].astype(str)

        # If viewing by trend category we need skill-level mentions and trend labels
        trends_df = None
        if view_mode == "By Trend Category":
            try:
                trends_df = self.analyzer.analyze_skill_trends_full(pivot_df)
            except Exception as e:
                st.warning(f"Could not compute trend categories: {e}")
                trends_df = None

        # --- Group by date + column ---
        grouped = exploded.groupby(["date", column_choice])
        mentions_df = grouped["job_ids"].nunique().reset_index(name="mentions")

        # --- Total jobs per date ---
        total_jobs_per_date = exploded.groupby("date")["job_ids"].nunique().reset_index(name="total_jobs")
        merged = mentions_df.merge(total_jobs_per_date, on="date")
        merged["prevalence"] = (merged["mentions"] / merged["total_jobs"]) * 100

        # --- Top N selector / selection UI ---
        st.markdown("### Selection")
        top_n = st.slider("Select Top N by Mentions", min_value=2, max_value=30, value=5, step=1)

        latest_date = merged["date"].max()
        top_n_items = (
            merged[merged["date"] == latest_date]
            .nlargest(top_n, "mentions")[column_choice]
            .tolist()
        )

        if view_mode != "By Trend Category":
            available_options = sorted(merged[column_choice].unique())
            selected_items = st.multiselect(
                f"Select {column_choice.title()}s to Plot:",
                options=available_options,
                default=top_n_items
            )
        else:
            # Trend Category mode: allow selecting categories and choose aggregated vs top-N
            if trends_df is None or trends_df.empty:
                st.info("Trend categories unavailable — falling back to skill view.")
                available_options = sorted(merged[column_choice].unique())
                selected_items = st.multiselect(
                    f"Select {column_choice.title()}s to Plot:",
                    options=available_options,
                    default=top_n_items
                )
                view_mode = "By Skill"
            else:
                cat_options = sorted(trends_df['trend_category'].unique().tolist())
                selected_cats = st.multiselect("Select Trend Categories:", options=cat_options, default=cat_options)
                agg_mode = st.radio("Trend view mode:", options=["Average by Category", f"Top {top_n} Skills per Category"], index=0)
                # build merged_skill (prevalence per skill per date) for further aggregation
                exploded_skill = df.explode("job_ids").dropna(subset=["skill", "job_ids"]).copy()
                exploded_skill["job_ids"] = exploded_skill["job_ids"].astype(str)
                grouped_skill = exploded_skill.groupby(["date", "skill"])['job_ids'].nunique().reset_index(name='mentions')
                total_jobs_per_date = exploded_skill.groupby('date')['job_ids'].nunique().reset_index(name='total_jobs')
                merged_skill = grouped_skill.merge(total_jobs_per_date, on='date')
                merged_skill['prevalence'] = (merged_skill['mentions'] / merged_skill['total_jobs']) * 100
                # Merge trend labels
                merged_skill = merged_skill.merge(trends_df[['skill','trend_category']], on='skill', how='left')
                # Filter categories
                merged_skill = merged_skill[merged_skill['trend_category'].isin(selected_cats)]
                # We'll handle plotting below and return early from selection logic
                selected_items = None

        if view_mode != "By Trend Category" and not selected_items:
            st.info("👆 Please select at least one option to see the trends.")
            return

        # --- Filter ---
        merged = merged[merged[column_choice].isin(selected_items)]

        # --- Toggle for visualization mode ---
        mode = st.radio(
            "Choose visualization mode:",
            options=["Combined Dual-axis", "Side-by-side"],
            index=0,
            horizontal=True
        )

        if mode == "Side-by-side":
            col1, col2 = st.columns(2)
            
            with col1:
                # If trend-category aggregated view is active, plot aggregated lines
                if view_mode == "By Trend Category":
                    # average prevalence per category over time
                    avg_df = merged_skill.groupby(['date','trend_category'])['prevalence'].mean().reset_index()
                    fig_mentions = px.line(
                        avg_df,
                        x='date', y='prevalence', color='trend_category', markers=True,
                        title='Average Prevalence by Trend Category Over Time',
                        labels={'date':'date','prevalence':'Prevalence (% of Jobs)','trend_category':'Trend Category'}
                    )
                else:
                    fig_mentions = px.line(
                        merged,
                        x="date", y="mentions", color=column_choice,
                        markers=True,
                        title=f"{column_choice.title()} Mentions Trends Over Time",
                        labels={"date": "date", "mentions": "Mentions (# of Jobs)", column_choice: column_choice.title()}
                    )
                fig_mentions.update_layout(legend_title=column_choice.title(), xaxis=dict(tickformat="%b %Y"))
                st.plotly_chart(fig_mentions, use_container_width=True)
                
                # Responsive caption for mentions
                if view_mode == "By Trend Category":
                    latest_data = avg_df[avg_df['date'] == latest_date]
                    if not latest_data.empty:
                        top_grower = latest_data.loc[latest_data['prevalence'].idxmax()]
                        st.caption(f"**Highest average prevalence**: {top_grower['trend_category']} with {top_grower['prevalence']:.1f}% of jobs")
                else:
                    latest_data = merged[merged["date"] == latest_date]
                    if not latest_data.empty:
                        top_grower = latest_data.loc[latest_data['mentions'].idxmax()]
                        st.caption(f"**Highest demand**: {top_grower[column_choice]} with {int(top_grower['mentions'])} job postings")

            with col2:
                # Prevalence trend
                if view_mode == "By Trend Category":
                    fig_prevalence = px.line(
                        avg_df,
                        x='date', y='prevalence', color='trend_category', markers=True,
                        title='Average Prevalence by Trend Category Over Time',
                        labels={'date':'date','prevalence':'Prevalence (% of Jobs)','trend_category':'Trend Category'}
                    )
                else:
                    fig_prevalence = px.line(
                        merged,
                        x="date", y="prevalence", color=column_choice,
                        markers=True,
                        title=f"{column_choice.title()} Prevalence Trends Over Time",
                        labels={"date": "date", "prevalence": "Prevalence (% of Jobs)", column_choice: column_choice.title()}
                    )
                fig_prevalence.update_layout(legend_title=column_choice.title(), xaxis=dict(tickformat="%b %Y"))
                st.plotly_chart(fig_prevalence, use_container_width=True)
                
                # Responsive caption for prevalence
                latest_data = merged[merged["date"] == latest_date]
                if not latest_data.empty:
                    if view_mode == "By Trend Category":
                        most_prevalent = latest_data.loc[latest_data['prevalence'].idxmax()]
                        st.caption(f"**Widest adoption**: {most_prevalent['trend_category']} in {most_prevalent['prevalence']:.1f}% of all jobs")
                    else:
                        most_prevalent = latest_data.loc[latest_data['prevalence'].idxmax()]
                        st.caption(f"**Widest adoption**: {most_prevalent[column_choice]} in {most_prevalent['prevalence']:.1f}% of all jobs")

        else:  # Combined Dual-axis
            st.subheader("Combined Mentions & Prevalence Trends")
            st.caption("Blue lines = Number of jobs requiring this skill | Green lines = Percentage of all jobs requiring this skill")

            for item in selected_items:
                subset = merged[merged[column_choice] == item]
                if subset.empty:
                    continue

                fig = go.Figure()

                # Mentions on left axis
                fig.add_trace(go.Scatter(
                    x=subset["date"], y=subset["mentions"],
                    mode="lines+markers", name=f"{item} - Mentions",
                    line=dict(color="blue")
                ))

                # Prevalence on right axis
                fig.add_trace(go.Scatter(
                    x=subset["date"], y=subset["prevalence"],
                    mode="lines+markers", name=f"{item} - Prevalence",
                    line=dict(color="green"), yaxis="y2"
                ))

                # Layout with dual axis
                fig.update_layout(
                    title=f"{item} - Mentions & Prevalence Over Time",
                    xaxis=dict(title="date", tickformat="%b %Y"),
                    yaxis=dict(title="Mentions (# of Jobs)", side="left"),
                    yaxis2=dict(title="Prevalence (% of Jobs)", overlaying="y", side="right"),
                    legend=dict(x=1.05, y=1),
                    margin=dict(l=40, r=40, t=60, b=40)
                )

                st.plotly_chart(fig, use_container_width=True)
                
                # Individual trend caption with skill name
                if len(subset) > 1:
                    current_mentions = subset['mentions'].iloc[-1]
                    prev_mentions = subset['mentions'].iloc[0]
                    current_prevalence = subset['prevalence'].iloc[-1]
                    prev_prevalence = subset['prevalence'].iloc[0]
                    
                    if current_mentions > prev_mentions and current_prevalence > prev_prevalence:
                        st.caption(f"**{item}**: Growing fast - both job count and market share increasing")
                    elif current_mentions > prev_mentions and current_prevalence <= prev_prevalence:
                        st.caption(f"**{item}**: Specializing - more jobs but concentrated in specific roles")
                    elif current_mentions <= prev_mentions and current_prevalence > prev_prevalence:
                        st.caption(f"**{item}**: Spreading - fewer jobs but across more types of roles")
                    else:
                        st.caption(f"**{item}**: Declining - both job count and market share decreasing")
                        
    def _render_enhanced_role_comparison(self, pivot_df: pd.DataFrame):
        """Optimized role comparison with better performance."""
        st.header("Role Comparison (Skills & Categories)")

        if pivot_df.empty or "cleaned_title_category" not in pivot_df.columns:
            st.warning("⚠️ Role data not available in pivot table.")
            return

        # --- UI Controls ---
        col_config, col_roles = st.columns([1, 2])
        
        with col_config:
            column_choice = st.radio("Compare by:", ["skill", "skill_category"], index=1, horizontal=True)
            metric_choice = st.radio("Metric:", ["mentions", "prevalence"], index=1, horizontal=True)
            top_n = st.slider("Top N to display:", 5, 50, 15, 5)

        # --- Role selection ---
        roles = pivot_df["cleaned_title_category"].dropna().unique().tolist()
        if len(roles) < 2:
            st.warning("⚠️ Need at least 2 roles for comparison.")
            return

        with col_roles:
            role_col1, role_col2 = st.columns(2)
            with role_col1:
                role1 = st.selectbox("First Role", roles, key="role1_comp")
            with role_col2:
                available_roles = [r for r in roles if r != role1]
                role2 = st.selectbox("Second Role", available_roles, key="role2_comp")

        if not (role1 and role2):
            st.info("👆 Please select two roles to compare.")
            return
        
        # Filter and process both roles in one operation
        role_data = pivot_df[pivot_df["cleaned_title_category"].isin([role1, role2])].copy()
        
        # Explode job_ids once for both roles
        role_data_exploded = role_data.explode("job_ids")
        role_data_exploded = role_data_exploded.dropna(subset=[column_choice, "job_ids"])
        role_data_exploded["job_ids"] = role_data_exploded["job_ids"].astype(str)
        
        # Calculate metrics in one groupby operation
        grouped = role_data_exploded.groupby(["cleaned_title_category", column_choice])
        skill_mentions = grouped["job_ids"].nunique().reset_index(name="mentions")
        
        # Calculate total jobs per role
        role_totals = role_data_exploded.groupby("cleaned_title_category")["job_ids"].nunique().reset_index(name="total_jobs")
        
        # Merge and calculate prevalence
        merged_full = skill_mentions.merge(role_totals, on="cleaned_title_category")
        merged_full["prevalence"] = (merged_full["mentions"] / merged_full["total_jobs"]) * 100
        
        # --- Prepare comparison data ---
        # Pivot to get role1 vs role2 comparison
        comparison_data = merged_full.pivot_table(
            index=column_choice,
            columns="cleaned_title_category",
            values=metric_choice,
            aggfunc='first'
        ).fillna(0)
        
        # Ensure both roles are present (in case one has no data for some skills)
        for role in [role1, role2]:
            if role not in comparison_data.columns:
                comparison_data[role] = 0
        
        # Calculate top N by combined metric
        comparison_data["total"] = comparison_data[role1] + comparison_data[role2]
        top_comparison = comparison_data.nlargest(top_n, "total")
        
        # --- Plot ---
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name=role1,
            x=top_comparison.index,
            y=top_comparison[role1],
            marker_color="#1f77b4"  # Consistent blue
        ))
        fig.add_trace(go.Bar(
            name=role2,
            x=top_comparison.index,
            y=top_comparison[role2],
            marker_color="#2ca02c"  # Consistent green
        ))

        metric_label = "Job Mentions" if metric_choice == "mentions" else "Prevalence (%)"
        fig.update_layout(
            title=f"{role1} vs {role2} - {metric_label}",
            xaxis_title=column_choice.replace('_', ' ').title(),
            yaxis_title=metric_label,
            barmode="group",
            xaxis_tickangle=-45,
            showlegend=True,
            margin=dict(b=100)  # Extra bottom margin for long labels
        )

        st.plotly_chart(fig, use_container_width=True, config={"responsive": True})
        
        # --- Strategic captions ---
        if not top_comparison.empty:
            # Find biggest difference
            top_comparison['difference'] = abs(top_comparison[role1] - top_comparison[role2])
            biggest_gap = top_comparison.loc[top_comparison['difference'].idxmax()]
            
            # Find unique strengths
            role1_top = top_comparison.loc[top_comparison[role1].idxmax()]
            role2_top = top_comparison.loc[top_comparison[role2].idxmax()]
            
            if metric_choice == "mentions":
                st.caption(f"**Biggest gap**: {biggest_gap.name} → {int(biggest_gap[role1])} vs {int(biggest_gap[role2])} jobs")
                st.caption(f"**Specialties**: {role1} leads in {role1_top.name}, {role2} leads in {role2_top.name}")
            else:
                st.caption(f"**Biggest gap**: {biggest_gap.name} → {biggest_gap[role1]:.1f}% vs {biggest_gap[role2]:.1f}% of jobs")
                st.caption(f"**Core focus**: {role1} relies on {role1_top.name}, {role2} depends on {role2_top.name}")
                
    def _render_enhanced_salary_analysis(self, df: pd.DataFrame):
        """Enhanced salary analysis with multiple perspectives"""
        st.header("Salary Analysis")

        if 'salary_category' not in df.columns or df['salary_category'].isna().all():
            st.warning("⚠️ Salary data not available")
            return
        
        df['avg_salary_usd'] = df['avg_salary_usd'].replace(0,np.NAN)
        salary_data = df[df['salary_category'] != 'Unknown']
        if salary_data.empty:
            st.warning("⚠️ No valid salary data to analyze")
            return

        # --- Salary Distribution ---
        st.subheader("Overall Salary Distribution")
        fig = px.histogram(
            salary_data, 
            x="salary_category",
            title="Salary Distribution",
            labels={"salary_category": "Salary"},
            nbins=20,
            color_discrete_sequence=["#1f77b4"]
        )
        st.plotly_chart(fig, config={"responsive": True})
        
        # Calculate key statistics for caption
        salary_stats = salary_data['avg_salary_usd'].describe()
        st.caption(f"Most common salary range shown in histogram. Median: ${salary_stats['50%']:,.0f}, Range: ${salary_stats['min']:,.0f} - ${salary_stats['max']:,.0f}")

        col1, col2 = st.columns(2)

        # --- Salary by Role ---
        with col1:
            if "cleaned_title_category" in salary_data.columns:
                st.subheader("Salary by Role")
                title_salary = (
                salary_data.groupby("cleaned_title_category")["avg_salary_usd"]
                .mean()
                .reset_index()
                .sort_values("avg_salary_usd", ascending=False)
                .head(10)
            )
                fig_role = px.bar(
                    title_salary,
                    x="cleaned_title_category",
                    y="avg_salary_usd",
                    orientation="h",

                    title="Salary Range by Role",
                    labels={"cleaned_title_category": "Role", "avg_salary_usd": "Salary (avg est.)"},
                    color="cleaned_title_category",
                    color_continuous_scale="Viridis"
                )
                st.plotly_chart(fig_role, config={"responsive": True})
                
                # Role comparison caption
                role_medians = salary_data.groupby('cleaned_title_category')['avg_salary_usd'].median()
                if not role_medians.empty:
                    highest_role = role_medians.idxmax()
                    lowest_role = role_medians.idxmin()
                    st.caption(f"{highest_role} offers highest median salary, {lowest_role} shows widest compensation range across experience levels")

        # --- Salary by Seniority ---
        with col2:
            if "seniority_level" in salary_data.columns:
                st.subheader("Salary by Seniority")
                fig_seniority = px.box(
                    salary_data,
                    x="seniority_level",
                    y="avg_salary_usd",
                    title="Salary Range by Seniority",
                    labels={"seniority_level": "Seniority", "avg_salary_usd": "Salary (min est.)"},
                    color="seniority_level"
                )
                st.plotly_chart(fig_seniority, config={"responsive": True})
                
                # Seniority progression caption
                seniority_medians = salary_data.groupby('seniority_level')['avg_salary_usd'].median()
                if len(seniority_medians) > 1:
                    seniority_range = seniority_medians.max() - seniority_medians.min()
                    st.caption(f"Career progression shows {seniority_range:,.0f} salary increase from entry to senior levels")

        # --- Salary by Skill Category ---
        if "primary_job_type" in salary_data.columns:
            st.subheader("Salary by Job Type")
            
            fig_skill = px.box(
                salary_data,
                y="avg_salary_usd",
                x="primary_job_type",
                title="Top 10 Job Types by Avg Salary",
                labels={"avg_salary_usd": "Avg Salary", "primary_job_type": "Job Type"},
                color="primary_job_type", 
            ) 
            st.plotly_chart(fig_skill, config={"responsive": True})
            
            # Skill category caption
            if not salary_data.empty:
                top_skill = salary_data.iloc[0]
                salary_gap = salary_data['avg_salary_usd'].iloc[0] - salary_data['avg_salary_usd'].iloc[-1]
                st.caption(f"{top_skill['primary_job_type']} commands highest premium at ${top_skill['avg_salary_usd']:,.0f}, "
                        f"{salary_gap:,.0f} gap between highest and lowest paying skill categories")
                
    def _render_footer(self):
        """Render dashboard footer"""
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; color: #666;'>
            <p>Data Science Job Market Dashboard • Built with Streamlit • 
            <a href='https://github.com/your-repo' target='_blank'>View Source Code</a></p>
        </div>
        """, unsafe_allow_html=True)

import traceback

def main():
    """Main function to run the enhanced dashboard"""
    
    # Optional: show full Streamlit error details in dev mode
    st.set_option("client.showErrorDetails", True)
    
    try:
        # Initialize dashboard
        dashboard = EnhancedDataScienceJobsDashboard()
        
        # Render the dashboard
        dashboard.render_enhanced_dashboard()
        
    except Exception as e:
        # Show a concise error message
        st.error(f"⚠️ Error initializing dashboard: {e}")
        
        # Show full traceback for debugging
        st.subheader("Debug Info:")
        st.text(traceback.format_exc())
        
        # Friendly troubleshooting tips
        st.info("""
        **Troubleshooting tips:**
        1. Ensure the data files exist in the data/interim/ directory
        2. Run the cleaning pipeline (02_cleaning.ipynb) first
        3. Check that all required packages are installed
        """)

if __name__ == "__main__":
    main()
