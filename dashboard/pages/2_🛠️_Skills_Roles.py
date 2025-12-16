# dashboard/pages/2_🛠️_Skills_Roles.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from app import get_dashboard_manager
from utils.dashboard_utils import render_filter_summary

def main():
    st.title("🛠️ Skills & Career Paths")
    st.markdown("""
    **Discover which skills matter for your career goals and how to develop them over time.**
    """)
    
    # Get dashboard manager
    dashboard_manager = get_dashboard_manager()
    
    # Setup sidebar filters
    filters = dashboard_manager.setup_sidebar_filters()
    
    # Apply filters
    main_df, filtered_df = dashboard_manager.apply_filters(filters)
    
    # Show filter summary
    render_filter_summary(filters, main_df, dashboard_manager.df)
    
    if filtered_df.empty:
        st.warning("No data available with current filters")
        return

    # Story-driven tabs
    tab1, tab2 = st.tabs(["🎯 Choose Your Path", "📈 Build Your Skills"])

    with tab1:
        render_career_paths(filtered_df, dashboard_manager.role_analyzer)

    with tab2:
        render_skill_development(filtered_df, dashboard_manager.role_analyzer)

def render_career_paths(pivot_df, role_analyzer):
    """Help users choose and compare career paths"""
    st.header("🎯 Choose Your Career Path")
    
    if "cleaned_title_category" not in pivot_df.columns:
        st.warning("Role data not available")
        return

    # Get available roles
    roles = pivot_df["cleaned_title_category"].dropna().unique().tolist()
    
    st.subheader("Compare Career Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        role1 = st.selectbox("First career path", roles, key="role1")
    
    with col2:
        available_roles = [r for r in roles if r != role1]
        role2 = st.selectbox("Compare with", available_roles, key="role2")

    if role1 and role2:
        # Show comparison heatmap using RoleAnalyzer
        try:
            comparison_df = role_analyzer.compare_role_skill_profiles(pivot_df, roles=[role1, role2])
            
            if not comparison_df.empty:
                # Get top differentiating skills
                comparison_df['difference'] = abs(comparison_df[role1] - comparison_df[role2])
                top_differentiating = comparison_df.nlargest(10, 'difference')
                
                fig = px.imshow(
                    top_differentiating[[role1, role2]],
                    aspect="auto",
                    title=f'Key Differences: {role1} vs {role2}',
                    labels=dict(x="Career Path", y="Key Skills", color="Importance (%)"),
                    color_continuous_scale="RdBu",
                    text_auto=".1f"
                )
                st.plotly_chart(fig, use_container_width=True)
                
                # Career choice insights
                st.subheader("💡 Which Path is Right for You?")
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Choose {role1} if you:**")
                    role1_unique = top_differentiating[top_differentiating[role1] > top_differentiating[role2]].head(3)
                    for skill in role1_unique.index:
                        st.write(f"• Want to focus on **{skill}**")
                
                with col2:
                    st.write(f"**Choose {role2} if you:**")
                    role2_unique = top_differentiating[top_differentiating[role2] > top_differentiating[role1]].head(3)
                    for skill in role2_unique.index:
                        st.write(f"• Want to focus on **{skill}**")
                        
        except Exception as e:
            st.error(f"Unable to compare roles: {e}")

def render_skill_development(pivot_df, role_analyzer):
    """Show how to develop skills for career growth"""
    st.header("📈 Build Your Skills")
    
    # First, get top skills to show users what's available
    with st.spinner("Loading available skills..."):
        try:
            # Get top skills from the data to populate the selection
            skill_counts = pivot_df['skill'].value_counts()
            available_skills = skill_counts.head(50).index.tolist()  # Top 50 skills
            
            if not available_skills:
                st.info("No skills data available")
                return
                
        except Exception as e:
            st.error(f"Unable to load skills data: {e}")
            return

    st.subheader("Your Skill Development Roadmap")
    
    # Let user choose skills they want to develop
    selected_skills = st.multiselect(
        "Select skills to track your development:",
        options=available_skills,
        default=available_skills[:3],
        max_selections=6,
        help="Choose up to 6 skills to see how they progress through career levels"
    )
    
    if not selected_skills:
        st.info("Select skills to see development path")
        return
    
    # Get progression data for selected skills using RoleAnalyzer
    with st.spinner("Analyzing skill progression..."):
        try:
            progression_df = role_analyzer.analyze_skill_progression_data(pivot_df, selected_skills)
            
            if progression_df.empty:
                st.info("No progression data available for selected skills")
                return
                
        except Exception as e:
            st.error(f"Unable to analyze skill progression: {e}")
            return
    
    # Define seniority order
    seniority_order = ['Intern','Entry-level', 'Junior', 'Mid-level', 'Senior', 'Lead', 'Executive','Manager']
    available_levels = [level for level in seniority_order if level in progression_df['seniority_level'].unique()]
    
    if len(available_levels) < 2:
        st.warning("Need more seniority levels for development path analysis")
        return
    
    # Create development roadmap
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**📊 Your Development Progress**")
        
        fig = px.line(
            progression_df,
            x='seniority_level',
            y='prevalence',
            color='skill',
            markers=True,
            title='Skill Importance at Each Career Stage',
            labels={'seniority_level': 'Career Stage', 'prevalence': 'Importance (% of Jobs)'},
            category_orders={'seniority_level': available_levels}
        )
        fig.update_layout(height=400)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.write("**🎯 Learning Recommendations**")
        
        for skill in selected_skills:
            skill_data = progression_df[progression_df['skill'] == skill].sort_values('seniority_level')
            if len(skill_data) > 1:
                entry_level = skill_data[skill_data['seniority_level'] == available_levels[0]]['prevalence'].iloc[0] if available_levels[0] in skill_data['seniority_level'].values else 0
                senior_level = skill_data[skill_data['seniority_level'] == available_levels[-1]]['prevalence'].iloc[0] if available_levels[-1] in skill_data['seniority_level'].values else 0
                
                if entry_level > 0 and senior_level > 0:
                    if senior_level > entry_level * 1.5:
                        growth_pct = ((senior_level / entry_level) - 1) * 100
                        st.success(f"**{skill}**: Keep developing - grows {growth_pct:.0f}% by senior level")
                    elif senior_level < entry_level * 0.8:
                        st.info(f"**{skill}**: Foundation skill - most important early in career")
                    else:
                        st.write(f"**{skill}**: Consistent importance throughout career")
                else:
                    st.write(f"**{skill}**: Data available for some career stages")
    
    # Action plan
    st.subheader("🚀 Your Action Plan")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Immediate Focus (Next 6 months):**")
        early_skills = []
        for skill in selected_skills:
            skill_data = progression_df[progression_df['skill'] == skill]
            if not skill_data.empty and available_levels[0] in skill_data['seniority_level'].values:
                entry_importance = skill_data[skill_data['seniority_level'] == available_levels[0]]['prevalence'].iloc[0]
                early_skills.append((skill, entry_importance))
        
        # Sort by importance and show top 3
        for skill, importance in sorted(early_skills, key=lambda x: x[1], reverse=True)[:3]:
            st.write(f"• **{skill}** ({importance:.1f}% of entry-level jobs)")
    
    with col2:
        st.write("**Long-term Development (2+ years):**")
        growing_skills = []
        for skill in selected_skills:
            skill_data = progression_df[progression_df['skill'] == skill].sort_values('seniority_level')
            if len(skill_data) > 1:
                if available_levels[0] in skill_data['seniority_level'].values and available_levels[-1] in skill_data['seniority_level'].values:
                    growth = skill_data[skill_data['seniority_level'] == available_levels[-1]]['prevalence'].iloc[0] - skill_data[skill_data['seniority_level'] == available_levels[0]]['prevalence'].iloc[0]
                    if growth > 5:  # Significant growth
                        growing_skills.append((skill, growth))
        
        for skill, growth in sorted(growing_skills, key=lambda x: x[1], reverse=True)[:3]:
            st.write(f"• **{skill}** (+{growth:.1f}% growth to senior level)")

if __name__ == "__main__":
    main()