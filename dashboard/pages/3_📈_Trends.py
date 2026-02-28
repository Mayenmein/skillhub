# dashboard/pages/3_📈_Trends.py

import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
import numpy as np
from app import get_dashboard_manager
from utils.dashboard_utils import render_filter_summary

# Cache the pivot table and basic calculations
@st.cache_data(ttl=3600)  # Cache for 1 hour
def get_monthly_totals(_pivot_df):
    """Get monthly totals - cached to avoid recomputation"""
    months = sorted(_pivot_df["published"].unique())
    monthly_totals = _pivot_df.groupby("published")["job_ids"].apply(
        lambda x: len(set([i for sublist in x for i in sublist]))
    ).reindex(months).values
    return months, monthly_totals

@st.cache_data(ttl=3600)
def get_skill_month_matrix(_pivot_df):
    """Get skill-month matrix - cached to avoid recomputation"""
    months = sorted(_pivot_df["published"].unique())
    skill_month_matrix = _pivot_df.pivot_table(
        index="skill", columns="published", values="mentions", aggfunc="sum", fill_value=0
    ).reindex(columns=months)
    return months, skill_month_matrix

def get_smoothed_trend_for_skills(pivot_df, skill_names, smoothing_alpha=0.7):
    """Get smoothed trend data ONLY for selected skills (lazy evaluation)"""
    if not skill_names:
        return {}
    
    # Get cached monthly data
    months, monthly_totals = get_monthly_totals(pivot_df)
    
    # Get skill-month matrix for selected skills only
    skill_month_matrix = pivot_df[pivot_df["skill"].isin(skill_names)].pivot_table(
        index="skill", columns="published", values="mentions", aggfunc="sum", fill_value=0
    ).reindex(columns=months)
    
    # Add missing skills (if any) with zeros
    missing_skills = set(skill_names) - set(skill_month_matrix.index)
    if missing_skills:
        zero_data = pd.DataFrame(0, index=list(missing_skills), columns=months)
        skill_month_matrix = pd.concat([skill_month_matrix, zero_data])
    
    smoothed_trends = {}
    
    for skill in skill_names:
        if skill in skill_month_matrix.index:
            raw_values = skill_month_matrix.loc[skill].values.astype(float)
            
            # Apply exponential smoothing
            smoothed = np.zeros_like(raw_values)
            if len(raw_values) > 0:
                smoothed[0] = raw_values[0]
                for i in range(1, len(raw_values)):
                    smoothed[i] = smoothing_alpha * raw_values[i] + (1 - smoothing_alpha) * smoothed[i-1]
            
            # Calculate market share trend
            raw_market_share = (raw_values / monthly_totals) * 100
            smoothed_share = np.zeros_like(raw_market_share)
            if len(raw_market_share) > 0:
                smoothed_share[0] = raw_market_share[0] if not np.isnan(raw_market_share[0]) else 0
                for i in range(1, len(raw_market_share)):
                    if not np.isnan(raw_market_share[i]):
                        smoothed_share[i] = smoothing_alpha * raw_market_share[i] + (1 - smoothing_alpha) * smoothed_share[i-1]
                    else:
                        smoothed_share[i] = smoothed_share[i-1]
            
            smoothed_trends[skill] = {
                'raw_values': raw_values,
                'smoothed_values': smoothed,
                'raw_market_share': raw_market_share,
                'smoothed_market_share': smoothed_share,
                'dates': months
            }
    
    return smoothed_trends

def main():
    st.title("📈 Skill Trends Analysis")
    st.markdown("""
    **Analyze skill demand trends and growth patterns in the job market.**
    Identify emerging opportunities and track skill evolution over time.
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

    # Run trend analysis using TrendAnalyzer
    with st.spinner("Analyzing skill trends..."):
        try:
            trends_df = dashboard_manager.trend_analyzer.analyze_skill_trends_enhanced(
                filtered_df, 
                min_prevalence=2.5, 
                min_avg_mentions=10,
                min_months=9,
                smoothing_alpha=0.7
            )
            
            if trends_df.empty:
                st.info("Not enough data for trend analysis. Try broader filters.")
                return
                
        except Exception as e:
            st.error(f"Unable to analyze trends at the moment: {e}")
            return
    
    # Update boolean flags for new 5-category system
    trends_df = update_trend_flags(trends_df)
    
    # Two simple tabs
    tab1, tab2 = st.tabs(["📊 Growth Analysis", "🎯 Market Position"])
    
    with tab1:
        render_growth_analysis(trends_df, filtered_df)
    
    with tab2:
        render_market_position(trends_df)

def update_trend_flags(trends_df):
    """Update boolean flags for the new 5-category system"""
    # Create new boolean columns based on 5 categories
    trends_df["is_hot_growing"] = trends_df["trend_category"] == "Hot & Growing"
    trends_df["is_stable_in_demand"] = trends_df["trend_category"] == "Stable & In-Demand"
    trends_df["is_niche_specialized"] = trends_df["trend_category"] == "Niche & Specialized"
    trends_df["is_declining_risky"] = trends_df["trend_category"] == "Declining & Risky"
    trends_df["is_emerging_new"] = trends_df["trend_category"] == "Emerging & New"
    
    # Keep legacy columns for compatibility
    trends_df["is_emerging"] = trends_df["is_emerging_new"] | trends_df["is_hot_growing"]
    trends_df["is_growing"] = trends_df["is_hot_growing"] | trends_df["is_stable_in_demand"]
    trends_df["is_stable"] = trends_df["is_stable_in_demand"] | trends_df["is_niche_specialized"]
    trends_df["is_declining"] = trends_df["is_declining_risky"]
    trends_df["is_special"] = trends_df["is_niche_specialized"]
    
    return trends_df

def render_growth_analysis(trends_df, pivot_df):
    """Show skill trends by growth category with combined plots"""
    st.header("📊 Skill Growth Analysis")
    
    # Convert dates for consistent formatting
    pivot_df["published"] = pd.to_datetime(pivot_df["published"], format="%Y.0_%m.0")
    
    # Category selection with new 5-category system
    st.subheader("Choose Growth Category to Explore")
    
    category = st.selectbox(
        "Select trend category:",
        [
            "🚀 Hot & Growing", 
            "💎 Stable & In-Demand", 
            "🎯 Niche & Specialized",
            "⚠️ Declining & Risky",
            "🌱 Emerging & New",
            "⭐ All Trending Skills"
        ]
    )
    
    # Filter skills based on selection
    if category == "🚀 Hot & Growing":
        filtered_skills = trends_df[trends_df['is_hot_growing']]
        category_desc = "High demand skills showing rapid growth - Great for career advancement"
    elif category == "💎 Stable & In-Demand":
        filtered_skills = trends_df[trends_df['is_stable_in_demand']]
        category_desc = "Established skills with steady demand - Foundation skills to master"
    elif category == "🎯 Niche & Specialized":
        filtered_skills = trends_df[trends_df['is_niche_specialized']]
        category_desc = "Specialized skills for specific roles - Lower competition but fewer opportunities"
    elif category == "⚠️ Declining & Risky":
        filtered_skills = trends_df[trends_df['is_declining_risky']]
        category_desc = "Skills losing demand - Consider transitioning away"
    elif category == "🌱 Emerging & New":
        filtered_skills = trends_df[trends_df['is_emerging_new']]
        category_desc = "New technologies with potential - Monitor before investing heavily"
    else:  # All Trending Skills
        filtered_skills = trends_df[trends_df['is_hot_growing'] | trends_df['is_emerging_new']]
        category_desc = "All skills showing positive growth trends"
    
    if filtered_skills.empty:
        st.info(f"No skills found in category: {category}")
        return
    
    st.write(f"**{category}** - {category_desc}")
    
    # Add student advice based on category
    advice = {
        "🚀 Hot & Growing": "✅ **Learn Now** - High job openings, increasing salaries",
        "💎 Stable & In-Demand": "✅ **Master** - Foundation skills, always needed",
        "🎯 Niche & Specialized": "🤔 **Specialize if interested** - Fewer jobs but less competition",
        "⚠️ Declining & Risky": "❌ **Avoid** - Jobs disappearing, skills becoming obsolete",
        "🌱 Emerging & New": "🔍 **Explore & Monitor** - Learn basics, watch adoption"
    }
    
    if category in advice:
        st.info(advice[category])
    
    # Skill selector - only show skills from chosen category
    available_skills = filtered_skills['skill'].tolist()
    
    if not available_skills:
        st.info("No skills available in selected category")
        return
    
    # Auto-select top 3 skills by growth rate or prevalence
    if category in ["🚀 Hot & Growing", "🌱 Emerging & New"]:
        top_skills = filtered_skills.nlargest(3, 'CAGR_pct')['skill'].tolist()
    elif category == "⚠️ Declining & Risky":
        top_skills = filtered_skills.nsmallest(3, 'CAGR_pct')['skill'].tolist()
    else:
        top_skills = filtered_skills.nlargest(3, 'current_prevalence')['skill'].tolist()
    
    selected_skills = st.multiselect(
        f"Select skills to visualize ({len(available_skills)} available):",
        options=available_skills,
        default=top_skills,
        help="Choose 2-4 skills for clear comparison"
    )[:4]  # Limit to 4 skills
    
    if not selected_skills:
        st.info("Please select skills to visualize")
        return
    
    # Create skill info dictionary for O(1) lookups
    skill_info_dict = {row['skill']: row for _, row in filtered_skills.iterrows()}
    
    # Generate combined plots for selected skills
    st.subheader("📈 Skill Trend Visualization")
    st.markdown("""
    **Visualization Guide:**
    - **Raw Data (Bars)**: Actual monthly job postings
    - **Trend Line (Solid)**: Smoothed trend showing the underlying pattern
    - **Market Share (Dashed)**: Percentage of all job postings requiring this skill
    """)
    
    # Load smoothed data ONLY for selected skills (lazy evaluation)
    with st.spinner("Generating trend visualizations..."):
        smoothed_data = get_smoothed_trend_for_skills(pivot_df, selected_skills, smoothing_alpha=0.7)
    
    # Create combined plots for each selected skill
    for skill in selected_skills:
        if skill not in smoothed_data or skill not in skill_info_dict:
            st.warning(f"Data not available for skill: {skill}")
            continue
        
        # Get trend data
        skill_trends = smoothed_data[skill]
        dates = skill_trends['dates']
        raw_values = skill_trends['raw_values']
        smoothed_values = skill_trends['smoothed_values']
        smoothed_market_share = skill_trends['smoothed_market_share']
        
        # Get trend info from dictionary (O(1) lookup)
        skill_info = skill_info_dict[skill]
        
        # Create combined plot with trend lines
        fig = go.Figure()
        
        # Raw job mentions (bars)
        fig.add_trace(go.Bar(
            x=dates,
            y=raw_values,
            name="Raw Job Postings",
            marker_color="lightgray",
            opacity=0.5,
            yaxis="y"
        ))
        
        # Smoothed trend line
        fig.add_trace(go.Scatter(
            x=dates,
            y=smoothed_values,
            name="Smoothed Trend",
            line=dict(color="#1f77b4", width=4),
            mode='lines',
            yaxis="y"
        ))
        
        # Market share trend (secondary axis)
        fig.add_trace(go.Scatter(
            x=dates,
            y=smoothed_market_share,
            name="Market Share %",
            line=dict(color="#ff7f0e", width=3, dash='dash'),
            yaxis="y2"
        ))
        
        # Trend color based on category
        trend_color = {
            "Hot & Growing": "#2ca02c",
            "Stable & In-Demand": "#1f77b4",
            "Niche & Specialized": "#9467bd",
            "Declining & Risky": "#d62728",
            "Emerging & New": "#ff7f0e"
        }.get(skill_info['trend_category'], "#1f77b4")
        
        # Trend direction indicator
        trend_direction = "↗️" if skill_info['CAGR_pct'] > 0 else "↘️" if skill_info['CAGR_pct'] < 0 else "➡️"
        
        fig.update_layout(
            title=f"{trend_direction} {skill} | {skill_info['trend_category']} | Growth: {skill_info['CAGR_pct']:+.1f}%",
            title_font_color=trend_color,
            xaxis=dict(
                title="Time", 
                tickformat="%b %Y",
                tickangle=45
            ),
            yaxis=dict(
                title="Number of Job Postings",
                side="left",
                title_font=dict(color="#1f77b4"),
                tickfont=dict(color="#1f77b4"),
                gridcolor='lightgray',
                zerolinecolor='lightgray'
            ),
            yaxis2=dict(
                title="Market Share (%)",
                side="right",
                overlaying="y",
                title_font=dict(color="#ff7f0e"),
                tickfont=dict(color="#ff7f0e"),
                gridcolor='rgba(255, 127, 14, 0.1)',
                zerolinecolor='lightgray'
            ),
            legend=dict(
                x=1.05,
                y=1,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='lightgray',
                borderwidth=1
            ),
            hovermode="x unified",
            height=450,
            plot_bgcolor='rgba(240, 240, 240, 0.1)',
            margin=dict(r=150)
        )
        
        # Add trend annotations
        if skill_info['CAGR_pct'] < -10:
            fig.add_annotation(
                x=dates[-1],
                y=smoothed_values[-1],
                text="📉 Steady Decline",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowcolor="#d62728",
                font=dict(color="#d62728", size=12)
            )
        elif skill_info['CAGR_pct'] > 15:
            fig.add_annotation(
                x=dates[-1],
                y=smoothed_values[-1],
                text="🚀 Rapid Growth",
                showarrow=True,
                arrowhead=1,
                arrowsize=1,
                arrowcolor="#2ca02c",
                font=dict(color="#2ca02c", size=12)
            )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Quick stats for this skill
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            current_raw = int(raw_values[-1]) if len(raw_values) > 0 else 0
            prev_raw = int(raw_values[-2]) if len(raw_values) > 1 else current_raw
            delta_raw = current_raw - prev_raw
            st.metric("Current Jobs", current_raw, delta=f"{delta_raw:+d}")
        
        with col2:
            current_share = float(smoothed_market_share[-1]) if len(smoothed_market_share) > 0 else 0
            prev_share = float(smoothed_market_share[-2]) if len(smoothed_market_share) > 1 else current_share
            delta_share = current_share - prev_share
            st.metric("Market Share", f"{current_share:.1f}%", delta=f"{delta_share:+.1f}%")
        
        with col3:
            st.metric("Growth Rate", f"{skill_info['CAGR_pct']:+.1f}%")
        
        with col4:
            st.metric("Trend Slope", f"{skill_info['trend_slope']:+.3f}")
        
        with col5:
            color_html = {
                "Hot & Growing": "🟢",
                "Stable & In-Demand": "🔵",
                "Niche & Specialized": "🟣",
                "Declining & Risky": "🔴",
                "Emerging & New": "🟠"
            }.get(skill_info['trend_category'], "⚪")
            st.markdown(f"{color_html} **{skill_info['trend_category']}**")
        
        # Simple Trend Explanation for Non-Technical Users
        with st.expander(f"📊 What This Means for {skill}"):
            
            if skill_info['trend_category'] == "Hot & Growing":
                st.success(f"""
                ## 🔥 **Why {skill} is HOT & GROWING:**
                
                **In Simple Terms:**
                - Companies are hiring **MORE** people with {skill} every year
                - It's already in **HIGH DEMAND** right now
                - Job opportunities are **INCREASING FAST**
                
                **By the Numbers:**
                - **Growing at:** {skill_info['CAGR_pct']:+.0f}% per year
                - **Current Jobs:** {int(skill_info['avg_mentions_per_month'])} postings/month
                - **Market Share:** {skill_info['current_prevalence']:.1f}% of all tech jobs
                
                **💡 For Your Career:**
                **BEST TIME TO LEARN!** This skill will give you:
                - More job opportunities
                - Better salary potential  
                - Career growth options
                
                **✅ Action Plan:**
                1. Start learning {skill} this month
                2. Practice 1-2 hours daily
                3. Build a small project
                4. Apply for jobs in 3-6 months
                """)
                
            elif skill_info['trend_category'] == "Stable & In-Demand":
                st.info(f"""
                ## 🛡️ **Why {skill} is STABLE & IN-DEMAND:**
                
                **In Simple Terms:**
                - Companies **ALWAYS** need people with {skill}
                - Job demand is **STEADY AND RELIABLE**
                - It's a **FOUNDATION** skill for many careers
                
                **By the Numbers:**
                - **Growing at:** {skill_info['CAGR_pct']:+.0f}% per year
                - **Current Jobs:** {int(skill_info['avg_mentions_per_month'])} postings/month
                - **Market Share:** {skill_info['current_prevalence']:.1f}% of all tech jobs
                
                **💡 For Your Career:**
                **SAFE AND SMART CHOICE!** This skill will give you:
                - Job security
                - Long-term career foundation
                - Wide range of opportunities
                
                **✅ Action Plan:**
                1. Master {skill} thoroughly
                2. Get certified if possible
                3. Combine with other skills
                4. You'll always be employable
                """)
                
            elif skill_info['trend_category'] == "Declining & Risky":
                st.error(f"""
                ## ⚠️ **Why {skill} is DECLINING & RISKY:**
                
                **In Simple Terms:**
                - Companies are hiring **FEWER** people with {skill}
                - Job opportunities are **DECREASING**
                - It's being **REPLACED** by newer technologies
                
                **By the Numbers:**
                - **Declining at:** {skill_info['CAGR_pct']:+.0f}% per year
                - **Current Jobs:** {int(skill_info['avg_mentions_per_month'])} postings/month
                - **Market Share:** {skill_info['current_prevalence']:.1f}% of all tech jobs
                
                **💡 For Your Career:**
                **BE CAREFUL!** Learning this skill might mean:
                - Fewer job openings in the future
                - Lower salary growth
                - Need to retrain later
                
                **🔄 Better Alternatives:**
                Consider learning these instead:
                """)
                
                # Find growing alternatives
                similar_growing = trends_df[
                    (trends_df['skill'] != skill) & 
                    (trends_df['is_hot_growing'] | trends_df['is_stable_in_demand'])
                ].head(3)
                
                if not similar_growing.empty:
                    for _, alt_skill in similar_growing.iterrows():
                        growth_emoji = "🔥" if alt_skill['CAGR_pct'] > 10 else "📈"
                        st.write(f"{growth_emoji} **{alt_skill['skill']}** (Growing at {alt_skill['CAGR_pct']:+.0f}% per year)")
                
                st.write(f"""
                **✅ Action Plan:**
                1. Don't invest too much time in {skill}
                2. If you know it already, maintain it but don't specialize
                3. Learn one of the alternatives above
                4. Update your resume with newer skills
                """)
                
            elif skill_info['trend_category'] == "Emerging & New":
                st.warning(f"""
                ## 🌱 **Why {skill} is EMERGING & NEW:**
                
                **In Simple Terms:**
                - This is a **NEW TECHNOLOGY**
                - Job demand is **STARTING TO GROW**
                - Could be **BIG IN THE FUTURE**, but not certain yet
                
                **By the Numbers:**
                - **Growing at:** {skill_info['CAGR_pct']:+.0f}% per year (fast!)
                - **Current Jobs:** {int(skill_info['avg_mentions_per_month'])} postings/month
                - **Market Share:** {skill_info['current_prevalence']:.1f}% of all tech jobs (small but growing)
                
                **💡 For Your Career:**
                **EXPLORE & MONITOR!** This skill could be:
                - A future career advantage if it takes off
                - A waste of time if it doesn't get popular
                - Good for early adopters who like new tech
                
                **✅ Action Plan:**
                1. Learn the basics of {skill}
                2. Build a small project to understand it
                3. Watch job postings for 3-6 months
                4. If demand keeps growing, learn more deeply
                """)
                
            elif skill_info['trend_category'] == "Niche & Specialized":
                st.info(f"""
                ## 🎯 **Why {skill} is NICHE & SPECIALIZED:**
                
                **In Simple Terms:**
                - Used in **SPECIFIC JOB ROLES** only
                - **FEWER** total job openings
                - **LESS COMPETITION** for those jobs
                
                **By the Numbers:**
                - **Growing at:** {skill_info['CAGR_pct']:+.0f}% per year
                - **Current Jobs:** {int(skill_info['avg_mentions_per_month'])} postings/month
                - **Market Share:** {skill_info['current_prevalence']:.1f}% of all tech jobs (specialized)
                
                **💡 For Your Career:**
                **SPECIALIZE IF PASSIONATE!** This skill is good for:
                - People who love this specific area
                - Those wanting to be experts in a narrow field
                - Lower competition but fewer job options
                
                **✅ Action Plan:**
                1. Only learn {skill} if you truly enjoy it
                2. Become an expert - there's less competition
                3. Combine with broader skills for more options
                4. Look for specialized companies that need it
                """)
            
            # Simple visualization explanation
            st.markdown("---")
            st.subheader("📈 How to Read the Chart Above")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("""
                **Gray Bars = Monthly Job Postings**
                - Taller bars = More jobs that month
                - Shorter bars = Fewer jobs
                - Shows real hiring activity
                """)
                
            with col2:
                st.markdown("""
                **Blue Line = Job Trend**
                - Going UP = More jobs over time ✅
                - Going DOWN = Fewer jobs over time ⚠️
                - Flat = Steady job market 📊
                """)
            
            # Add simple trend indicators
            if skill_info['CAGR_pct'] > 15:
                st.success(f"**📊 Trend Summary:** {skill} is growing VERY FAST ({skill_info['CAGR_pct']:+.0f}% per year)")
            elif skill_info['CAGR_pct'] > 5:
                st.info(f"**📊 Trend Summary:** {skill} is growing steadily ({skill_info['CAGR_pct']:+.0f}% per year)")
            elif skill_info['CAGR_pct'] < -5:
                st.error(f"**📊 Trend Summary:** {skill} is declining ({skill_info['CAGR_pct']:+.0f}% per year)")
            else:
                st.info(f"**📊 Trend Summary:** {skill} is stable (growth near 0%)")
        
        # Add student advice for this specific skill
        advice_dict = {
            "Hot & Growing": f"**✅ Priority Learning**: {skill} is in high demand with strong growth. Invest time to learn this skill now.",
            "Stable & In-Demand": f"**✅ Foundation Skill**: {skill} is widely used and stable. Essential for many roles.",
            "Niche & Specialized": f"**🎯 Specialized Opportunity**: {skill} is used in specific roles. Consider if you want to specialize.",
            "Declining & Risky": f"**⚠️ Caution**: {skill} is losing demand. Consider upskilling to newer alternatives.",
            "Emerging & New": f"**🌱 Watch & Learn**: {skill} is new and growing. Learn basics but wait for wider adoption."
        }
        
        st.info(advice_dict.get(skill_info['trend_category'], ""))
        
        st.markdown("---")

def render_market_position(trends_df):
    """Show market share analysis with boxplots and growth visualization"""
    st.header("🎯 Skill Market Position")
    
    # Create color mapping for new categories
    category_colors = {
        "Hot & Growing": "#2ca02c",
        "Stable & In-Demand": "#1f77b4",
        "Niche & Specialized": "#9467bd",
        "Declining & Risky": "#d62728",
        "Emerging & New": "#ff7f0e"
    }
    
    # Market Share Distribution by Category
    st.subheader("📊 Market Share Distribution by Category")
    
    # Order categories by typical career progression
    category_order = ["Hot & Growing", "Stable & In-Demand", "Niche & Specialized", "Emerging & New", "Declining & Risky"]
    
    # Create boxplot
    fig_box = px.box(
        trends_df,
        x='trend_category',
        y='current_prevalence',
        color='trend_category',
        color_discrete_map=category_colors,
        category_orders={"trend_category": category_order},
        title='Market Share Distribution Across Skill Categories',
        labels={
            'current_prevalence': 'Market Share (%)',
            'trend_category': 'Skill Category'
        },
        points="all",
        hover_data=['skill']
    )
    
    fig_box.update_layout(
        xaxis_tickangle=0,
        showlegend=False,
        height=500
    )
    
    st.plotly_chart(fig_box, use_container_width=True)
    
    # Key insights from boxplot
    st.subheader("💡 Key Insights")
    
    # Calculate median market share by category
    median_shares = trends_df.groupby('trend_category')['current_prevalence'].median()
    median_shares = median_shares.reindex(category_order).dropna()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Median Market Share by Category:**")
        for category in category_order:
            if category in median_shares:
                share = median_shares[category]
                color_icon = {
                    "Hot & Growing": "🟢",
                    "Stable & In-Demand": "🔵", 
                    "Niche & Specialized": "🟣",
                    "Declining & Risky": "🔴",
                    "Emerging & New": "🟠"
                }.get(category, "⚪")
                st.write(f"{color_icon} **{category}**: {share:.1f}%")
    
    with col2:
        st.write("**Student Career Advice:**")
        advice = {
            "Hot & Growing": "Learn for job growth",
            "Stable & In-Demand": "Master for job security",
            "Niche & Specialized": "Specialize for unique roles",
            "Declining & Risky": "Avoid or transition away",
            "Emerging & New": "Watch and explore basics"
        }
        for category in category_order[:3]:
            if category in advice:
                st.write(f"• **{category}**: {advice[category]}")
    
    # Growth vs Market Share Scatter Plot
    st.subheader("🚀 Growth Rate vs Market Share")
    
    fig_scatter = px.scatter(
        trends_df,
        x='current_prevalence',
        y='CAGR_pct',
        color='trend_category',
        color_discrete_map=category_colors,
        size='current_prevalence',
        hover_name='skill',
        title='Skill Positioning: Current Market Share vs Annual Growth',
        labels={
            'current_prevalence': 'Current Market Share (%)',
            'CAGR_pct': 'Annual Growth Rate (%)'
        },
        size_max=20
    )
    
    fig_scatter.add_hline(y=0, line_dash="dash", line_color="red", opacity=0.7)
    fig_scatter.add_vline(x=trends_df['current_prevalence'].median(), line_dash="dash", line_color="gray", opacity=0.7)
    
    fig_scatter.add_annotation(x=75, y=25, text="🚀 High Growth Leaders", showarrow=False, font=dict(size=10))
    fig_scatter.add_annotation(x=75, y=-25, text="📉 Declining Giants", showarrow=False, font=dict(size=10))
    fig_scatter.add_annotation(x=25, y=25, text="🌱 Emerging Stars", showarrow=False, font=dict(size=10))
    fig_scatter.add_annotation(x=25, y=-25, text="🎯 Niche Skills", showarrow=False, font=dict(size=10))
    
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Simple interpretation
    st.subheader("🎯 How to Use This Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("""
        **For Job Seekers:**
        - **Top-right**: High-demand skills to learn
        - **Top-left**: New opportunities to explore  
        - **Bottom-right**: Skills to maintain but not learn new
        - **Bottom-left**: Skills to phase out
        """)
    
    with col2:
        st.info("""
        **For Students:**
        1. Start with **Stable & In-Demand** (foundation)
        2. Add **Hot & Growing** (career boost)
        3. Explore **Emerging & New** (future potential)
        4. Avoid **Declining & Risky** unless needed
        """)
    
    # Top performers section
    st.subheader("🏆 Top Skills by Category")
    
    categories_to_show = ["Hot & Growing", "Stable & In-Demand", "Emerging & New"]
    
    for category in categories_to_show:
        if category in trends_df['trend_category'].unique():
            category_skills = trends_df[trends_df['trend_category'] == category]
            
            if not category_skills.empty:
                if category == "Hot & Growing":
                    top_skills = category_skills.nlargest(3, 'CAGR_pct')
                    metric = "Growth"
                    value = lambda x: f"{x['CAGR_pct']:+.1f}%"
                elif category == "Stable & In-Demand":
                    top_skills = category_skills.nlargest(3, 'current_prevalence')
                    metric = "Market Share"
                    value = lambda x: f"{x['current_prevalence']:.1f}%"
                else:
                    top_skills = category_skills.nlargest(3, 'CAGR_pct')
                    metric = "Growth"
                    value = lambda x: f"{x['CAGR_pct']:+.1f}%"
                
                st.write(f"**{category}:**")
                for _, skill in top_skills.iterrows():
                    st.write(f"• **{skill['skill']}** - {metric}: {value(skill)}")
                st.write("")

if __name__ == "__main__":
    main()