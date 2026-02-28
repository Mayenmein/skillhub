# SkillHub: Data Science Job Market Analytics Dashboard

<!-- Add your main screenshot/dashboard preview here -->
<!-- ![Dashboard Preview](images/dashboard-preview.png) -->

![Version](https://img.shields.io/badge/version-1.0.0-blue)
![Python](https://img.shields.io/badge/python-3.9+-green)
![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-red)
![License](https://img.shields.io/badge/license-MIT-yellow)

**Transform raw job market data into actionable career intelligence for data professionals.**

## Overview

SkillHub analyzes real-time data science job postings to identify skill demand trends, helping professionals make data-driven career decisions. From scraping job boards to interactive visualizations, we bridge the gap between market needs and career development.

### The Challenge
- Data science skills evolve faster than traditional career advice
- Professionals waste time learning obsolete or low-demand technologies
- Hiring decisions often lack real-time market intelligence
- Educational programs struggle to keep curricula industry-relevant

### Our Solution
An end-to-end analytics platform that transforms job posting data into clear, actionable insights through an intuitive dashboard, enabling smarter career decisions across the data science ecosystem.

## Who It's For

| Role | Primary Use Case |
|------|------------------|
| **Students & Graduates** | Identify high-demand skills for job readiness |
| **Data Professionals** | Strategic upskilling based on market trends |
| **HR & Recruiters** | Data-driven talent acquisition and planning |
| **Educators** | Curriculum aligned with industry demands |

## Key Features

### Real-Time Market Intelligence
- **Live skill demand tracking** across 100+ data science technologies
- **Growth/decline classification** (Emerging, Growing, Stable, Declining)
- **Market share analysis** by industry and experience level

### Career Decision Support
- **Personalized skill recommendations** based on career goals
- **Learning roadmap generator** with estimated timelines
- **Salary correlation insights** for skill combinations

### Interactive Analytics
- **Comparative skill analysis** with side-by-side visualizations
- **Historical trend tracking** with predictive indicators
- **Industry-specific heat maps** and demand patterns

### End-to-End Pipeline
- **Automated data collection** from major job platforms
- **Intelligent skill standardization** and categorization
- **Real-time dashboard updates** with fresh insights

## How It Works

<!-- <img src="docs/images/flow_chart.png" alt="System Architecture" width="500"> -->

### 1. Data Collection
Scrape and aggregate job postings from LinkedIn, Indeed, and specialized data science boards, capturing skills, requirements, and compensation data.

### 2. Processing & Analysis
- Clean and standardize 500+ data science skills using pandas
- Create skill pivot tables for multi-dimensional analysis
- Calculate demand growth rates and market positions
- Identify emerging trends and declining technologies

### 3. Visualization & Insights
- Interactive Streamlit dashboard with real-time updates
- Comparative analysis tools and trend forecasting
- Downloadable reports and personalized recommendations

## Methodology & Key Findings

### Skill Trend Analysis
Skills are classified using a proprietary **volume-adjusted momentum score** that combines:
- Recent growth momentum (3-month rolling average)
- Baseline mention volume for statistical significance
- Non-linear trend detection for emerging vs. peaking classification

### Salary-Skill Analysis ⚠️
Our regression analysis reveals that **skills alone explain approximately 6% of salary variation** (R² = 0.062). This honest finding underscores an important reality:

> *While technical skills matter, factors like years of experience, company size, industry, location, and individual negotiation play substantially larger roles in determining actual compensation.*

**What this means for you:**
- Use our skill premium rankings for **relative comparisons**, not absolute predictions
- Focus on skill combinations that show positive synergy effects
- Consider skills as one piece of your career strategy, alongside experience-building and networking

## Quick Start

### Prerequisites
- Python 3.9+
- pip package manager

```bash
# Clone repository
git clone https://github.com/Mayenmein/skillhub.git
cd skillhub

# Install dependencies
pip install -r requirements.txt

# Launch dashboard
streamlit run app.py
```

## Basic Usage
Explore Trends: Navigate to the Trends page to see skill demand evolution

Compare Skills: Use the comparison tool for side-by-side analysis

Get Recommendations: Input your current skills for personalized suggestions

Export Insights: Download reports for offline review

##  Contributing
We welcome contributions! Please see our Contributing Guidelines for details.

1. Fork the repository

2. Create a feature branch (git checkout -b feature/amazing-feature)

3. Commit your changes (git commit -m 'Add amazing feature')

4. Push to the branch (git push origin feature/amazing-feature)

Open a Pull Request

##  License
Distributed under the MIT License. See LICENSE for more information.

##  Contact & Support
- Project Link: https://github.com/Mayenmein/skillhub

- Issue Tracker: https://github.com/Mayenmein/skillhub/issues



##  Acknowledgments
- Open-source data science community

- Job platform APIs and data providers

- Streamlit and Plotly development teams

- Contributors and early adopters