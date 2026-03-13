"""Main orchestrator for the data science jobs analyzer"""
from src.analysis.analyze_jobs import BaseAnalyzer  
from src.analysis.skill_analyzer import SkillAnalyzer
from src.analysis.trend_analyzer import TrendAnalyzer
from src.analysis.seniority_analyzer import SeniorityAnalyzer
from src.analysis.role_analyzer import RoleAnalyzer
from src.analysis.ecosystem_analyzer import EcosystemAnalyzer
from src.analysis.salary_analyzer import SalarySkillRegressionAnalyzer

from src.visualizations.market_analysis_plot import MarketAnalysisPlot
from src.visualizations.role_analysis_plot import RoleAnalysisPlot
from src.visualizations.trends_plot import TrendsAnalysisPlot

class DataScienceJobsAnalyzer:
    """
    Comprehensive analyzer for data science job market trends
    Modular version with separated responsibilities
    """
    
    def __init__(self, data_dir: str = "../data"): 
        self.skill_analyzer = SkillAnalyzer(data_dir)
        self.trend_analyzer = TrendAnalyzer(data_dir)
        self.seniority_analyzer = SeniorityAnalyzer(data_dir)
        self.role_analyzer = RoleAnalyzer(data_dir)
        self.ecosystem_analyzer = EcosystemAnalyzer(data_dir)
        


   
   