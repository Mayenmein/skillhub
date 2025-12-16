"""Main orchestrator for the data science jobs analyzer"""
from src.core.data_processor import DataProcessor
from src.analysis.skill_analyzer import SkillAnalyzer
from src.analysis.trend_analyzer import TrendAnalyzer
from src.analysis.seniority_analyzer import SeniorityAnalyzer
from src.analysis.role_analyzer import RoleAnalyzer
from src.analysis.ecosystem_analyzer import EcosystemAnalyzer

class DataScienceJobsAnalyzer:
    """
    Comprehensive analyzer for data science job market trends
    Modular version with separated responsibilities
    """
    
    def __init__(self, data_dir: str = "../data"):
        # Initialize component analyzers
        self.data_processor = DataProcessor(data_dir)
        self.skill_analyzer = SkillAnalyzer(data_dir)
        self.trend_analyzer = TrendAnalyzer(data_dir)
        self.seniority_analyzer = SeniorityAnalyzer(data_dir)
        self.role_analyzer = RoleAnalyzer(data_dir)
        self.ecosystem_analyzer = EcosystemAnalyzer(data_dir)
   
   