"""Base class with shared functionality"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging,sys
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))
from src.core.config_skills import SKILL_CATEGORIES, SENIORITY_ORDER
import matplotlib.pyplot as plt
class BaseAnalyzer:
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        self.reports_dir = self.data_dir.parent / "reports"
        self.figures_dir = self.reports_dir / "figures"
        
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Style settings
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))
        self.skill_to_category = {
            skill.lower(): cat for cat, skills in SKILL_CATEGORIES.items() for skill in skills
        } 
        self.seniority_order = SENIORITY_ORDER
        
        # Set up logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _convert_to_list(self, x) -> list:
        """Convert string representation of list to actual list"""
        if pd.isna(x):
            return []
        if isinstance(x, list):
            return x
        try:
            if isinstance(x, str) and x.startswith("["):
                return eval(x)
            else:
                return [s.strip() for s in str(x).split(",") if s.strip()]
        except:
            return []