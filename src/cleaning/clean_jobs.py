# src/processing/clean_skills.py
import pandas as pd
import numpy as np
import re
 
import logging 

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')
               
def clean_job_type(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and categorize job types into separate columns.
    Optimized version - maintains ALL original patterns.
    """
    df_clean = df.copy()
    
    type_mapping = {
        'job_type': {
            'full_time': [
                'full-time', 'full time', 'permanent', 'regular', 'employee', 'voltijds', 
                'vollzeit', 'heltid', 'a jornada completa', 'fulltid', 'fuldtid',
                'pełny etat', 'полная занятость', '全职', 'período integral', 'cdi', 'cdi cadre'
            ],
            'part_time': [
                'part-time', 'part time', 'teilzeit', 'deeltijds', 'meio período', 
                'частичная занятость', 'working student'
            ],
            'contract': [
                'contract', 'freelance', 'temporary', 'fixed term', 'consultant', 
                'contrat', 'contrato', 'werkvertrag', 'project based', 'fellowship', 
                'billable', 'fte or 1099', 'limited', 'cdd'
            ]
        },
        'work_mode': {
            'remote': [
                'remote', 'remoto', 'remoto primeiro', 'full remoto'
            ],
            'hybrid': [
                'hybrid'
            ],
            'onsite': [
                'in-office', 'in office', 'in-person', 'office-based', 
                'on-site', 'onsite', 'onroll', 'on-roll', 'on-rolls', 'in-office'
            ]
        },
        'internship': [
            'internship', 'intern', 'stage', 'staż', 'stagiair', 'praktikant', 
            'stagista', 'co-op', 'co op', 'graduate', 'industrial placement',
            'binance accelerator program', 'apprenticeship', 'thesis', 'training'
        ]
    }
                
    # Build regex patterns for each category
    job_type_patterns = {}
    for category, terms in type_mapping['job_type'].items(): 
        terms_sorted = sorted(terms, key=len, reverse=True)
        pattern = r'\b(?:' + '|'.join(re.escape(term) for term in terms_sorted) + r')\b'
        job_type_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    work_mode_patterns = {}
    for category, terms in type_mapping['work_mode'].items():
        terms_sorted = sorted(terms, key=len, reverse=True)
        pattern = r'\b(?:' + '|'.join(re.escape(term) for term in terms_sorted) + r')\b'
        work_mode_patterns[category] = re.compile(pattern, re.IGNORECASE)
    
    # Internship patterns 
    internship_terms = sorted(type_mapping['internship'], key=len, reverse=True)
    internship_pattern = re.compile(
        r'\b(?:' + '|'.join(re.escape(term) for term in internship_terms) + r')\b', 
        re.IGNORECASE
    )
    
    def categorize_job_details(job_type_val):
        """
        Categorize job type and work mode from raw job type value.
        Returns tuple of (job_type, work_mode, is_intern)
        """
        # Handle non-string inputs (including lists/arrays)
        if not isinstance(job_type_val, str):
            # If it's a list/array with at least one element, take the first element as string
            if isinstance(job_type_val, (list, np.ndarray)) and len(job_type_val) > 0:
                job_type_str = str(job_type_val[0])
            else:
                return 'unknown', 'unknown', False
        else:
            job_type_str = job_type_val
        
        # Check for NaN/None
        if pd.isna(job_type_str) or not isinstance(job_type_str, str):
            return 'unknown', 'unknown', False
        
        job_type_lower = job_type_str.lower()
        
        # Clean special characters but preserve spaces between words
        job_type_clean = re.sub(r'[^a-zA-Z\s,;/-]', ' ', job_type_lower)
        
        # Initialize results
        job_type_result = 'unknown'
        work_mode_result = 'unknown'
        is_intern = False
        
        # Check for internship first 
        if internship_pattern.search(job_type_clean):
            is_intern = True
            job_type_result = 'internship'
        
        # Check job type if not already set to internship
        if job_type_result == 'unknown':
            for jt, pattern in job_type_patterns.items():
                if pattern.search(job_type_clean):
                    job_type_result = jt
                    break
        
        # Check work mode 
        for wm, pattern in work_mode_patterns.items():
            if pattern.search(job_type_clean):
                work_mode_result = wm
                break
        
        if is_intern and work_mode_result == 'unknown':
            for wm, pattern in work_mode_patterns.items():
                if pattern.search(job_type_clean):
                    work_mode_result = wm
                    break
        
        return job_type_result, work_mode_result, is_intern
    
    # Apply categorization (single pass through data)
    results = df_clean['types'].apply(categorize_job_details)
    
    # Unpack results efficiently
    df_clean['job_type'] = [r[0] for r in results]
    df_clean['work_mode'] = [r[1] for r in results] 
        
    df_clean.loc[df_clean['job_type'] == 'unknown', 'job_type'] = np.nan
    df_clean.loc[df_clean['work_mode'] == 'unknown', 'work_mode'] = np.nan
            
    logger.info(f"Job types cleaned. {len(df_clean)} records processed.")
    
    return df_clean
            
