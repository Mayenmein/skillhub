# src/processing/clean_skills.py
import pandas as pd
import numpy as np
import re

from pathlib import Path
import logging
from typing import List, Optional

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

from src.core.config_salary import EXCHANGE_RATES, COUNTRY_CURRENCY, skip_patterns, currency_patterns


class DataScienceJobsCleaner:
    """
    Clean and preprocess data science job listings from Found.dev API
    """
        
    def __init__(self, data_dir: Path = Path("../data")):
        """
        Initialize the cleaner with data directory paths
        
        Args:
            data_dir: Root data directory path
        """
        self.data_dir = data_dir
        self.raw_dir = self.data_dir / "raw"
        self.interim_dir = self.data_dir / "interim"
        self.processed_dir = self.data_dir / "processed"
        
        # Create directories if they don't exist
        self.interim_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
    def load_raw_data(self, file_path: Path) -> pd.DataFrame:
        """
        Load raw data from batch CSV files
        
        Args:
            batch_files: Specific batch files to load. If None, loads all batches
            
        Returns:
            Combined DataFrame of all raw data
        """
        if not file_path:
            raise FileNotFoundError("No file found")
        
        
        combined_df = pd.read_csv(file_path)
        
        return combined_df
    
    def clean_location_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance location information"""
        df_clean = df.copy()
        
        # Country standardization mapping
        COUNTRY_MAP = {
            # US variations
            **{state: 'USA' for state in ['AL','AK','AZ','AR','CA','CO','CT','DE','FL','GA',
                                        'HI','ID','IL','IN','IA','KS','KY','LA','ME','MD',
                                        'MA','MI','MN','MS','MO','MT','NE','NV','NH','NJ',
                                        'NM','NY','NC','ND','OH','OK','OR','PA','RI','SC',
                                        'SD','TN','TX','UT','VT','VA','WA','WV','WI','WY']},
            **{us_var: 'USA' for us_var in ['US', 'U S.', 'UNITED STATES', 'UNITED STATES OF AMERICA', 'CA US','USA']},
            
            # Other countries
            'UK': 'United Kingdom', 'UNITED KINGDOM': 'United Kingdom','ENGLAND':'United Kingdom',
            'UAE': 'United Arab Emirates', 'FR': 'France', 'ZA': 'South Africa',
            'CY': 'Cyprus', 'UA': 'Ukraine', 'IT': 'Italy', 'MAROC': 'Morocco',
            'CZECHIA': 'Czech Republic', 'BOSNIA': 'Bosnia and Herzegovina',
            'KOREA': 'South Korea', 'UNKNOWN': 'Unknown', '[ ]': 'Unknown',
            'EUROPE': 'Region/Continent', 'NORTH AMERICA': 'Region/Continent',
            'LATIN AMERICA': 'Region/Continent','TURKIYE':'Turkey','TUNISIE':'Tunisia','BRASIL':'Brazil',
            'JP': 'Japan','UY':'Uruguay','PH':'Philippines','SG':'Singapore','NL':'Netherlands',
            'SE':'Sweden','CH':'Switzerland','GR':'Greece','IN':'India','PT':'Portugal',
            'AU':'Australia','NZ':'New Zealand','BE':'Belgium','MX':'Mexico','RU':'Russia',
            'TR':'Turkey','EG':'Egypt','FI':'Finland','SA':'Saudi Arabia','IE':'Ireland',
            'The Netherlands'.upper():'Netherlands','Bosnia and Herzegovina'.upper():'Bosnia and Herzegovina',
            'Korea, Republic Of'.upper():'South Korea'
        }
        
        # Extract and clean country data
        def extract_country(location):
            if pd.isna(location): return 'Unknown'
            country = str(location).split(':')[0].split(';')[0].strip().upper()
            return COUNTRY_MAP.get(country, country.title())
        
        df_clean['country'] = df_clean['country'].apply(extract_country)
        
        logger.info("✅ Location data cleaned")
        return df_clean
      
    def clean_job_type(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean and categorize job types into separate columns.
        Handles multilingual and mixed-format job type fields.
        """
        df_clean = df.copy()

        # Define standard mappings (lowercased)
        type_mapping = {
            'job_type': {
                'full_time': [
                    'full-time', 'full time', 'permanent', 'regular', 'employee', 'voltijds', 
                    'vollzeit', 'heltid', 'a jornada completa', 'fulltid', 'fuldtid',
                    'pełny etat', 'полная занятость', '全职', 'período integral'
                ],
                'part_time': [
                    'part-time', 'part time', 'teilzeit', 'deeltijds', 'meio período', 
                    'чaстичная занятость', 'working student'
                ],
                'contract': [
                    'contract', 'freelance', 'temporary', 'fixed term', 'consultant', 
                    'contrat', 'contrato', 'werkvertrag', 'project based', 'fellowship', 
                    'billable', 'fte or 1099', 'limited', 'cdi', 'cdi cadre'
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

        def categorize_job_details(job_type):
            if pd.isna(job_type):
                return {
                    'job_type': 'unknown',
                    'work_mode': 'unknown', 
                    'is_intern': False
                }

            job_type_lower = str(job_type).lower()
            job_type_lower = re.sub(r'[^a-zA-Z\s,;/-]', ' ', job_type_lower)
            tokens = re.split(r'[,;/]', job_type_lower)

            # Initialize with NaN/False values
            result = {
                'job_type': 'unknown',
                'work_mode': 'unknown',
                'is_intern': False
            }

            # Check for job types
            job_type_found = False
            for token in tokens:
                token = token.strip()
                for category, variants in type_mapping['job_type'].items():
                    if any(v in token for v in variants):
                        result['job_type'] = category
                        job_type_found = True
                        break
                if job_type_found:
                    break

            # Check for work modes
            work_mode_found = False
            for token in tokens:
                token = token.strip()
                for category, variants in type_mapping['work_mode'].items():
                    if any(v in token for v in variants):
                        result['work_mode'] = category
                        work_mode_found = True
                        break
                if work_mode_found:
                    break

            # Check for internship
            for token in tokens:
                token = token.strip()
                if any(intern_term in token for intern_term in type_mapping['internship']):
                    result['is_intern'] = True
                    # If it's an internship and no job type found, set as internship type
                    if not job_type_found:
                        result['job_type'] = 'internship'
                    break

            return result

        # Apply categorization
        job_details = df_clean['type'].apply(categorize_job_details)
        
        # Create separate columns
        df_clean['job_type'] = job_details.apply(lambda x: x['job_type'])
        df_clean['work_mode'] = job_details.apply(lambda x: x['work_mode'])
        df_clean['is_intern'] = job_details.apply(lambda x: x['is_intern'])
        
        # Replace 'unknown' with NaN for better handling
        df_clean['job_type'] = df_clean['job_type'].replace('unknown', np.nan)
        df_clean['work_mode'] = df_clean['work_mode'].replace('unknown', np.nan)

        # Optional: Keep the original cleaned_job_type for backward compatibility
        df_clean['cleaned_job_type'] = df_clean['type'].apply(
            lambda x: ['unknown'] if pd.isna(x) else [str(x).lower()]
        )

        logger.info("✅ Job types cleaned.") 

        return df_clean
    
    def clean_salary_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and enhance salary information with robust error handling"""
        df_clean = df.copy()
        
        # Convert 0 values to NaN
        salary_cols = ['salary', 'salary_min', 'salary_max']
        for col in salary_cols:
            df_clean[col] = df_clean[col].replace(0, np.nan)
        
        def extract_and_convert_salary(salary_str, country, company=None):
            """Core salary extraction logic - simplified"""
            try:
                if pd.isna(salary_str):
                    return np.nan, np.nan
                
                salary_str = str(salary_str).strip().lower()
                if any(pattern in salary_str for pattern in skip_patterns):
                    return np.nan, np.nan
                
                # Normalize country
                country_name = normalize_country(country) if isinstance(country, str) else None
                
                # Determine currency
                currency = detect_currency(salary_str, country_name)
                exchange_rate = EXCHANGE_RATES.get(currency, 1.0)
                
                # Extract numbers by context
                numbers = extract_numbers_by_context(salary_str)
                
                # Convert to annual USD
                min_salary, max_salary = convert_to_annual_usd(numbers, salary_str, exchange_rate)
                
                # Apply sanity checks
                min_usd, max_usd = apply_sanity_checks(min_salary, max_salary, country_name, currency, salary_str)
                
                return min_usd, max_usd
                
            except Exception:
                return np.nan, np.nan
        
        def normalize_country(country):
            """Simplify country normalization"""
            if not country or not isinstance(country, str):
                return None
                
            country = country.strip().title()
            return (country.replace('Usa', 'United States')
                        .replace('Uk', 'United Kingdom')
                        .replace('U.k.', 'United Kingdom')
                        .replace('U.s.', 'United States')
                        .replace('America', 'United States')
                        .replace('Us', 'United States'))
        
        def detect_currency(salary_str, country_name):
            """Determine currency from string and country"""
            # Check for currency symbols in string
            for pattern, curr in currency_patterns.items():
                if pattern in salary_str:
                    return curr
            
            # Fall back to country mapping
            if country_name and country_name in COUNTRY_CURRENCY:
                return COUNTRY_CURRENCY[country_name]
            
            return 'USD'
        
        def extract_numbers_by_context(salary_str):
            """Extract numbers with their context"""
            # Remove currency symbols
            clean_str = re.sub(r'[$€£¥₹₱₩₫₺₦₽]', '', salary_str)
            
            # Extract ranges first
            range_match = re.search(r'(\d[\d,]*\.?\d*)\s*[-–—]\s*(\d[\d,]*\.?\d*)', clean_str)
            if range_match:
                try:
                    num1 = float(range_match.group(1).replace(',', ''))
                    num2 = float(range_match.group(2).replace(',', ''))
                    if num1 > 0 and num2 > 0:
                        return {'type': 'range', 'min': min(num1, num2), 'max': max(num1, num2)}
                except:
                    pass
            
            # Extract all numbers
            numbers = []
            for match in re.findall(r'(\d[\d,]+\.?\d*)', clean_str):
                try:
                    num = float(match.replace(',', ''))
                    if num > 10:  # Ignore small numbers
                        numbers.append(num)
                except:
                    continue
            
            if not numbers:
                return {'type': 'none', 'numbers': []}
            
            # Determine context
            if 'hour' in salary_str or '/h' in salary_str:
                return {'type': 'hourly', 'numbers': numbers}
            elif 'month' in salary_str:
                return {'type': 'monthly', 'numbers': numbers}
            elif 'year' in salary_str or 'annual' in salary_str:
                return {'type': 'annual', 'numbers': numbers}
            else:
                return {'type': 'unknown', 'numbers': numbers}
        
        def convert_to_annual_usd(numbers_data, salary_str, exchange_rate):
            """Convert extracted numbers to annual USD"""
            if numbers_data['type'] == 'range':
                min_val, max_val = numbers_data['min'], numbers_data['max']
            elif numbers_data['type'] == 'none' or not numbers_data.get('numbers'):
                return np.nan, np.nan
            else:
                nums = numbers_data['numbers']
                min_val, max_val = min(nums), max(nums)
            
            # Apply k notation multiplier
            if 'k' in salary_str:
                # Check if numbers are in thousands
                if max_val < 1000:
                    min_val *= 1000
                    max_val *= 1000
            
            # Convert based on period
            if numbers_data['type'] == 'hourly':
                min_val *= 2080  # 40hrs/week * 52 weeks
                max_val *= 2080
            elif numbers_data['type'] == 'monthly':
                min_val *= 12
                max_val *= 12
            
            # Apply currency conversion
            return min_val * exchange_rate, max_val * exchange_rate
        
        def apply_sanity_checks(min_usd, max_usd, country_name, currency, salary_str):
            """Apply reasonable range checks"""
            if pd.isna(min_usd) or pd.isna(max_usd):
                return np.nan, np.nan
            
            # Set reasonable bounds
            min_bound, max_bound = 15000, 500000
            
            if country_name:
                if country_name in ['Philippines', 'Vietnam', 'India', 'Indonesia']:
                    max_bound = 100000
                    # Special check for Philippines
                    if country_name == 'Philippines' and currency == 'USD' and min_usd > 50000:
                        return np.nan, np.nan
            
            # Check if values are reasonable
            if (min_usd < min_bound and max_usd < min_bound) or \
            (min_usd > max_bound and max_usd > max_bound):
                # Skip if not project/course based
                if not any(term in salary_str for term in ['per course', 'per project', 'course', 'project']):
                    return np.nan, np.nan
            
            # Fix ordering and extreme ratios
            if min_usd > max_usd:
                min_usd, max_usd = max_usd, min_usd
            
            if min_usd > 0 and max_usd / min_usd > 10:
                avg = (min_usd + max_usd) / 2
                min_usd, max_usd = avg * 0.8, avg * 1.2
            
            return min_usd, max_usd
        
        def detect_salary_type(salary_str):
            """Simple salary type detection"""
            if pd.isna(salary_str):
                return 'unknown'
            
            salary_lower = str(salary_str).lower()
            type_patterns = [
                (['/hour', 'hourly', 'per hour'], 'hourly'),
                (['/month', 'monthly', 'per month'], 'monthly'),
                (['/year', 'annual', 'annually', 'per year'], 'annual'),
                (['per course', 'course'], 'per_course'),
                (['per project', 'project', 'contract'], 'per_project')
            ]
            
            for patterns, salary_type in type_patterns:
                if any(pattern in salary_lower for pattern in patterns):
                    return salary_type
            return 'unknown'
        
        def categorize_salary(salary, salary_type, country):
            """Categorize salary into bands"""
            if pd.isna(salary):
                return 'Unknown'
            
            salary = float(salary)
            
            # Determine region
            region = 'default'
            if isinstance(country, str):
                country_lower = country.lower()
                if any(c in country_lower for c in ['philippines', 'india', 'vietnam', 'indonesia']):
                    region = 'low_col'
                elif any(c in country_lower for c in ['mexico', 'brazil', 'colombia']):
                    region = 'medium_col'
            
            # Define thresholds
            if salary_type in ['per_course', 'per_project']:
                thresholds = {
                    'low_col': [(1000, 'Low Per-Course (<1k)'), (3000, 'Medium Per-Course (1k-3k)'), 
                            (7000, 'High Per-Course (3k-7k)'), (float('inf'), 'Very High Per-Course (>7k)')],
                    'medium_col': [(2000, 'Low Per-Course (<2k)'), (5000, 'Medium Per-Course (2k-5k)'), 
                                (10000, 'High Per-Course (5k-10k)'), (float('inf'), 'Very High Per-Course (>10k)')],
                    'default': [(3000, 'Low Per-Course (<3k)'), (8000, 'Medium Per-Course (3k-8k)'), 
                            (15000, 'High Per-Course (8k-15k)'), (float('inf'), 'Very High Per-Course (>15k)')]
                }
            else:
                thresholds = {
                    'low_col': [(10000, 'Very Low (<10k)'), (25000, 'Low (10k-25k)'), 
                            (50000, 'Medium (25k-50k)'), (100000, 'High (50k-100k)'), 
                            (float('inf'), 'Very High (>100k)')],
                    'medium_col': [(15000, 'Very Low (<15k)'), (35000, 'Low (15k-35k)'), 
                                (70000, 'Medium (35k-70k)'), (120000, 'High (70k-120k)'), 
                                (float('inf'), 'Very High (>120k)')],
                    'default': [(30000, 'Low (<30k)'), (60000, 'Medium-Low (30k-60k)'), 
                            (100000, 'Medium (60k-100k)'), (150000, 'Medium-High (100k-150k)'), 
                            (250000, 'High (150k-250k)'), (500000, 'Very High (250k-500k)'), 
                            (float('inf'), 'Executive (>500k)')]
                }
            
            for threshold, label in thresholds[region]:
                if salary <= threshold:
                    return label
            
            return 'Unknown'
        
        # Apply extraction
        results = []
        for _, row in df_clean.iterrows():
            min_salary, max_salary = extract_and_convert_salary(
                row['salary'],
                row.get('country', ''),
                row.get('company', None)
            )
            results.append((min_salary, max_salary))
        
        df_clean['salary_min_usd'] = [r[0] for r in results]
        df_clean['salary_max_usd'] = [r[1] for r in results]
        df_clean['avg_salary_usd'] = df_clean[['salary_min_usd', 'salary_max_usd']].mean(axis=1)
        
        # Add metadata
        df_clean['salary_type'] = df_clean['salary'].apply(detect_salary_type)
        df_clean['salary_category'] = df_clean.apply(
            lambda row: categorize_salary(row['avg_salary_usd'], row['salary_type'], row.get('country', '')),
            axis=1
        )
        
        # Filter extreme outliers - FIXED: Can't use list with .at
        mask = df_clean['avg_salary_usd'].notna()
        for idx in df_clean[mask].index:
            salary = df_clean.at[idx, 'avg_salary_usd']
            salary_str = str(df_clean.at[idx, 'salary']).lower()
            country = str(df_clean.at[idx, 'country']).lower() if 'country' in df_clean.columns else ''
            
            # Remove obvious errors
            if salary > 5000000 or \
            ('philippines' in country and salary > 200000 and 'monthly' in salary_str) or \
            ('meta' in salary_str and salary > 1000000):
                # Set each column individually
                df_clean.at[idx, 'salary_min_usd'] = np.nan
                df_clean.at[idx, 'salary_max_usd'] = np.nan
                df_clean.at[idx, 'avg_salary_usd'] = np.nan
        
        valid_count = df_clean['avg_salary_usd'].notna().sum()
        logger.info(f"✅ Salary data cleaned. {valid_count}/{len(df_clean)} valid salaries found.")
        
        return df_clean
    
    def convert_dates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert date columns to proper datetime format
        
        Args:
            df: DataFrame with date columns
            
        Returns:
            DataFrame with converted dates
        """
        df_clean = df.copy()
        
        # Convert published date
        df_clean['published'] = pd.to_datetime(df_clean['published'], utc=True, errors='coerce')
        
        # Extract date components
        df_clean['published_year'] = df_clean['published'].dt.year
        df_clean['published_month'] = df_clean['published'].dt.month
        
        df_clean["date"] = df_clean["published_year"].astype(str) + "_" + df_clean["published_month"].astype(str).str.zfill(2)
        
        logger.info("✅ Date data converted")
        return df_clean
        
    def run_full_cleaning_pipeline(self, batch_files: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Execute the complete data cleaning pipeline
        
        Args:
            batch_files: Specific batch files to process
            
        Returns:
            Fully cleaned DataFrame
        """
        logger.info("🚀 Starting data cleaning pipeline...")
        
        # Load raw data
        df = self.load_raw_data(self.data_dir/'combined_jobs.csv')
        
        # Apply cleaning steps
        df_clean = (df
                   .pipe(self.clean_location_data)
                   .pipe(self.clean_job_type)
                   .pipe(self.clean_salary_data)
                   .pipe(self.convert_dates) 
                    )
        df_clean = df_clean.reset_index(drop=True)
        # Final data quality check
        self._quality_check(df_clean)
        
        logger.info(f"🎉 Cleaning complete! Final dataset: {len(df_clean)} records")
        
        return df_clean
    
    def _quality_check(self, df: pd.DataFrame):
        """
        Perform basic data quality checks
        
        Args:
            df: Cleaned DataFrame to check
        """
        logger.info("🔍 Performing data quality checks...")
        
        # Check for missing values
        missing_data = df.isnull().sum()
        total_records = len(df)
        
        for column, missing_count in missing_data.items():
            if missing_count > 0:
                percentage = (missing_count / total_records) * 100
                logger.info(f"   {column}: {missing_count} missing ({percentage:.1f}%)")
        
        # Check data types
        logger.info("📊 Data types:")
        for column, dtype in df.dtypes.items():
            logger.info(f"   {column}: {dtype}")
    
    def save_cleaned_data(self, df: pd.DataFrame, output_type: str = "interim"):
        """
        Save cleaned data to appropriate directory
        
        Args:
            df: Cleaned DataFrame to save
            output_type: "interim" for partially cleaned, "processed" for final
        """
        if output_type == "interim":
            output_path = self.interim_dir / "cleaned_jobs.csv"
        elif output_type == "processed":
            output_path = self.processed_dir / "processed_jobs.csv"
        else:
            raise ValueError("output_type must be 'interim' or 'processed'")
        
        df.to_csv(output_path, index=False)
        logger.info(f"💾 Saved {len(df)} records to {output_path}")
