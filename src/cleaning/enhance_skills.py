import pandas as pd
import ast
import logging
# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

import warnings
warnings.filterwarnings('ignore')

class SkillEnhancer:
    GENERIC_SKILLS = {
        'Data Science', 'Artificial Intelligence', 'Ai', 'Analytics',
        'Data Analysis', 'Programming', 'Coding', 'Statistics', 'Ml', 'Machine Learning', 'Data Engineer'
    }

    @staticmethod
    def parse_skills(skills_str: str) -> list[str]:
        """Parse a skills string into a cleaned list."""
        if not isinstance(skills_str, str) or not skills_str.strip():
            return []

        try:
            # Convert list-like strings safely
            skills = ast.literal_eval(skills_str) if skills_str.startswith('[') else skills_str.split(',')
            # Normalize and filter
            cleaned = {
                s.strip().title() for s in skills
                if s and s.strip() and s.strip().title() not in SkillEnhancer.GENERIC_SKILLS
            }
            return list(cleaned)
        except Exception:
            logger.warning(f"Could not parse skills string: {skills_str}")
            return []

    @staticmethod
    def enhance_skills_data(df: pd.DataFrame) -> pd.DataFrame:
        """Enhance DataFrame by cleaning skills and adding skill count."""
        df = df.copy()
        df['skills'] = df['skills'].apply(SkillEnhancer.parse_skills)
        df['skills_count'] = df['skills'].apply(len)
        logger.info("✅ Skills data enhanced")
        return df
