import os
import re
import joblib
import torch
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util

import warnings
warnings.filterwarnings('ignore')

class HybridJobTitleClassifier:
    """
    Job title classifier that:
    - Uses SentenceTransformer for semantic similarity.
    - Classifies titles into predefined categories or marks as 'Other'.
    - Extracts seniority levels.
    - Preserves all original DataFrame columns.
    - Avoids duplicate rows by aligning on index, not title.
    """

    def __init__(self,
                 model_name="all-MiniLM-L6-v2", 
                 category_embeddings_path="categories.pkl"):

        cache_folder = "C:\\Users\\MARIE\\.cache\\huggingface\\hub"
        self.model = SentenceTransformer(model_name, cache_folder=cache_folder, local_files_only=True)
         
        self.category_embeddings_path = category_embeddings_path

        # Core job role categories
        self.categories = [
            "Data Scientist",
            "Machine Learning Engineer",
            "AI Engineer",
            "Data Analyst",
            "Data Engineer",
            "MLOps Engineer",
            "Data Architect",
            "Analytics Engineer",
            "Research Scientist",
            "Data Science Manager",
            "AI Researcher",
            "Product Data Scientist"
        ]
        self.category_embeddings = self.model.encode(self.categories, normalize_embeddings=True)

        # Seniority patterns (regex-based)
        self.seniority_patterns = [
            (r'\b(intern|internship|trainee|apprentice|student)\b', 'Intern'),
            (r'\b(entry[-\s]?level|junior|jr\.?)\b', 'Junior'),
            (r'\b(associate|mid[-\s]?level|intermediate)\b', 'Mid-level'),
            (r'\b(senior|sr\.?)\b', 'Senior'),
            (r'\b(lead|principal|specialist|staff)\b', 'Lead'),
            (r'\b(manager|head|director|supervisor)\b', 'Manager'),
            (r'\b(vice president|vp|chief|executive|cto|ceo|founder|co-founder)\b', 'Executive')
        ]

    def extract_seniority(self, title: str) -> str:
        """Extract seniority level using regex."""
        if not title or not isinstance(title, str):
            return "Unknown"
        title_lower = title.lower()
        for pattern, label in self.seniority_patterns:
            if re.search(pattern, title_lower):
                return label
        return "Unspecified"

    def classify_dataframe(self, df: pd.DataFrame, title_col: str, threshold =.6) -> pd.DataFrame:
        """
        Classify job titles based on semantic similarity to predefined categories.
        Titles below similarity threshold are marked as 'Other'.
        Returns same number of rows with 3 new columns.
        """
        df = df.copy()
        titles = df[title_col].fillna("").tolist()

        # Encode job titles
        title_embeddings = self.model.encode(titles, normalize_embeddings=True)
        similarities = util.cos_sim(title_embeddings, self.category_embeddings)

        results = []
        for i, title in enumerate(titles):
            best_idx = torch.argmax(similarities[i]).item()
            best_score = similarities[i][best_idx].item()
            seniority = self.extract_seniority(title)

            category = self.categories[best_idx] if best_score > threshold else 'Other' 

            results.append((title, category, seniority, best_score))

        # Results DataFrame (index-aligned)
        results_df = pd.DataFrame(results, columns=[
            title_col,
            "cleaned_title_category",
            "seniority_level",
            "similarity_score"
        ])
        results_df.index = df.index

        # Merge with original DataFrame
        df_final = pd.concat([
            df,
            results_df[["cleaned_title_category", "seniority_level", "similarity_score"]]
        ], axis=1)
       
        return df_final

    def save_state(self):
        """Persist category embeddings."""
        joblib.dump(self.category_embeddings, self.category_embeddings_path)
        print(f"✅ State saved: {len(self.categories)} predefined categories.")


import pandas as pd

test_data = pd.DataFrame({
    "title": [
        "Data Scientist",
        "Senior Data Scientist",
        "Junior Data Analyst",
        "Machine Learning Engineer",
        "Lead Data Engineer",
        "AI Engineer",
        "Research Scientist",
        "Principal Data Architect",
        "Data Science Manager",
        "MLOps Engineer",
        "Business Intelligence Analyst",
        "Data Visualization Specialist",
        "NLP Researcher",
        "Student Data Intern",
        "Chief AI Officer"
    ]
})

clf = HybridJobTitleClassifier()
results = clf.classify_dataframe(test_data, "title",threshold=.5)
print(results[["title", "cleaned_title_category", "seniority_level", "similarity_score"]])
