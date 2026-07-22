import pandas as pd
import numpy as np
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

class SalarySkillResidualAnalyzer:
    """
    Two-stage residual CatBoost salary analysis (backward compatible with pivot tables)
    1. Baseline model: confounders (company_name, published, location, etc.)
    2. Skill model: residuals on skills only
    Extracts SHAP skill premiums and top skill interactions.
    """

    def __init__(self,
                 salary_column='salary',
                 confounder_cols=None,
                 cat_cols_stage2=None,
                 min_samples_per_skill=5):
        self.salary_column = salary_column
        self.confounder_cols = confounder_cols if confounder_cols else ['company_name', 'published', 'location']
        self.cat_cols_stage2 = cat_cols_stage2 if cat_cols_stage2 else []
        self.min_samples = min_samples_per_skill

        self.baseline_model = None
        self.skill_model = None
        self.baseline_residuals = None
        self.results = {}

    # ---------------- Stage 1: Baseline model ----------------
    def fit_baseline_model(self, df: pd.DataFrame):
        X_conf = df[self.confounder_cols].copy()
        for col in X_conf.select_dtypes(include='object').columns:
            X_conf[col] = X_conf[col].fillna('Unknown').astype(str)

        y = df[self.salary_column]

        self.baseline_model = CatBoostRegressor(
            iterations=1000,
            depth=6,
            learning_rate=0.05,
            loss_function='RMSE',
            verbose=False
        )
        self.baseline_model.fit(X_conf, y)
        y_pred = self.baseline_model.predict(X_conf)
        self.baseline_residuals = y - y_pred
        return r2_score(y, y_pred)

    # ---------------- Stage 2: Skill model (pivot-compatible) ----------------
    def fit_skill_model_from_pivot(self, df: pd.DataFrame, pivot_df: pd.DataFrame):
        """
        Accept pivot table with job_ids per skill
        """
        # Reconstruct skill matrix (rows=jobs, cols=skills)
        exploded = pivot_df.explode('job_ids')
        skill_matrix = pd.crosstab(exploded['job_ids'], exploded['skill'])
        skill_matrix = skill_matrix.reindex(df.index, fill_value=0)  # ensure alignment

        # Filter rare skills
        valid_skills = [c for c in skill_matrix.columns if skill_matrix[c].sum() >= self.min_samples]
        skill_matrix = skill_matrix[valid_skills]

        # Include optional categorical controls
        for col in self.cat_cols_stage2:
            if col in df.columns:
                skill_matrix[col] = df[col].fillna('Unknown').astype(str)
            else:
                skill_matrix[col] = 'Unknown'

        self.skill_model = CatBoostRegressor(
            iterations=1500,
            depth=7,
            learning_rate=0.04,
            loss_function='RMSE',
            verbose=False
        )
        self.skill_model.fit(skill_matrix, self.baseline_residuals, cat_features=self.cat_cols_stage2)

        r2_skills = r2_score(self.baseline_residuals, self.skill_model.predict(skill_matrix))
        return r2_skills, skill_matrix

    # ---------------- SHAP skill premiums ----------------
    def extract_skill_shap(self, skill_matrix: pd.DataFrame):
        pool = Pool(skill_matrix, cat_features=self.cat_cols_stage2)
        shap_values = self.skill_model.get_feature_importance(pool, type='ShapValues')
        shap_matrix = pd.DataFrame(shap_values[:, :-1], columns=skill_matrix.columns, index=skill_matrix.index)

        skill_cols = [c for c in skill_matrix.columns if c not in self.cat_cols_stage2]

        premiums = []
        for skill in skill_cols:
            mask = skill_matrix[skill] == 1
            if mask.sum() > 0:
                avg_shap = shap_matrix.loc[mask, skill].mean()
                premiums.append({
                    'skill': skill,
                    'dollar_premium': avg_shap,
                    'prevalence': skill_matrix[skill].mean()
                })

        return pd.DataFrame(premiums).sort_values('dollar_premium', ascending=False)

    # ---------------- SHAP skill interactions ----------------
    def extract_skill_interactions(self, skill_matrix: pd.DataFrame, top_n=20):
        pool = Pool(skill_matrix, cat_features=self.cat_cols_stage2)
        interactions_raw = self.skill_model.get_feature_importance(pool, type='Interaction')

        skill_cols = [c for c in skill_matrix.columns if c not in self.cat_cols_stage2]
        interactions = []
        for f1_idx, f2_idx, score in interactions_raw[:top_n]:
            f1_name = skill_matrix.columns[f1_idx]
            f2_name = skill_matrix.columns[f2_idx]
            if f1_name in skill_cols and f2_name in skill_cols:
                interactions.append({
                    'skill1': f1_name,
                    'skill2': f2_name,
                    'synergy': score * 1000  # scaling for visualization
                })

        return pd.DataFrame(interactions).sort_values('synergy', ascending=False)

    # ---------------- Full two-stage pipeline ----------------
    def analyze_skills_from_pivot(self, df: pd.DataFrame, pivot_df: pd.DataFrame, top_interactions=20):
        print("Fitting baseline model (confounders)...")
        r2_baseline = self.fit_baseline_model(df)
        print(f"Baseline R²: {r2_baseline:.3f}")

        print("Fitting skill model on residuals from pivot table...")
        r2_skills, skill_matrix = self.fit_skill_model_from_pivot(df, pivot_df)
        print(f"Skill model R² on residuals: {r2_skills:.3f}")

        print("Extracting SHAP skill premiums...")
        skill_premiums = self.extract_skill_shap(skill_matrix)

        print(f"Extracting top {top_interactions} skill interactions...")
        skill_interactions = self.extract_skill_interactions(skill_matrix, top_n=top_interactions)

        self.results = {
            'baseline_r2': r2_baseline,
            'skill_r2': r2_skills,
            'skill_premiums': skill_premiums,
            'skill_interactions': skill_interactions
        }
        return self.results