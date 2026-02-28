import pandas as pd
import numpy as np
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.preprocessing import StandardScaler
from itertools import combinations
from typing import Dict, List, Tuple, Optional
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
import json
from pathlib import Path

class SalarySkillRegressionAnalyzer:
    """
    Salary analysis built on existing skill pivot table
    Handles location adjustment and missing salary filtering
    """
    
    def __init__(self, 
                 salary_column: str = 'salary',
                 location_column: str = 'country',  # or 'location' depending on your data
                 min_samples_per_skill: int = 5,
                 max_interaction_features: int = 100):
        """
        Args:
            salary_column: Column name for salary in original data
            location_column: Column for location normalization
            min_samples_per_skill: Minimum jobs per skill to include
            max_interaction_features: Limit to prevent combinatorial explosion
        """
        self.salary_column = salary_column
        self.location_column = location_column
        self.min_samples = min_samples_per_skill
        self.max_interaction_features = max_interaction_features
        
        # Models and results
        self.model = None
        self.scaler = StandardScaler()
        self.location_adjustments = {}
        self.results = {}
        
        # Feature mapping
        self.feature_to_skills = {}  # Maps features back to skills
        self.skill_to_feature = {}   # Maps skills to feature indices
    
    def analyze_from_pivot(
        self,
        original_df: pd.DataFrame,  # Original data with salary
        skill_pivot_df: pd.DataFrame,  # Your existing pivot table
        adjust_for_location: bool = True,
        reference_location: Optional[str] = None,
        include_interactions: bool = True
    ) -> Dict:
        """
        Complete salary analysis using existing pivot table
        
        Args:
            original_df: Original DataFrame with salary column
            skill_pivot_df: Your skill pivot table from DataProcessor
            adjust_for_location: Whether to normalize salaries by location
            reference_location: Reference location for normalization
            include_interactions: Whether to include skill interaction terms
        
        Returns:
            Complete analysis results
        """
        print("Starting salary skill regression analysis...")
        
        # Step 1: Merge salary data with pivot table
        merged_data = self._merge_salary_with_pivot(original_df, skill_pivot_df)
        print(f"Merged {len(merged_data['X'])} jobs with salary data")
        
        # Step 2: Adjust for location if needed
        if adjust_for_location and self.location_column in merged_data['X'].columns:
            merged_data = self._adjust_salaries_for_location(
                merged_data, reference_location
            )
            print(f"Applied location adjustments")
        
        # Step 3: Create feature matrix
        X, y, feature_names = self._create_feature_matrix(
            merged_data, include_interactions
        )
        print(f"Created {len(feature_names)} features")
        
        # Step 4: Fit regression model
        model_results = self._fit_regression_model(X, y)
        
        # Step 5: Extract insights
        insights = self._extract_insights(
            X, y, model_results, merged_data, original_df
        )
        
        # Store complete results
        self.results = {
            'model_performance': model_results['performance'],
            'skill_premiums': insights['skill_premiums'],
            'interaction_effects': insights['interaction_effects'],
            'location_insights': insights['location_insights'],
            'recommendations': insights['recommendations'],
            'predictor': lambda skills, loc=None: self._predict_salary(
                skills, loc, X.columns.tolist()
            ),
            'data_statistics': {
                'total_jobs': len(X),
                'skills_analyzed': len([f for f in feature_names if '×' not in f]),
                'interactions_analyzed': len([f for f in feature_names if '×' in f]),
                'avg_salary': y.mean(),
                'median_salary': y.median()
            }
        }
        
        return self.results
    
    def _merge_salary_with_pivot(
        self,
        original_df: pd.DataFrame,
        pivot_df: pd.DataFrame
    ) -> Dict:
        """
        Merge salary data from original_df with skill pivot table
        """
        # Ensure we have job_id in both dataframes
        if 'job_id' not in original_df.columns:
            original_df = original_df.reset_index().rename(columns={'index': 'job_id'})
        
        if 'job_ids' not in pivot_df.columns:
            raise ValueError("Pivot table must have 'job_ids' column")
        
        # Filter original_df to only rows with salary
        salary_df = original_df[original_df[self.salary_column].notna()].copy()
        
        if len(salary_df) == 0:
            raise ValueError(f"No rows with non-null {self.salary_column}")
        
        # Create job-skill mapping from pivot table
        skill_mapping = {}
        for _, row in pivot_df.iterrows():
            skill = row['skill']
            job_ids = row['job_ids']
            
            if isinstance(job_ids, list):
                for job_id in job_ids:
                    if job_id not in skill_mapping:
                        skill_mapping[job_id] = []
                    skill_mapping[job_id].append(skill)
        
        # Create binary matrix
        all_skills = set()
        for skills in skill_mapping.values():
            all_skills.update(skills)
        
        all_skills = sorted(list(all_skills))
        
        # Create feature matrix
        X = pd.DataFrame(0, index=salary_df['job_id'], columns=all_skills)
        for job_id, skills in skill_mapping.items():
            if job_id in X.index:
                X.loc[job_id, skills] = 1
        
        # Add location and other metadata
        for col in [self.location_column, 'seniority_level', 'standardized_title']:
            if col in salary_df.columns:
                X[col] = salary_df.set_index('job_id')[col]
        
        # Target variable
        y = pd.Series(
            salary_df.set_index('job_id')[self.salary_column],
            index=X.index
        )
        
        return {
            'X': X,
            'y': y,
            'original_df': salary_df,
            'pivot_df': pivot_df
        }
    
    def _adjust_salaries_for_location(
        self,
        data: Dict,
        reference_location: Optional[str] = None
    ) -> Dict:
        """
        Adjust salaries for location cost differences
        """
        X = data['X']
        y = data['y']
        
        if self.location_column not in X.columns:
            return data
        
        # Calculate location medians
        location_stats = []
        for location in X[self.location_column].unique():
            loc_mask = X[self.location_column] == location
            if loc_mask.sum() >= self.min_samples:
                loc_salaries = y[loc_mask]
                location_stats.append({
                    'location': location,
                    'median': loc_salaries.median(),
                    'mean': loc_salaries.mean(),
                    'count': len(loc_salaries)
                })
        
        location_stats_df = pd.DataFrame(location_stats)
        
        if len(location_stats_df) < 2:
            return data
        
        # Calculate overall median
        overall_median = y.median()
        
        # Calculate adjustment factors
        self.location_adjustments = {}
        for _, row in location_stats_df.iterrows():
            loc = row['location']
            loc_median = row['median']
            
            if reference_location and loc == reference_location:
                adj_factor = 1.0
            elif reference_location and reference_location in location_stats_df['location'].values:
                ref_median = location_stats_df.loc[
                    location_stats_df['location'] == reference_location, 'median'
                ].iloc[0]
                adj_factor = ref_median / loc_median if loc_median > 0 else 1.0
            else:
                adj_factor = overall_median / loc_median if loc_median > 0 else 1.0
            
            self.location_adjustments[loc] = {
                'adjustment_factor': adj_factor,
                'original_median': loc_median,
                'sample_size': row['count']
            }
        
        # Apply adjustments
        adjusted_y = y.copy()
        for idx in y.index:
            location = X.loc[idx, self.location_column]
            if location in self.location_adjustments:
                adj_factor = self.location_adjustments[location]['adjustment_factor']
                adjusted_y.loc[idx] = y.loc[idx] * adj_factor
        
        # Update data
        data['y'] = adjusted_y
        data['y_original'] = y  # Keep original for reference
        data['location_stats'] = location_stats_df
        
        return data
    
    def _create_feature_matrix(
        self,
        data: Dict,
        include_interactions: bool = True
    ) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
        """
        Create feature matrix from binary skill matrix
        """
        X = data['X']
        y = data['y']
        
        # Filter to only skill columns (remove metadata columns)
        skill_columns = [col for col in X.columns 
                        if col not in [self.location_column, 'seniority_level', 'standardized_title']]
        
        X_skills = X[skill_columns].copy()
        
        # Filter rare skills
        skill_counts = X_skills.sum()
        common_skills = skill_counts[skill_counts >= self.min_samples].index.tolist()
        X_skills = X_skills[common_skills]
        
        # Generate interactions if requested
        interaction_features = []
        if include_interactions and len(common_skills) >= 2:
            # Pass a copy to avoid modifying the original
            interaction_features = self._generate_interaction_features(
                X_skills.copy(), common_skills
            )
        
        # Combine all features - X_skills already contains the interaction features
        # because _generate_interaction_features adds them to the passed dataframe
        all_features = common_skills + interaction_features
        X_final = X_skills  # X_skills now includes interaction columns
        
        # Limit total features if needed
        if len(all_features) > self.max_interaction_features:
            # Select most predictive features
            feature_scores = []
            for feature in all_features:
                if feature in X_final.columns:
                    corr, _ = stats.pearsonr(X_final[feature], y)
                    feature_scores.append((feature, abs(corr), X_final[feature].sum()))
            
            feature_scores.sort(key=lambda x: (x[1], x[2]), reverse=True)
            selected_features = [fs[0] for fs in feature_scores[:self.max_interaction_features]]
            X_final = X_final[selected_features]
            all_features = selected_features
        
        return X_final, y, all_features
    
    def _generate_interaction_features(
        self,
        X: pd.DataFrame,
        skills: List[str]
    ) -> List[str]:
        """
        Generate interaction features from most frequent skill pairs
        """
        interactions = []
        
        # Get top skills by frequency
        skill_freq = X.sum()
        top_skills = skill_freq.nlargest(min(20, len(skills))).index.tolist()
        
        for i, skill1 in enumerate(top_skills):
            for skill2 in top_skills[i+1:]:
                if len(interactions) >= 50:  # Limit interactions
                    break
                
                # Check co-occurrence
                cooccurrence = ((X[skill1] == 1) & (X[skill2] == 1)).sum()
                
                if cooccurrence >= max(3, self.min_samples):
                    # Create interaction feature
                    interaction_name = f"{skill1}×{skill2}"
                    X[interaction_name] = (X[skill1] & X[skill2]).astype(int)
                    interactions.append(interaction_name)
        
        return interactions
    
    def _fit_regression_model(
        self,
        X: pd.DataFrame,
        y: pd.Series
    ) -> Dict:
        """
        Fit regression model with automatic feature selection
        """
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Use LassoCV for automatic feature selection
        self.model = LassoCV(
            cv=5, 
            random_state=42, 
            max_iter=5000,
            n_alphas=100
        )
        self.model.fit(X_scaled, y)
        
        # Get predictions
        y_pred = self.model.predict(X_scaled)
        
        # Performance metrics
        r2 = self.model.score(X_scaled, y)
        
        # Cross-validation scores
        from sklearn.model_selection import cross_val_score
        cv_scores = cross_val_score(self.model, X_scaled, y, cv=5, scoring='r2')
        
        # Feature importance
        coefficients = dict(zip(X.columns, self.model.coef_))
        nonzero_features = {k: v for k, v in coefficients.items() if v != 0}
        
        return {
            'performance': {
                'r2': r2,
                'r2_cv_mean': cv_scores.mean(),
                'r2_cv_std': cv_scores.std(),
                'rmse': np.sqrt(np.mean((y - y_pred) ** 2)),
                'mae': np.mean(np.abs(y - y_pred))
            },
            'coefficients': coefficients,
            'nonzero_features': nonzero_features,
            'feature_count': len(nonzero_features)
        }
    
    def _extract_insights(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_results: Dict,
        merged_data: Dict,
        original_df: pd.DataFrame
    ) -> Dict:
        """
        Extract actionable insights from model results
        """
        coefs = model_results['coefficients']
        
        # 1. Individual skill premiums
        skill_premiums = self._extract_skill_premiums(X, y, coefs)
        
        # 2. Interaction effects
        interaction_effects = self._extract_interaction_effects(X, y, coefs)
        
        # 3. Location insights (if available)
        location_insights = self._extract_location_insights(merged_data, coefs, X)
        
        # 4. Generate recommendations
        recommendations = self._generate_recommendations(
            skill_premiums, interaction_effects, location_insights
        )
        
        return {
            'skill_premiums': skill_premiums,
            'interaction_effects': interaction_effects,
            'location_insights': location_insights,
            'recommendations': recommendations
        }
    
    def _extract_skill_premiums(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        coefficients: Dict
    ) -> pd.DataFrame:
        """Extract individual skill premiums"""
        premiums = []
        
        for feature, coef in coefficients.items():
            if '×' not in feature and coef != 0:  # Individual skill
                skill_data = {
                    'skill': feature,
                    'coefficient': coef,
                    'premium_category': self._categorize_premium(coef),
                    'prevalence': X[feature].mean(),
                    'sample_size': X[feature].sum(),
                    'avg_salary_with_skill': y[X[feature] == 1].mean() if X[feature].sum() > 0 else np.nan,
                    'median_salary_with_skill': y[X[feature] == 1].median() if X[feature].sum() > 0 else np.nan
                }
                premiums.append(skill_data)
        
        if not premiums:
            return pd.DataFrame()
        
        premiums_df = pd.DataFrame(premiums)
        premiums_df['abs_coefficient'] = premiums_df['coefficient'].abs()
        premiums_df = premiums_df.sort_values('abs_coefficient', ascending=False)
        
        return premiums_df
    
    def _extract_interaction_effects(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        coefficients: Dict
    ) -> pd.DataFrame:
        """Extract interaction effects"""
        interactions = []
        
        for feature, coef in coefficients.items():
            if '×' in feature and coef != 0:
                skills = feature.split('×')
                if len(skills) == 2:
                    skill1, skill2 = skills
                    
                    # Get individual coefficients
                    coef1 = coefficients.get(skill1, 0)
                    coef2 = coefficients.get(skill2, 0)
                    
                    # Calculate synergy
                    if abs(coef1) + abs(coef2) > 0:
                        synergy = coef / (abs(coef1) + abs(coef2))
                    else:
                        synergy = 0
                    
                    interaction_data = {
                        'skill1': skill1,
                        'skill2': skill2,
                        'interaction_coefficient': coef,
                        'synergy': synergy,
                        'synergy_category': self._categorize_synergy(synergy),
                        'prevalence': X[feature].mean(),
                        'sample_size': X[feature].sum(),
                        'avg_combo_salary': y[X[feature] == 1].mean() if X[feature].sum() > 0 else np.nan
                    }
                    interactions.append(interaction_data)
        
        if not interactions:
            return pd.DataFrame()
        
        interactions_df = pd.DataFrame(interactions)
        interactions_df['abs_synergy'] = interactions_df['synergy'].abs()
        interactions_df = interactions_df.sort_values('abs_synergy', ascending=False)
        
        return interactions_df
    
    def _extract_location_insights(
        self,
        merged_data: Dict,
        coefficients: Dict,
        X: pd.DataFrame
    ) -> Dict:
        """Extract location-specific insights"""
        insights = {
            'adjustment_factors': self.location_adjustments,
            'top_locations': [],
            'skill_location_variation': []
        }
        
        if self.location_column in merged_data['X'].columns:
            # Top paying locations
            location_stats = merged_data.get('location_stats', pd.DataFrame())
            if not location_stats.empty:
                insights['top_locations'] = location_stats.nlargest(5, 'median').to_dict('records')
            
            # Skill premium variation by location
            top_skills = [
                feature for feature in coefficients.keys() 
                if '×' not in feature and coefficients[feature] != 0
            ][:5]
            
            for skill in top_skills:
                if skill in X.columns:
                    location_premiums = []
                    
                    for location in merged_data['X'][self.location_column].unique():
                        loc_mask = merged_data['X'][self.location_column] == location
                        skill_mask = X[skill] == 1
                        
                        if (loc_mask & skill_mask).sum() >= 3:
                            skill_salary = merged_data['y_original'][loc_mask & skill_mask].median()
                            overall_salary = merged_data['y_original'][loc_mask].median()
                            
                            if overall_salary > 0:
                                premium_pct = ((skill_salary - overall_salary) / overall_salary) * 100
                                location_premiums.append({
                                    'location': location,
                                    'premium_pct': premium_pct,
                                    'sample_size': (loc_mask & skill_mask).sum()
                                })
                    
                    if location_premiums:
                        location_premiums.sort(key=lambda x: x['premium_pct'], reverse=True)
                        insights['skill_location_variation'].append({
                            'skill': skill,
                            'top_location': location_premiums[0]['location'],
                            'top_premium': location_premiums[0]['premium_pct'],
                            'variation': max(p['premium_pct'] for p in location_premiums) - min(p['premium_pct'] for p in location_premiums) if len(location_premiums) > 1 else 0
                        })
        
        return insights
    
    def _generate_recommendations(
        self,
        skill_premiums: pd.DataFrame,
        interaction_effects: pd.DataFrame,
        location_insights: Dict
    ) -> Dict:
        """Generate actionable recommendations"""
        recommendations = {
            'skill_development': [],
            'skill_combinations': [],
            'location_strategy': []
        }
        
        # Skill development recommendations
        if not skill_premiums.empty:
            top_skills = skill_premiums[skill_premiums['coefficient'] > 0].head(5)
            for _, skill in top_skills.iterrows():
                recommendations['skill_development'].append({
                    'skill': skill['skill'],
                    'reason': f"Adds ${skill['coefficient']:,.0f} to salary",
                    'prevalence': f"{skill['prevalence']:.1%}",
                    'priority': 'High' if skill['coefficient'] > 10000 else 'Medium'
                })
        
        # Skill combination recommendations
        if not interaction_effects.empty:
            top_interactions = interaction_effects[interaction_effects['synergy'] > 0.1].head(3)
            for _, interaction in top_interactions.iterrows():
                recommendations['skill_combinations'].append({
                    'combination': f"{interaction['skill1']} + {interaction['skill2']}",
                    'synergy': f"{interaction['synergy']:.0%}",
                    'reason': f"Creates ${interaction['interaction_coefficient']:,.0f} extra value",
                    'sample_size': interaction['sample_size']
                })
        
        # Location strategy
        if 'top_locations' in location_insights and location_insights['top_locations']:
            for loc in location_insights['top_locations'][:3]:
                recommendations['location_strategy'].append({
                    'location': loc['location'],
                    'median_salary': f"${loc['median']:,.0f}",
                    'action': 'Target for high-value hires',
                    'sample_size': loc['count']
                })
        
        return recommendations
    
    def _predict_salary(
        self,
        skills: List[str],
        location: Optional[str] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """Predict salary for given skills and location"""
        if self.model is None:
            raise ValueError("Model not trained")
        
        if feature_names is None:
            feature_names = self.model.feature_names_in_
        
        # Create feature vector
        features = {}
        for feat in feature_names:
            if '×' in feat:
                # Interaction term
                skill1, skill2 = feat.split('×')
                features[feat] = 1 if (skill1 in skills and skill2 in skills) else 0
            else:
                features[feat] = 1 if feat in skills else 0
        
        # Ensure all features present
        X_new = pd.DataFrame([features])
        missing = set(feature_names) - set(X_new.columns)
        for feat in missing:
            X_new[feat] = 0
        X_new = X_new[feature_names]
        
        # Scale and predict
        X_scaled = self.scaler.transform(X_new)
        adjusted_salary = self.model.predict(X_scaled)[0]
        
        # Adjust for location if specified
        if location and location in self.location_adjustments:
            adj_factor = self.location_adjustments[location]['adjustment_factor']
            local_salary = adjusted_salary / adj_factor if adj_factor != 0 else adjusted_salary
        else:
            local_salary = adjusted_salary
        
        return {
            'location_adjusted_salary': float(adjusted_salary),
            'local_salary': float(local_salary),
            'skills': skills,
            'location': location if location else 'national_average'
        }
    
    def _categorize_premium(self, premium: float) -> str:
        """Categorize skill premium"""
        if premium > 20000: return "Exceptional"
        if premium > 10000: return "High"
        if premium > 5000: return "Medium"
        if premium > 0: return "Low"
        if premium > -5000: return "Neutral"
        return "Penalty"
    
    def _categorize_synergy(self, synergy: float) -> str:
        """Categorize synergy"""
        if synergy > 0.3: return "Exceptional Synergy"
        if synergy > 0.2: return "High Synergy"
        if synergy > 0.1: return "Moderate Synergy"
        if synergy > 0: return "Mild Synergy"
        return "Negative"
    
    def save_results(self, output_path: str):
        """Save analysis results to file"""
        if not self.results:
            raise ValueError("No results to save. Run analyze_from_pivot first.")
        
        # Prepare results for serialization
        results_to_save = {}
        for key, value in self.results.items():
            if key == 'predictor':
                continue  # Skip predictor function
            elif isinstance(value, pd.DataFrame):
                results_to_save[key] = value.to_dict('records')
            else:
                results_to_save[key] = value
        
        # Save to JSON
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2, default=str)
        
        print(f" Results saved to {output_path}")
    
    def load_results(self, input_path: str):
        """Load analysis results from file"""
        with open(input_path, 'r') as f:
            results = json.load(f)
        
        # Convert DataFrames back from dict
        for key, value in results.items():
            if isinstance(value, list) and len(value) > 0 and isinstance(value[0], dict):
                results[key] = pd.DataFrame(value)
        
        self.results = results
        print(f"Results loaded from {input_path}")
        return results


# ============================================================================
# INTEGRATION WITH YOUR EXISTING PIPELINE
# ============================================================================

def integrate_with_pipeline():
    """Example of integrating with your existing pipeline"""
    from src.core.data_processor import DataProcessor
    from src.analyzers.skill_analyzer import SkillAnalyzer
    
    # 1. Load and process data using your existing classes
    processor = DataProcessor()
    df = processor.load_cleaned_data()
    
    # 2. Create skill pivot table (using your existing method)
    skill_pivot_df = processor.create_skill_pivot(df)
    
    # 3. Initialize salary analyzer
    salary_analyzer = SalarySkillRegressionAnalyzer(
        salary_column='salary',  # Your salary column name
        location_column='country',  # Your location column
        min_samples_per_skill=5,
        max_interaction_features=150
    )
    
    # 4. Run analysis
    results = salary_analyzer.analyze_from_pivot(
        original_df=df,
        skill_pivot_df=skill_pivot_df,
        adjust_for_location=True,
        reference_location='United States',  # Or your reference country
        include_interactions=True
    )
    
    # 5. Save results
    salary_analyzer.save_results('salary_skill_analysis.json')
    
    # 6. Example predictions
    example_skills = ['python', 'machine learning', 'sql']
    
    # National average prediction
    national_pred = salary_analyzer.results['predictor'](example_skills)
    print(f"National average: ${national_pred['location_adjusted_salary']:,.0f}")
    
    # Location-specific prediction
    if 'United Kingdom' in salary_analyzer.location_adjustments:
        uk_pred = salary_analyzer.results['predictor'](example_skills, 'United Kingdom')
        print(f"UK adjusted: ${uk_pred['local_salary']:,.0f}")
    
    return results


# ============================================================================
# QUICK ANALYSIS FUNCTION FOR YOUR NOTEBOOKS
# ============================================================================

def quick_salary_analysis(
    df: pd.DataFrame,
    skill_pivot_df: pd.DataFrame,
    salary_col: str = 'salary',
    location_col: str = 'country'
) -> Dict:
    """
    Quick one-function analysis for notebooks
    
    Returns dictionary with key insights
    """
    analyzer = SalarySkillRegressionAnalyzer(
        salary_column=salary_col,
        location_column=location_col,
        min_samples_per_skill=5
    )
    
    results = analyzer.analyze_from_pivot(
        original_df=df,
        skill_pivot_df=skill_pivot_df,
        adjust_for_location=True,
        include_interactions=True
    )
    
    # Extract key insights
    key_insights = {
        'top_skills': results['skill_premiums'].head(10)[['skill', 'coefficient', 'premium_category']].to_dict('records'),
        'top_interactions': results['interaction_effects'].head(5)[['skill1', 'skill2', 'synergy']].to_dict('records'),
        'model_performance': results['model_performance'],
        'predict': analyzer._predict_salary  # Attach prediction function
    }
    
    return key_insights


if __name__ == "__main__":
    # Example usage
    print("Testing SalarySkillRegressionAnalyzer")
    print("=" * 60)
    
    # This would be your actual data loading
    # results = integrate_with_pipeline()
    
    print("✅ Analyzer ready for integration with your pipeline")