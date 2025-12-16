# test_clean_jobs.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import os
import sys
import torch
from unittest.mock import patch, MagicMock

import joblib

# Add the src directory to Python path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.cleaning.clean_jobs import DataScienceJobsCleaner, HybridJobTitleClassifier, SkillEnhancer


class TestDataScienceJobsCleaner:
    """Test cases for DataScienceJobsCleaner class"""
    
    @pytest.fixture
    def cleaner(self):
        """Fixture to create a cleaner instance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield DataScienceJobsCleaner(data_dir=Path(temp_dir))
    
    @pytest.fixture
    def sample_raw_data(self):
        """Fixture with sample raw data for testing"""
        return pd.DataFrame({
            'title': ['Data Scientist', 'Machine Learning Engineer', 'Data Analyst'],
            'country': ['USA', 'United Kingdom', 'Germany'],
            'type': ['full-time', 'part-time', 'contract'],
            'salary': ['$100,000', '£50,000', '€75,000'],
            'published': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'skills': ['[python, sql]', '[python, tensorflow]', '[excel, sql]'],
            'salary_min': [90000, 40000, 60000],
            'salary_max': [110000, 50000, 80000]
        })
    
    @pytest.fixture
    def sample_csv_file(self, sample_raw_data):
        """Fixture to create a temporary CSV file"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_raw_data.to_csv(f.name, index=False)
            yield f.name
        os.unlink(f.name)

    def test_init(self, cleaner):
        """Test initialization with custom data directory"""
        temp_dir = Path(tempfile.mkdtemp())
        cleaner = DataScienceJobsCleaner(data_dir=temp_dir)
        
        assert cleaner.data_dir == temp_dir
        assert cleaner.raw_dir == temp_dir / "raw"
        assert cleaner.interim_dir == temp_dir / "interim"
        assert cleaner.processed_dir == temp_dir / "processed"
        
        # Test directory creation
        assert cleaner.interim_dir.exists()
        assert cleaner.processed_dir.exists()

    def test_load_raw_data(self, cleaner, sample_csv_file):
        """Test loading raw data from CSV file"""
        df = cleaner.load_raw_data(Path(sample_csv_file))
        
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 3
        assert 'title' in df.columns
        assert 'country' in df.columns

    def test_load_raw_data_file_not_found(self, cleaner):
        """Test loading with non-existent file"""
        with pytest.raises(FileNotFoundError):
            cleaner.load_raw_data(Path("nonexistent_file.csv"))

    def test_clean_location_data(self, cleaner, sample_raw_data):
        """Test location data cleaning"""
        cleaned_df = cleaner.clean_location_data(sample_raw_data)
        
        assert 'country' in cleaned_df.columns
        # Test country standardization
        assert cleaned_df['country'].iloc[0] == 'USA'
        assert cleaned_df['country'].iloc[1] == 'United Kingdom'
        assert cleaned_df['country'].iloc[2] == 'Germany'

    def test_clean_location_data_edge_cases(self, cleaner):
        """Test location cleaning with edge cases"""
        edge_cases = pd.DataFrame({
            'country': [None, 'Unknown', 'US', 'FR', 'JP', 'Some Unknown Country']
        })
        
        cleaned_df = cleaner.clean_location_data(edge_cases)
        
        # Check that None becomes 'Unknown'
        assert cleaned_df['country'].iloc[0] == 'Unknown'
        # Check standardization works
        assert cleaned_df['country'].iloc[2] == 'USA'
        assert cleaned_df['country'].iloc[3] == 'France'

    def test_clean_job_type(self, cleaner, sample_raw_data):
        """Test job type cleaning and categorization"""
        
        cleaned_df = cleaner.clean_job_type(sample_raw_data)
        
        assert 'cleaned_job_type' in cleaned_df.columns
        assert 'primary_job_type' in cleaned_df.columns
        
        # Test that job types are properly categorized
        assert 'full_time' in cleaned_df['cleaned_job_type'].iloc[0]
        assert 'part_time' in cleaned_df['cleaned_job_type'].iloc[1]
        assert 'contract' in cleaned_df['cleaned_job_type'].iloc[2]

    def test_clean_job_type_multilingual(self, cleaner):
        """Test job type cleaning with multilingual entries"""
        multilingual_data = pd.DataFrame({
            'type': [
                'vollzeit',  # German for full-time
                'deeltijds',  # Dutch for part-time
                'stage',  # French for internship
                'remoto',  # Spanish for remote
                'unknown_type'
            ]
        })
        
        cleaned_df = cleaner.clean_job_type(multilingual_data)
        
        assert 'full_time' in cleaned_df['cleaned_job_type'].iloc[0]
        assert 'part_time' in cleaned_df['cleaned_job_type'].iloc[1]
        assert 'internship' in cleaned_df['cleaned_job_type'].iloc[2]
        assert 'remote' in cleaned_df['cleaned_job_type'].iloc[3]

    def test_clean_salary_data(self, cleaner, sample_raw_data):
        """Test salary data cleaning and conversion"""
        # Add more salary test cases
        sample_raw_data['salary'] = ['$100,000', 'Competitive', None]
        sample_raw_data['country'] = ['USA', 'United Kingdom', 'Germany']
        
        cleaned_df = cleaner.clean_salary_data(sample_raw_data)
        
        assert 'salary_min_usd' in cleaned_df.columns
        assert 'salary_max_usd' in cleaned_df.columns
        assert 'salary_category' in cleaned_df.columns
        
        # Test that numeric salaries are converted
        assert not pd.isna(cleaned_df['salary_min_usd'].iloc[0])
        assert not pd.isna(cleaned_df['salary_max_usd'].iloc[0])
        
        # Test that non-numeric salaries become NaN
        assert pd.isna(cleaned_df['salary_min_usd'].iloc[1])
        assert pd.isna(cleaned_df['salary_max_usd'].iloc[2])

    def test_clean_salary_data_edge_cases(self, cleaner):
        """Test salary cleaning with edge cases"""
        edge_cases = pd.DataFrame({
            'salary': [
                '$100k',  # k notation
                '50,000 - 70,000 USD',  # range with currency
                'Market rate',  # non-numeric
                '100000',  # plain number
                '50€ per hour',  # hourly rate
                '5000 monthly'  # monthly rate
            ],
            'salary_min': [100_000, 50_000,None,100_000,50,5000],
            'salary_max': [100_000,70_000,None,100_000,50,5000],
            'country': ['USA', 'USA', 'USA', 'USA', 'Germany', 'France']
        })
        
        cleaned_df = cleaner.clean_salary_data(edge_cases)
        
        # Should handle various formats without crashing
        assert len(cleaned_df) == 6
        assert 'salary_min_usd' in cleaned_df.columns

    def test_convert_dates(self, cleaner, sample_raw_data):
        """Test date conversion functionality"""
        cleaned_df = cleaner.convert_dates(sample_raw_data)
        
        assert 'published_year' in cleaned_df.columns
        assert 'published_month' in cleaned_df.columns
        
        # Test that dates are properly converted
        assert cleaned_df['published_year'].iloc[0] == 2023
        assert cleaned_df['published_month'].iloc[0] == 1
        
        # Test datetime conversion
        assert pd.api.types.is_datetime64_any_dtype(cleaned_df['published'])

    def test_convert_dates_invalid(self, cleaner):
        """Test date conversion with invalid dates"""
        invalid_dates = pd.DataFrame({
            'published': ['invalid_date', '2023-13-01', None, '2023-01-01']
        })
        
        cleaned_df = cleaner.convert_dates(invalid_dates)
        
        # Should handle invalid dates gracefully
        assert pd.isna(cleaned_df['published'].iloc[0])  # invalid becomes NaT
        assert pd.isna(cleaned_df['published'].iloc[1])  # invalid month becomes NaT
        assert pd.isna(cleaned_df['published'].iloc[2])  # None becomes NaT

    def test_quality_check(self, cleaner, sample_raw_data):
        """Test data quality check functionality"""
        # Intentionally create some missing data
        sample_raw_data.loc[0, 'title'] = None
        sample_raw_data.loc[1, 'salary'] = None
        
        # This should run without errors and log missing data
        cleaner._quality_check(sample_raw_data)

    def test_save_cleaned_data(self, cleaner, sample_raw_data):
        """Test saving cleaned data"""
        # Test interim save
        cleaner.save_cleaned_data(sample_raw_data, "interim")
        interim_path = cleaner.interim_dir / "cleaned_jobs.csv"
        assert interim_path.exists()
        
        # Test processed save
        cleaner.save_cleaned_data(sample_raw_data, "processed")
        processed_path = cleaner.processed_dir / "processed_jobs.csv"
        assert processed_path.exists()
        
        # Test invalid output type
        with pytest.raises(ValueError):
            cleaner.save_cleaned_data(sample_raw_data, "invalid_type")

    @patch.object(DataScienceJobsCleaner, 'load_raw_data')
    def test_run_full_cleaning_pipeline(self, mock_load, cleaner, sample_raw_data):
        """Test full cleaning pipeline"""
        mock_load.return_value = sample_raw_data
        
        # Mock the methods that are called in the pipeline but not defined
        with patch.object(cleaner, 'clean_location_data') as mock_location, \
             patch.object(cleaner, 'clean_job_type') as mock_job_type,\
             patch.object(cleaner, 'clean_salary_data') as mock_salary, \
             patch.object(cleaner, 'convert_dates') as mock_dates:
            
            mock_location.return_value = sample_raw_data
            mock_job_type.return_value = sample_raw_data
            mock_dates.return_value = sample_raw_data
            mock_salary.return_value = sample_raw_data
            
            result = cleaner.run_full_cleaning_pipeline()
            
            assert isinstance(result, pd.DataFrame)
            mock_load.assert_called_once()
            mock_salary.assert_called_once()
            mock_location.assert_called_once()
            mock_job_type.assert_called_once()
            mock_dates.assert_called_once()


class TestHybridJobTitleClassifier:
    """Test suite for HybridJobTitleClassifier."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Setup shared resources for all tests."""
        self.cluster_path = tmp_path / "clusters.pkl"
        self.cat_path = tmp_path / "categories.pkl"

        # Create a fake SentenceTransformer mock
        self.mock_model = MagicMock()
        self.mock_model.encode.side_effect = (
            lambda x, normalize_embeddings=True: np.array(
                [[i + j for j in range(3)] for i, _ in enumerate(x)]
            ) if isinstance(x, list) else np.array([1.0, 0.0, 0.0])
        )

    @pytest.fixture
    def classifier(self):
        """Patch SentenceTransformer so init doesn't download a model."""
        with patch("src.clean_jobs.SentenceTransformer", return_value=self.mock_model):
            clf = HybridJobTitleClassifier(
                model_name="mock-model",
                cluster_path=str(self.cluster_path),
                category_embeddings_path=str(self.cat_path)
            )
            clf.model = self.mock_model
            clf.category_embeddings = np.random.rand(len(clf.categories), 3)
            return clf

    def test_extract_seniority(self, classifier):
        assert classifier.extract_seniority("Senior Data Scientist") == "Senior"
        assert classifier.extract_seniority("Lead AI Engineer") == "Lead"
        assert classifier.extract_seniority("Intern Data Analyst") == "Intern"
        assert classifier.extract_seniority("Data Engineer") == "Unspecified"
        assert classifier.extract_seniority(None) == "Unknown"

    @patch("src.clean_jobs.KMeans")
    @patch("src.clean_jobs.util")
    def test_classify_dataframe(self, mock_util, mock_kmeans, classifier):
        df = pd.DataFrame({
            "title": ["Senior Data Scientist", "Junior ML Engineer", "Random AI Wizard"]
        })

        # Mock similarity & clustering
        mock_util.cos_sim.side_effect = lambda a, b: torch.tensor(np.random.rand(len(a), len(b)))
        mock_kmeans_instance = MagicMock()
        mock_kmeans_instance.fit_predict.return_value = np.array([0, 1])
        mock_kmeans_instance.cluster_centers_ = np.random.rand(2, 3)
        mock_kmeans.return_value = mock_kmeans_instance

        result = classifier.classify_dataframe(df, "title", n_clusters=2)

        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(df)
        assert {"cleaned_title_category", "seniority_level", "similarity_score"}.issubset(result.columns)
        assert result["cleaned_title_category"].notna().all()

    def test_save_state(self, classifier):
        classifier.clusters = {"Cluster A": np.array([1, 2, 3])}
        with patch("joblib.dump") as mock_dump:
            classifier.save_state()
            assert mock_dump.call_count == 2
            args1, args2 = mock_dump.call_args_list
            assert isinstance(args1[0][0], dict)
            assert isinstance(args2[0][0], np.ndarray)

    def test_load_existing_clusters(self, tmp_path):
        cluster_path = tmp_path / "clusters.pkl"
        joblib.dump({"existing_cluster": np.array([1, 2, 3])}, cluster_path)

        with patch("src.clean_jobs.SentenceTransformer", return_value=self.mock_model):
            clf = HybridJobTitleClassifier(cluster_path=str(cluster_path))
            assert "existing_cluster" in clf.clusters

class TestSkillEnhancer:
    """Test cases for SkillEnhancer class"""
    
    def test_parse_skills_valid_list_string(self):
        """Test parsing valid list-like skill strings"""
        test_cases = [
            ("['python', 'sql', 'tensorflow']", ['Python', 'Sql', 'Tensorflow']),
            ('["python", "machine learning"]', ['Python', 'Machine Learning']),
            ('python,sql,r', ['Python', 'Sql', 'R']),
            ('python, sql, r', ['Python', 'Sql', 'R']),
        ]
        
        for input_str, expected in test_cases:
            result = SkillEnhancer.parse_skills(input_str)
            assert set(result) == set(expected)  # Order doesn't matter

    def test_parse_skills_edge_cases(self):
        """Test parsing edge cases"""
        test_cases = [
            (None, []),
            ("", []),
            ("   ", []),
            ("[]", []),
            ("['']", []),
            ("['Data Science']", []),  # Should filter out generic skills
            ("['Python', 'Data Science', 'SQL']", ['Python', 'Sql']),  # Filter generic
        ]
        
        for input_str, expected in test_cases:
            result = SkillEnhancer.parse_skills(input_str)
            assert set(result) == set(expected)

    def test_parse_skills_malformed(self):
        """Test parsing malformed skill strings"""
        result = SkillEnhancer.parse_skills("['python, 'sql")  # Malformed list
        assert isinstance(result, list)  # Should return empty list rather than crash

    def test_enhance_skills_data(self):
        """Test enhancing DataFrame with skills data"""
        sample_df = pd.DataFrame({
            'skills': [
                "['python', 'sql']",
                "['tensorflow', 'pytorch']",
                "",
                "['data science', 'python']"  # Contains generic skill
            ],
            'other_col': [1, 2, 3, 4]
        })
        
        enhanced_df = SkillEnhancer.enhance_skills_data(sample_df)
        
        assert 'skills_count' in enhanced_df.columns
        assert enhanced_df['skills_count'].iloc[0] == 2
        assert enhanced_df['skills_count'].iloc[1] == 2
        assert enhanced_df['skills_count'].iloc[2] == 0
        assert enhanced_df['skills_count'].iloc[3] == 1  # One generic skill filtered
        
        # Check that skills are parsed as lists
        assert isinstance(enhanced_df['skills'].iloc[0], list)
        assert 'Python' in enhanced_df['skills'].iloc[0]
        
        # Check original columns preserved
        assert 'other_col' in enhanced_df.columns

    def test_enhance_skills_data_empty(self):
        """Test that empty DataFrame stays empty and gains expected columns."""
        empty_df = pd.DataFrame({'skills': []})  

        enhanced_df = SkillEnhancer.enhance_skills_data(empty_df)

        assert enhanced_df.empty
        assert 'skills_count' in enhanced_df.columns
        assert enhanced_df['skills'].tolist() == []

# Integration tests
class TestIntegration:
    """Integration tests for the complete pipeline"""
    
    @pytest.fixture
    def complete_sample_data(self):
        """Complete sample data for integration testing"""
        return pd.DataFrame({
            'title': ['Senior Data Scientist', 'ML Engineer', 'Data Analyst'],
            'country': ['USA', 'UK', 'Germany'],
            'type': ['full-time', 'contract', 'part-time'],
            'salary': ['$120,000', '£80,000', '€60,000'],
            'published': ['2023-01-01', '2023-02-01', '2023-03-01'],
            'skills': ["['python', 'sql', 'ml']", "['tensorflow', 'python']", "['excel', 'sql']"],
            'salary_min': [110000, 70000, 50000],
            'salary_max': [130000, 90000, 70000],
            'batch_source': ['source1', 'source2', 'source3'],
            'ai': [True, False, True],
            'published_dt': ['2023-01-01', '2023-02-01', '2023-03-01']
        })
    
    @pytest.fixture
    def cleaner(self):
        """Fixture to create a cleaner instance"""
        with tempfile.TemporaryDirectory() as temp_dir:
            yield DataScienceJobsCleaner(data_dir=Path(temp_dir))

    def test_end_to_end_cleaning(self, cleaner, complete_sample_data, tmp_path):
        """Test complete cleaning pipeline with real data"""
        # Save sample data to file
        input_file = tmp_path / "test_input.csv"
        complete_sample_data.to_csv(input_file, index=False)
        
        # Load and clean
        df = cleaner.load_raw_data(input_file)
        df = cleaner.clean_location_data(df)
        df = cleaner.clean_job_type(df)
        df = cleaner.clean_salary_data(df)
        df = cleaner.convert_dates(df)
        
        # Verify results
        assert len(df) == 3
        assert 'country' in df.columns
        assert 'cleaned_job_type' in df.columns
        assert 'salary_min_usd' in df.columns
        assert 'salary_max_usd' in df.columns
        assert 'published_year' in df.columns
        
        # Verify data quality
        assert not df['country'].isna().all()
        assert df['salary_min_usd'].dtype in [np.float64, np.int64]

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])