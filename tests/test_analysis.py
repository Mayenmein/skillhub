import os
import sys
from pathlib import Path
from collections import Counter

import pytest
import pandas as pd
import numpy as np


# Add the src directory to Python path to import the module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.analyze_jobs import exp_smooth, calculate_vectorized_metrics, DataScienceJobsAnalyzer


# ---------- BASIC NUMBA FUNCTION TESTS ----------

def test_exp_smooth_basic():
    y = np.array([[1.0, 2.0, 3.0, 4.0]])
    alpha = 0.5
    result = exp_smooth(y, alpha)
    assert result.shape == y.shape
    # manually verify smoothing logic
    expected = [1.0, 1.5, 2.25, 3.125]
    np.testing.assert_allclose(result[0], expected, rtol=1e-6)


def test_calculate_vectorized_metrics_shapes():
    y = np.array([[1, 2, 3, 4, 5],
                  [2, 2, 2, 2, 2]], dtype=np.float64)
    cagr, momentum, curr, peak = calculate_vectorized_metrics(y)
    n = y.shape[0]
    for arr in [cagr, momentum, curr, peak]:
        assert arr.shape == (n,)


def test_calculate_vectorized_metrics_values():
    y = np.array([[1, 2, 4, 8, 16]], dtype=np.float64)
    cagr, momentum, curr, peak = calculate_vectorized_metrics(y)
    assert cagr[0] > 0
    assert momentum[0] >= 0
    assert curr[0] == 16
    assert 0 < peak[0] <= 16


# ---------- FIXTURE FOR ANALYZER AND MOCK DATA ----------

@pytest.fixture
def mock_analyzer(tmp_path):
    data_dir = tmp_path / "data"
    (data_dir / "interim").mkdir(parents=True)
    (data_dir / "processed").mkdir()
    (data_dir.parent / "reports" / "figures").mkdir(parents=True)
    return DataScienceJobsAnalyzer(data_dir=str(data_dir))


@pytest.fixture
def mock_df():
    return pd.DataFrame({
        "country": ["US", "US", "UK", "UK"],
        "company": ["A", "B", "A", "C"],
        "cleaned_title_category": ["Data Scientist"] * 4,
        "seniority_level": ["Junior", "Mid-level", "Senior", "Junior"],
        "skills": [["Python", "SQL"], ["Python"], ["R", "Spark"], ["SQL"]],
        "published_year": [2025, 2025, 2025, 2025],
        "published_month": [1, 2, 3, 4],
        "primary_job_type": ["Full-Time"] * 4
    })


# ---------- CORE DATA PROCESSING TESTS ----------

def test_create_skill_pivot(mock_analyzer, mock_df):
    df = mock_analyzer.create_skill_pivot(mock_df)
    assert not df.empty
    assert "prevalence" in df.columns
    assert "skill" in df.columns
    assert (df["prevalence"] >= 0).all()


def test_aggregate_pivot(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    agg = mock_analyzer.aggregate_pivot(pivot, column="skill")
    assert not agg.empty
    assert "prevalence" in agg.columns
    assert agg["mentions"].sum() > 0


# ---------- ANALYSIS LAYERS TESTS ----------

def test_analyze_skill_frequency(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    freq = mock_analyzer.analyze_skill_frequency(pivot)
    assert isinstance(freq, pd.DataFrame)
    assert "skill" in freq.columns


def test_analyze_by_group(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    result = mock_analyzer.analyze_by_group(pivot, "seniority_level")
    assert isinstance(result, dict)
    assert all(isinstance(df, pd.DataFrame) for df in result.values())


def test_analyze_skill_trends_optimized(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    df = mock_analyzer.analyze_skill_trends_optimized(pivot)
    assert isinstance(df, pd.DataFrame)
    if not df.empty:
        assert "trend_category" in df.columns
        assert set(df["trend_category"]).issubset(set(mock_analyzer.trend_colors.keys()) | {"Stable"})


def test_get_trend_summary(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    trend_df = mock_analyzer.analyze_skill_trends_optimized(pivot)
    summary = mock_analyzer.get_trend_summary(trend_df)
    assert isinstance(summary, pd.DataFrame)


# ---------- ROLE & SENIORITY ANALYSIS ----------

def test_analyze_skills_by_role(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    results = mock_analyzer.analyze_skills_by_role(pivot)
    assert isinstance(results, dict)
    assert all(isinstance(df, pd.DataFrame) for df in results.values())


def test_analyze_skill_progression_data(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    skills = pivot["skill"].unique()[:3].tolist()
    df = mock_analyzer.analyze_skill_progression_data(pivot, skills)
    assert isinstance(df, pd.DataFrame)
    assert "prevalence" in df.columns


# ---------- CLUSTER & ECOSYSTEM TESTS ----------

def test_prepare_skill_combinations_fast(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    combos = mock_analyzer.prepare_skill_combinations_fast(pivot, min_mentions=1, top_n=5)
    assert isinstance(combos, pd.DataFrame)
    if not combos.empty:
        assert {"skill_1", "skill_2"}.issubset(combos.columns)


def test_analyze_skill_ecosystem(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    eco = mock_analyzer.analyze_skill_ecosystem(pivot)
    assert isinstance(eco, dict)
    assert "top_combinations" in eco
    assert "natural_clusters" in eco


# ---------- VISUALIZATION TESTS (sanity only, no figure assertions) ----------

def test_plot_trend_categories_distribution(mock_analyzer, mock_df):
    pivot = mock_analyzer.create_skill_pivot(mock_df)
    results = mock_analyzer.analyze_skill_trends_optimized(pivot)
    fig = mock_analyzer.plot_trend_categories_distribution(results)
    assert fig is None or hasattr(fig, "axes")


# allow running pytest directly
if __name__ == "__main__":
    pytest.main([__file__, "-q"])