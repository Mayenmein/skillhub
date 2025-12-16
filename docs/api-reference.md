# SkillHub API Reference

## DataScienceJobsAnalyzer

Core analysis engine for processing job market data and generating insights.

### Constructor

```python
DataScienceJobsAnalyzer(data_dir: str = "../data")
```

**Parameters:**
- `data_dir`: Base directory for data files

### Core Analysis Methods

#### analyze_skill_trends_full

```python
def analyze_skill_trends_full(self, pivot_df: pd.DataFrame) -> pd.DataFrame:
    """
    Comprehensive trend analysis with smart classification
    
    Parameters:
        pivot_df: DataFrame with columns [date, skill, mentions]
        
    Returns:
        DataFrame with trend metrics and classifications
    """
```

#### analyze_skill_ecosystem

```python
def analyze_skill_ecosystem(self, pivot_df: pd.DataFrame) -> Dict:
    """
    Analyze complete skill ecosystem
    
    Parameters:
        pivot_df: DataFrame with skills data
        
    Returns:
        Dict with ecosystem insights
    """
```

#### analyze_skill_progression_data

```python
def analyze_skill_progression_data(self, pivot_df: pd.DataFrame, skills_list: List[str]) -> pd.DataFrame:
    """
    Track progression/adoption of specific skills
    
    Parameters:
        pivot_df: Skills pivot table
        skills_list: Skills to analyze
        
    Returns:
        DataFrame with progression metrics
    """
```

### Helper Functions

#### create_skill_pivot

```python
def create_skill_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Create pivot table for skill analysis
    
    Parameters:
        df: Raw job postings data
        
    Returns:
        Pivoted DataFrame
    """
```

#### aggregate_pivot

```python
def aggregate_pivot(
    self, 
    filtered_df: pd.DataFrame, 
    column: str = "skill", 
    metric: str = "mentions"
) -> pd.DataFrame:
    """
    Aggregate pivot data with vectorized operations
    
    Parameters:
        filtered_df: Filtered DataFrame
        column: Column to aggregate by
        metric: Metric to calculate
        
    Returns:
        Aggregated DataFrame
    """
```

## Dashboard Components

### EnhancedDataScienceJobsDashboard

Main dashboard class with visualization methods.

#### Constructor

```python
EnhancedDataScienceJobsDashboard(data_dir: str = "data")
```

#### Page Rendering Methods

```python
def render_home(self):
    """Render home page with overview"""

def render_skills(self):
    """Render skills analysis page"""

def render_trends(self):
    """Render temporal trends page"""

def render_role_comparison(self):
    """Render role comparison page"""

def render_salary(self):
    """Render salary analysis page"""
```

### Sidebar Controls

#### setup_enhanced_sidebar

```python
def setup_enhanced_sidebar(df: pd.DataFrame) -> dict:
    """
    Create sidebar with interactive filters
    
    Parameters:
        df: DataFrame to filter
        
    Returns:
        Dict of selected filters
    """
```

## Data Processing Functions

### Trend Analysis

#### exp_smooth

```python
@njit(fastmath=True, parallel=True)
def exp_smooth(y: np.ndarray, alpha: float) -> np.ndarray:
    """
    Exponential smoothing with Numba acceleration
    
    Parameters:
        y: Input time series
        alpha: Smoothing factor
        
    Returns:
        Smoothed series
    """
```

#### calculate_vectorized_metrics

```python
@njit(fastmath=True, parallel=True)
def calculate_vectorized_metrics(
    values: np.ndarray,
    dates: np.ndarray
) -> Tuple[float, float, float, float]:
    """
    Calculate trend metrics with vectorization
    
    Parameters:
        values: Time series values
        dates: Corresponding dates
        
    Returns:
        Tuple of (cagr, momentum, prevalence, peak_ratio)
    """
```

## Constants & Configuration

### TREND_MAP

```python
TREND_MAP = {
    "Emerging": {
        "description": "High growth, increasing adoption",
        "conditions": ["high_momentum", "rising_trend"]
    },
    "Growing": {
        "description": "Steady growth, stable demand",
        "conditions": ["positive_momentum", "stable_trend"]
    },
    "Stable": {
        "description": "Consistent presence, core skill",
        "conditions": ["high_prevalence", "flat_trend"]
    },
    "Declining": {
        "description": "Decreasing demand",
        "conditions": ["negative_momentum", "falling_trend"]
    }
}
```

### SKILL_CATEGORIES

```python
SKILL_CATEGORIES = {
    "Programming": ["python", "r", "sql", "java", ...],
    "Machine Learning": ["tensorflow", "pytorch", "scikit-learn", ...],
    "Cloud": ["aws", "azure", "gcp", ...],
    "Big Data": ["hadoop", "spark", "kafka", ...],
    # ... other categories
}
```

## Error Handling

Most methods include error handling for:
- Empty/invalid DataFrames
- Missing columns
- Invalid date formats
- Type mismatches

Example:
```python
def safe_method(self, df: pd.DataFrame) -> pd.DataFrame:
    """Example error handling pattern"""
    if df.empty:
        logger.warning("Empty DataFrame provided")
        return pd.DataFrame()
        
    required_cols = ["col1", "col2"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"Missing required columns: {required_cols}")
        
    try:
        # Process data
        return result
    except Exception as e:
        logger.error(f"Processing failed: {str(e)}")
        raise
```