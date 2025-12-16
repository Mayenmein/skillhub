# SkillHub Developer Guide

## 🛠️ Development Setup

1. **Environment Setup**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dev dependencies
pip install -r requirements-dev.txt
```

2. **Pre-commit Hooks**
```bash
pre-commit install
```

## 🏗️ Architecture

### Core Components

1. **Data Pipeline**
   - `src/scrape_jobs.py`: Job data collection
   - `src/clean_jobs.py`: Data cleaning and preprocessing
   - `src/analyze_jobs.py`: Analysis and insights generation

2. **Dashboard**
   - `dashboard/app.py`: Main Streamlit application
   - `dashboard/pages/`: Individual dashboard pages
   - `dashboard/sidebar.py`: Shared filter controls

### Key Classes

#### DataScienceJobsAnalyzer

Core analysis engine with methods for:
- Skill frequency analysis
- Trend detection and forecasting
- Role comparison and clustering
- Market health indicators

```python
class DataScienceJobsAnalyzer:
    def __init__(self, data_dir: str = "../data"):
        self.data_dir = Path(data_dir)
        # ... setup paths and config

    def analyze_skill_trends_full(self, pivot_df: pd.DataFrame) -> pd.DataFrame:
        """Comprehensive trend analysis with classification"""
        # ... implementation
```

## 🧪 Testing

1. **Running Tests**
```bash
pytest tests/
pytest tests/test_analysis.py -v  # specific file
pytest tests/ -k "test_skill"     # pattern matching
```

2. **Writing Tests**
```python
def test_analyze_skill_frequency(mock_analyzer, mock_df):
    result = mock_analyzer.analyze_skill_frequency(mock_df)
    assert isinstance(result, pd.DataFrame)
    assert "skill" in result.columns
    assert "mentions" in result.columns
```

## 📈 Performance Optimization

1. **Numba-accelerated Functions**
   - Use `@njit` for numeric computations
   - Parallelize with `parallel=True` when safe
   - Example: `exp_smooth()` for trend smoothing

2. **Data Processing**
   - Use vectorized operations
   - Avoid loops where possible
   - Cache expensive computations

## 🔄 Workflow

1. Create feature branch
2. Implement changes
3. Add tests
4. Run pre-commit hooks
5. Submit PR

## 📊 Adding Dashboard Pages

1. Create new file in `dashboard/pages/`
2. Use consistent page config:
```python
import streamlit as st
from dashboard.app import EnhancedDataScienceJobsDashboard

st.set_page_config(
    page_title="SkillHub • Page Name",
    page_icon="📊"
)

dashboard = EnhancedDataScienceJobsDashboard()
dashboard.render_page_name()  # use appropriate render method
```

## 🎨 Style Guide

- Follow PEP 8
- Use type hints
- Document public methods
- Keep functions focused
- Add docstrings for complex logic

## 🔍 Debugging

1. **Streamlit**
   - Use `st.write()` for quick debugging
   - Check session state with `st.session_state`
   - Enable debug mode: `streamlit run --debug`

2. **Data Pipeline**
   - Use logging module
   - Save intermediate results
   - Validate data shapes and types

## 📦 Deployment

1. **Local**
```bash
streamlit run dashboard/app.py
```

2. **Server**
```bash
# Setup virtual env
python -m venv prod-env
source prod-env/bin/activate
pip install -r requirements.txt

# Run with proper host/port
streamlit run dashboard/app.py --server.port 80 --server.address 0.0.0.0
```

## 🔮 Future Development

- [ ] Add role prediction model
- [ ] Implement salary forecasting
- [ ] Add geographic heat maps
- [ ] Enhance trend detection
- [ ] Add API endpoints