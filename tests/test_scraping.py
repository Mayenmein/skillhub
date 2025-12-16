import pytest
import pandas as pd
import os
from unittest.mock import Mock, patch, MagicMock
import sys
import os

# Add the src directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from src.scraping.scrape_jobs import JobScraper


class TestJobScraper:
    """Test suite for JobScraper."""

    @pytest.fixture(autouse=True)
    def setup(self, tmp_path):
        """Set up a temporary output file for all tests."""
        self.output_file = tmp_path / "jobs_data.csv"
        self.scraper = JobScraper(output_file=self.output_file)

    # -------------------------------
    # fetch_jobs
    # -------------------------------
    @patch("src.scrape_jobs.requests.get")
    def test_fetch_jobs_success(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {"jobs": [{"title": "Data Scientist"}]}
        mock_resp.raise_for_status.return_value = None
        mock_get.return_value = mock_resp

        result = self.scraper.fetch_jobs(page=1, skill="AI")
        assert "jobs" in result
        mock_get.assert_called_once()
        args, kwargs = mock_get.call_args
        assert kwargs["params"]["skill"] == "AI"

    @patch("src.scrape_jobs.requests.get")
    def test_fetch_jobs_error(self, mock_get):
        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("Server error")
        mock_get.return_value = mock_resp

        with pytest.raises(Exception):
            self.scraper.fetch_jobs(page=1)

    # -------------------------------
    # process_job_data
    # -------------------------------
    def test_process_job_data_valid(self):
        jobs_data = [
            {"job": {"title": "ML Engineer", "city": "Paris", "country": "France"}, "company": {"name": "TechCorp"}}
        ]
        processed = self.scraper.process_job_data(jobs_data)

        assert isinstance(processed, list)
        assert len(processed) == 1
        assert processed[0]["title"] == "ML Engineer"
        assert "location" in processed[0]
        assert processed[0]["location"] == "Paris, France"

    def test_process_job_data_empty(self):
        assert self.scraper.process_job_data([]) == []
        assert self.scraper.process_job_data(None) == []

    # -------------------------------
    # save_jobs
    # -------------------------------
    def test_save_jobs_creates_new_file(self):
        jobs = [{"title": "AI Engineer", "company": "OpenAI"}]
        count = self.scraper.save_jobs(jobs)

        assert count == 1
        assert os.path.exists(self.output_file)
        df = pd.read_csv(self.output_file)
        assert "title" in df.columns

    def test_save_jobs_appends_existing_file(self):
        # Create an initial file
        df = pd.DataFrame([{"title": "Old Job"}])
        df.to_csv(self.output_file, index=False)

        new_jobs = [{"title": "New Job"}]
        count = self.scraper.save_jobs(new_jobs)
        assert count == 1
        combined = pd.read_csv(self.output_file)
        assert len(combined) == 2

    def test_save_jobs_empty_input(self):
        assert self.scraper.save_jobs([]) == 0

    def test_save_jobs_io_error(self, monkeypatch):
        jobs = [{"title": "AI Engineer"}]
        monkeypatch.setattr(pd.DataFrame, "to_csv", lambda *_, **__: (_ for _ in ()).throw(IOError("write fail")))
        assert self.scraper.save_jobs(jobs) == 0

    # -------------------------------
    # scrape_in_batches
    # -------------------------------
    @patch("src.scrape_jobs.tqdm", autospec=True)
    @patch.object(JobScraper, "save_jobs", return_value=2)
    @patch.object(JobScraper, "process_job_data", return_value=[{"title": "Mock Job"}])
    @patch.object(JobScraper, "fetch_jobs")
    def test_scrape_in_batches_single_batch(self, mock_fetch, mock_process, mock_save, mock_tqdm):
        """Test one full batch cycle with mock data."""
        mock_fetch.return_value = {"jobs": [{"job": {"title": "AI Scientist"}}]}
        mock_tqdm.return_value.__enter__.return_value = mock_tqdm
        mock_tqdm.return_value.__exit__.return_value = None
        mock_tqdm.update = MagicMock()
        mock_tqdm.set_postfix = MagicMock()

        count = self.scraper.scrape_in_batches(skill="AI", pages_per_batch=1, max_batches=1, delay=0)
        assert count > 0
        mock_fetch.assert_called()
        mock_save.assert_called()

    @patch("src.scrape_jobs.tqdm", autospec=True)
    def test_scrape_in_batches_invalid_pages(self, mock_tqdm):
        """Should print error and return 0 when pages_per_batch <= 0."""
        result = self.scraper.scrape_in_batches(pages_per_batch=0)
        assert result == 0

# Run the tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])