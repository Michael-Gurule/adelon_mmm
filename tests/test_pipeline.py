"""Tests for src/pipeline/ CLI entrypoints."""

import argparse
import json
from pathlib import Path

import pandas as pd
import pytest


class TestGeneratePipeline:
    def test_generate_creates_data_files(self, tmp_path):
        """adelon-generate should produce CSV, contributions CSV, and JSON."""
        from src.pipeline.generate import main

        args = argparse.Namespace(
            n_days=30,
            start_date="2023-01-01",
            seed=0,
            output_dir=str(tmp_path),
            log_level="WARNING",
            log_file=None,
        )
        main(args)

        assert (tmp_path / "mmm_daily_data.csv").exists()
        assert (tmp_path / "mmm_contributions_truth.csv").exists()
        assert (tmp_path / "mmm_ground_truth.json").exists()

    def test_generate_correct_row_count(self, tmp_path):
        """Generated CSV should have exactly n_days rows."""
        from src.pipeline.generate import main

        n_days = 45
        args = argparse.Namespace(
            n_days=n_days,
            start_date="2023-01-01",
            seed=0,
            output_dir=str(tmp_path),
            log_level="WARNING",
            log_file=None,
        )
        main(args)

        df = pd.read_csv(tmp_path / "mmm_daily_data.csv")
        assert len(df) == n_days

    def test_generate_ground_truth_structure(self, tmp_path):
        """Ground truth JSON should have expected top-level keys."""
        from src.pipeline.generate import main

        args = argparse.Namespace(
            n_days=30,
            start_date="2023-01-01",
            seed=0,
            output_dir=str(tmp_path),
            log_level="WARNING",
            log_file=None,
        )
        main(args)

        with open(tmp_path / "mmm_ground_truth.json") as f:
            gt = json.load(f)

        for key in ["baseline", "seasonality", "channels", "controls"]:
            assert key in gt

    def test_generate_reproducible_with_seed(self, tmp_path):
        """Same seed should produce identical data."""
        from src.pipeline.generate import main

        dir_a = tmp_path / "a"
        dir_b = tmp_path / "b"
        dir_a.mkdir()
        dir_b.mkdir()

        for d in [dir_a, dir_b]:
            args = argparse.Namespace(
                n_days=30,
                start_date="2023-01-01",
                seed=99,
                output_dir=str(d),
                log_level="WARNING",
                log_file=None,
            )
            main(args)

        df_a = pd.read_csv(dir_a / "mmm_daily_data.csv")
        df_b = pd.read_csv(dir_b / "mmm_daily_data.csv")
        pd.testing.assert_frame_equal(df_a, df_b)


class TestEvaluatePipeline:
    def test_evaluate_requires_trace(self, tmp_path):
        """adelon-evaluate should raise if trace file is missing."""
        from src.pipeline.evaluate import main

        args = argparse.Namespace(
            config="config/model_config.yaml",
            trace=str(tmp_path / "nonexistent.netcdf"),
            output_dir=str(tmp_path / "out"),
            ground_truth=None,
            log_level="WARNING",
            log_file=None,
        )
        with pytest.raises(FileNotFoundError, match="Trace file"):
            main(args)

    def test_evaluate_requires_config(self, tmp_path):
        """adelon-evaluate should raise if config file is missing."""
        from src.pipeline.evaluate import main

        args = argparse.Namespace(
            config=str(tmp_path / "nonexistent.yaml"),
            trace=None,
            output_dir=str(tmp_path / "out"),
            ground_truth=None,
            log_level="WARNING",
            log_file=None,
        )
        with pytest.raises(FileNotFoundError, match="Config file"):
            main(args)
