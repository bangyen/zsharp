# Copyright (c) 2025 Bangyen Pham
"""Tests for the multi-seed sweep script."""

from unittest.mock import MagicMock, patch

from scripts.sweep import build_config, format_table, main, run_sweep

CONFIGS = ["configs/sgd_baseline.yaml", "configs/zsharp_baseline.yaml"]


def _fake_train(config):
    """Return results whose accuracy encodes the seed."""
    results = MagicMock()
    results.model_dump.return_value = {
        "final_test_accuracy": 80.0 + config.seed
    }
    return results


class TestBuildConfig:
    """The sweep overrides only what it is asked to."""

    def test_applies_seed_epochs_and_checkpoint_dir(self, tmp_path):
        """Seed, epochs and checkpoint_dir replace the YAML values."""
        config = build_config(
            CONFIGS[1], 7, 3, regularize=False, checkpoint_dir=tmp_path
        )
        assert config.seed == 7
        assert config.train.epochs == 3
        assert config.train.checkpoint_dir == str(tmp_path)
        assert config.train.label_smoothing == 0.0
        assert config.train.strong_augmentation is False

    def test_regularize_enables_smoothing_and_augmentation(self, tmp_path):
        """--regularize turns on both regularization options."""
        config = build_config(
            CONFIGS[0], 1, 3, regularize=True, checkpoint_dir=tmp_path
        )
        assert config.train.label_smoothing == 0.1
        assert config.train.strong_augmentation is True


class TestRunSweep:
    """Every (config, seed) pair runs once and results are reused."""

    def test_runs_each_pair_and_skips_finished_runs(self, tmp_path):
        """A second invocation reads saved results instead of training."""
        with patch("scripts.sweep.train", side_effect=_fake_train) as mock:
            finals = run_sweep(
                CONFIGS, [1, 2], 5, regularize=True, out_dir=tmp_path
            )
            assert mock.call_count == 4
            again = run_sweep(
                CONFIGS, [1, 2], 5, regularize=True, out_dir=tmp_path
            )
            assert mock.call_count == 4

        assert (
            finals
            == again
            == {
                "sgd_baseline_e5_reg": [81.0, 82.0],
                "zsharp_baseline_e5_reg": [81.0, 82.0],
            }
        )
        assert (tmp_path / "zsharp_baseline_e5_reg_s2.json").exists()

    def test_variants_get_separate_checkpoint_dirs(self, tmp_path):
        """Checkpoints of different variants must not collide."""
        with patch("scripts.sweep.train", side_effect=_fake_train) as mock:
            run_sweep(CONFIGS, [1], 5, regularize=False, out_dir=tmp_path)

        dirs = {c.args[0].train.checkpoint_dir for c in mock.call_args_list}
        assert len(dirs) == 2

    def test_interrupted_run_stops_without_saving(self, tmp_path):
        """An interrupted run writes no result, so a rerun resumes it."""
        with patch("scripts.sweep.train", return_value=None):
            finals = run_sweep(
                CONFIGS, [1], 5, regularize=False, out_dir=tmp_path
            )
        assert finals == {}
        assert not list(tmp_path.glob("*.json"))


class TestOutput:
    """The summary table reports mean ± std per variant."""

    def test_format_table(self):
        """Std uses the sample formula; one seed reports zero spread."""
        table = format_table({"a": [80.0, 82.0], "b": [75.0]})
        assert "| a | 2 | 81.00 ± 1.41% |" in table
        assert "| b | 1 | 75.00 ± 0.00% |" in table

    def test_main_prints_table(self, tmp_path, capsys):
        """The CLI runs the sweep and prints its summary."""
        with patch("scripts.sweep.train", side_effect=_fake_train):
            code = main(
                ["--seeds", "3", "--epochs", "2", "--out-dir", str(tmp_path)]
            )
        assert code == 0
        assert "| zsharp_baseline_e2 | 1 | 83.00 ± 0.00% |" in (
            capsys.readouterr().out
        )
