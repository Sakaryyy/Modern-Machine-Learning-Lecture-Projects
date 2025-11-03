"""Automated ablation studies and hyper-parameter search orchestration."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from Project_2_Image_Classification.src.config.config import ProjectConfig
from Project_2_Image_Classification.src.data_loading.data_load_and_save import PreparedDataset
from Project_2_Image_Classification.src.training_routines import Trainer, TrainerConfig
from Project_2_Image_Classification.src.utils.logging import get_logger
from Project_2_Image_Classification.src.visualization.ablation_vis import AblationVisualizer, AblationVisualizerConfig
from Project_2_Image_Classification.src.visualization.hyperparameter_search_vis import (
    HyperparameterSearchVisualizer,
    HyperparameterSearchVisualizerConfig,
)

ModelBuilder = Callable[[Mapping[str, Any]], tuple[Any, Any]]


class HyperparameterExperimentManager:
    """Coordinate ablation studies and hyper-parameter searches."""

    MODEL_DEFINITION_FILENAME = "model_definition.yaml"

    def __init__(
            self,
            project_config: ProjectConfig,
            dataset: PreparedDataset,
            experiment_model_name: str,
            experiment_model_builder: Callable[[Mapping[str, Any]], tuple[Any, Any]],
            experiment_base_model_config: Mapping[str, Any],
            base_trainer_config: TrainerConfig,
            *,
            baseline_model_name: str,
            baseline_model_builder: Callable[[Mapping[str, Any]], tuple[Any, Any]],
    ) -> None:
        self._config = project_config
        self._dataset = dataset

        # model used for ablations / grid search
        self._experiment_model_name = experiment_model_name
        self._experiment_model_builder = experiment_model_builder
        if hasattr(experiment_base_model_config, "__dataclass_fields__"):
            self._experiment_base_model_config = asdict(experiment_base_model_config)
        else:
            self._experiment_base_model_config = dict(experiment_base_model_config)

        # model used for the "baseline" line
        self._baseline_model_name = baseline_model_name
        self._baseline_model_builder = baseline_model_builder

        self._base_trainer_config = base_trainer_config
        self._logger = get_logger(self.__class__.__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def run_ablation(self) -> Path:
        """Execute the ablation study as defined in :class:`ProjectConfig`."""

        ablation_cfg = self._config.ablation
        metric_name = ablation_cfg.metric
        study_root = self._create_study_root(ablation_cfg.output_subdir)
        results: list[Dict[str, Any]] = []

        self._logger.info(
            "Running baseline configuration '%s' for %d repeat(s) before ablations.",
            ablation_cfg.baseline_model,
            ablation_cfg.repeats,
        )
        baseline_records = self._run_baseline(study_root, ablation_cfg.repeats, metric_name)
        results.extend(baseline_records)
        parameter_amount = len(ablation_cfg.parameters.items())

        for p, (parameter, values) in enumerate(ablation_cfg.parameters.items()):
            values_amount = len(values)
            for v, value in enumerate(values):
                for repeat in range(ablation_cfg.repeats):
                    overrides = {parameter: value}
                    model_overrides, trainer_overrides = self._split_overrides(overrides)
                    run_name = f"{parameter}_{value}_rep{repeat + 1}_parameter_run_{p}-{parameter_amount}_value_run_{v}-{values_amount}"
                    self._logger.info(
                        "Ablation run %s (repeat %d) with overrides %s.",
                        run_name,
                        repeat + 1,
                        overrides,
                    )
                    trainer_config = self._prepare_trainer_config(run_name, trainer_overrides, study_root, repeat)
                    model, resolved_config = self._experiment_model_builder(
                        self._combine_experiment_model_config(model_overrides)
                    )
                    result = self._train_model(model, trainer_config, resolved_config)
                    metric_value = self._extract_metric(result, metric_name)
                    self._logger.info(
                        "Completed ablation run %s with %s=%.4f", run_name, metric_name, metric_value
                    )
                    results.append(
                        {
                            "parameter": parameter,
                            "value": value,
                            "repeat": repeat + 1,
                            metric_name: metric_value,
                            "output_dir": str(trainer_config.output_dir),
                            "checkpoint_path": str(result.checkpoint_path),
                        }
                    )

        visualizer = AblationVisualizer(
            AblationVisualizerConfig(output_directory=study_root)
        )
        summary, raw_frame = self._build_results_frames(results, metric_name, visualizer.tables_dir)
        visualizer.save_summary_table(summary)
        visualizer.save_metric_overview(summary, metric_name)
        try:
            delta_path = visualizer.save_delta_overview(summary, metric_name)
        except ValueError as exc:
            self._logger.debug("Skipping ablation delta plot: %s", exc)
        else:
            self._logger.info("Saved ablation delta plot to %s", delta_path)
        try:
            distribution_paths = visualizer.save_parameter_boxplots(raw_frame, metric_name)
            for parameter, path in distribution_paths.items():
                self._logger.info("Saved ablation distribution for %s to %s", parameter, path)
        except ValueError as exc:
            self._logger.warning("Could not create ablation distribution plots: %s", exc)
        self._copy_best_checkpoint(raw_frame, metric_name, study_root, prefix="ablation")
        return study_root

    def run_hyperparameter_search(self) -> Path:
        """Run the configured hyper-parameter grid search."""

        search_cfg = self._config.hyperparameter_search
        metric_name = search_cfg.evaluation_metric
        study_root = self._create_study_root(search_cfg.output_subdir)
        records: list[Dict[str, Any]] = []

        for index, combination in enumerate(search_cfg.iter_grid(), start=1):
            model_overrides, trainer_overrides = self._split_overrides(combination)
            run_name = self._format_combination_name(combination, index)
            self._logger.info("Hyper-parameter run %s", run_name)
            trainer_config = self._prepare_trainer_config(run_name, trainer_overrides, study_root, repeat_index=0)
            model, resolved_config = self._experiment_model_builder(
                self._combine_experiment_model_config(model_overrides)
            )
            result = self._train_model(model, trainer_config, resolved_config)
            metric_value = self._extract_metric(result, metric_name)
            self._logger.info(
                "Completed hyper-parameter run %s with %s=%.4f", run_name, metric_name, metric_value
            )
            record = {
                **{f"model__{k}": combination[k] for k in model_overrides},
                **{f"trainer__{k}": combination[k] for k in trainer_overrides},
                "metric": metric_value,
                "output_dir": str(trainer_config.output_dir),
                "checkpoint_path": str(result.checkpoint_path),
            }
            records.append(record)

        if not records:
            raise ValueError("Hyper-parameter search produced no results. Check the configuration.")

        frame = pd.DataFrame(records)
        frame_path = study_root / "hyperparameter_search_raw.csv"
        frame.to_csv(frame_path, index=False)

        visualizer = HyperparameterSearchVisualizer(
            HyperparameterSearchVisualizerConfig(output_directory=study_root)
        )
        raw_path = visualizer.tables_dir / "hyperparameter_search_raw.csv"
        frame.to_csv(raw_path, index=False)

        enriched = self._augment_hyperparameter_frame(frame)

        visualizer.save_ranked_results(enriched, "metric")
        if {"trainer__learning_rate", "trainer__weight_decay"}.issubset(enriched.columns):
            visualizer.save_metric_heatmap(enriched, "metric", "trainer__learning_rate", "trainer__weight_decay")
        visualizer.save_metric_distribution(enriched, "metric")
        try:
            visualizer.save_numeric_pairplot(enriched, "metric")
        except ValueError as exc:
            self._logger.warning("Skipping hyper-parameter pairplot: %s", exc)
        visualizer.save_top_configurations(enriched, "metric")
        try:
            effect_paths = visualizer.save_parameter_effects(enriched, "metric")
        except ValueError as exc:
            self._logger.debug("Skipping hyper-parameter effect plots: %s", exc)
        else:
            for column, path in effect_paths.items():
                self._logger.info("Saved hyper-parameter effect plot for %s to %s", column, path)
        self._copy_best_checkpoint(enriched, "metric", study_root, prefix="hyperparameter")
        return study_root

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_study_root(self, subdir: str) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        root = self._config.paths.analysis_dir / subdir / timestamp
        root.mkdir(parents=True, exist_ok=True)
        return root

    def _split_overrides(self, overrides: Mapping[str, Any]) -> tuple[Dict[str, Any], Dict[str, Any]]:
        model_overrides: Dict[str, Any] = {}
        trainer_overrides: Dict[str, Any] = {}
        for key, value in overrides.items():
            if key in self._experiment_base_model_config:
                model_overrides[key] = value
            else:
                trainer_overrides[key] = value
        return model_overrides, trainer_overrides

    def _combine_model_config(self, overrides: Mapping[str, Any]) -> Mapping[str, Any]:
        config = {**self._experiment_base_model_config}
        config.update(overrides)
        return config

    def _combine_experiment_model_config(self, overrides: Mapping[str, Any]) -> Mapping[str, Any]:
        config = {**self._experiment_base_model_config}
        config.update(overrides)
        return config

    def _prepare_trainer_config(
            self,
            run_name: str,
            overrides: Mapping[str, Any],
            study_root: Path,
            repeat_index: int,
    ) -> TrainerConfig:
        output_dir = study_root / run_name
        optimizer = self._base_trainer_config.optimizer
        scheduler = self._base_trainer_config.scheduler
        config = replace(self._base_trainer_config, output_dir=output_dir)

        optimizer_updates: Dict[str, Any] = {}
        scheduler_updates: Dict[str, Any] = {}

        for key, value in overrides.items():
            if key == "optimizer":
                optimizer_updates["name"] = str(value)
            elif key in {"weight_decay", "momentum", "beta1", "beta2", "eps"}:
                optimizer_updates[key] = float(value)
            elif key in {"nesterov", "centered"}:
                optimizer_updates[key] = bool(value)
            elif key == "scheduler":
                scheduler_updates["name"] = str(value)
            elif key in {"learning_rate", "warmup_init_value", "decay_rate", "alpha", "end_learning_rate"}:
                scheduler_updates[key] = float(value)
            elif key in {"warmup_steps", "transition_steps"}:
                scheduler_updates[key] = int(value)
            elif key == "batch_size":
                config = replace(config, batch_size=int(value))
            elif key == "eval_batch_size":
                config = replace(config, eval_batch_size=int(value))
            elif key == "num_epochs":
                config = replace(config, num_epochs=int(value))
            elif key == "log_every":
                config = replace(config, log_every=int(value))
            elif key == "evaluate_on_test":
                config = replace(config, evaluate_on_test=bool(value))

        seed = self._base_trainer_config.seed + repeat_index
        if optimizer_updates:
            optimizer = replace(optimizer, **optimizer_updates)
        if scheduler_updates:
            scheduler = replace(scheduler, **scheduler_updates)
        config = replace(config, optimizer=optimizer, scheduler=scheduler, seed=seed, output_dir=output_dir)
        return config

    def _train_model(
            self,
            model: Any,
            trainer_config: TrainerConfig,
            resolved_config: Any,
            *,
            model_name: str | None = None,
    ):
        trainer = Trainer(model, trainer_config)
        result = trainer.train(self._dataset)
        self._persist_model_definition(
            trainer_config.output_dir,
            resolved_config,
            trainer_config,
            model_name=model_name,
        )
        return result

    def _run_baseline(self, study_root: Path, repeats: int, metric_name: str) -> list[Dict[str, Any]]:
        """Train the baseline configuration used for comparisons."""

        baseline_records: list[Dict[str, Any]] = []
        for repeat in range(repeats):
            run_name = f"{self._baseline_model_name}_rep{repeat + 1}"
            trainer_config = self._prepare_trainer_config(run_name, {}, study_root, repeat)
            model, resolved_config = self._baseline_model_builder({})

            result = self._train_model(
                model,
                trainer_config,
                resolved_config,
                model_name=self._baseline_model_name,
            )
            metric_value = self._extract_metric(result, metric_name)
            self._logger.info(
                "Baseline run %s completed with %s=%.4f",
                run_name,
                metric_name,
                metric_value,
            )
            baseline_records.append(
                {
                    "parameter": "baseline",
                    "value": "default",
                    "repeat": repeat + 1,
                    metric_name: metric_value,
                    "output_dir": str(trainer_config.output_dir),
                    "checkpoint_path": str(result.checkpoint_path),
                }
            )
        return baseline_records

    def _persist_model_definition(
            self,
            output_dir: Path,
            model_config: Any,
            trainer_config: TrainerConfig,
            *,
            model_name: str | None = None,
    ) -> None:
        definition_path = output_dir / self.MODEL_DEFINITION_FILENAME
        model_config_dict = (
            asdict(model_config)
            if hasattr(model_config, "__dataclass_fields__")
            else dict(model_config)
        )
        payload = {
            "model_name": model_name or self._experiment_model_name,
            "config": model_config_dict,
            "trainer": trainer_config.to_dict(),
            "loss": asdict(trainer_config.loss),
        }
        with definition_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    def _extract_metric(self, result, metric_name: str) -> float:
        metrics = result.best_validation_metrics or {}
        value = metrics.get(metric_name)

        if value is None and metric_name.startswith("validation_"):
            stripped = metric_name[len("validation_"):]
            value = metrics.get(stripped)

        if value is None and metric_name.startswith("test_"):
            test_metrics = result.test_metrics or {}
            stripped = metric_name[len("test_"):]
            value = test_metrics.get(stripped)

        if value is None and metric_name.startswith("train_"):
            final_train = result.history.iloc[-1] if not result.history.empty else None
            if final_train is not None and metric_name in final_train:
                value = float(final_train[metric_name])

        if value is None:
            self._logger.warning(
                "Metric '%s' not found in recorded results. Falling back to NaN.",
                metric_name,
            )
            return float("nan")

        return float(value)

    def _build_results_frames(
            self,
            results: Iterable[Dict[str, Any]],
            metric_name: str,
            tables_dir: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not results:
            raise ValueError("No ablation experiments executed. Check the configuration.")
        frame = pd.DataFrame(results)
        tables_dir.mkdir(parents=True, exist_ok=True)
        raw_path = tables_dir / "ablation_raw_results.csv"
        frame.to_csv(raw_path, index=False)

        baseline_rows = frame[frame["parameter"] == "baseline"]
        baseline_mean = float(baseline_rows[metric_name].mean()) if not baseline_rows.empty else None

        aggregated = (
            frame.groupby(["parameter", "value"])[metric_name]
            .agg(["mean", "std", "count"])
            .reset_index()
            .rename(columns={"mean": metric_name, "std": f"{metric_name}_std", "count": "runs"})
        )

        if baseline_mean is not None and np.isfinite(baseline_mean):
            aggregated["delta_vs_baseline"] = aggregated[metric_name] - baseline_mean
            denominator = baseline_mean if baseline_mean != 0 else np.nan
            aggregated["relative_change_pct"] = aggregated["delta_vs_baseline"] / denominator * 100.0
            frame["delta_vs_baseline"] = frame[metric_name] - baseline_mean
        else:
            aggregated["delta_vs_baseline"] = np.nan
            aggregated["relative_change_pct"] = np.nan

        return aggregated, frame

    def _augment_hyperparameter_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Add derived statistics to the hyper-parameter search results."""

        enriched = frame.copy()

        def _normalize_blocks(value: Any) -> Sequence[Mapping[str, Any]] | None:
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, dict)):
                return [dict(block) for block in value]
            if isinstance(value, str):
                try:
                    parsed = yaml.safe_load(value)
                except Exception:
                    return None
                if isinstance(parsed, Sequence) and not isinstance(parsed, (str, bytes, dict)):
                    return [dict(block) for block in parsed]
            return None

        if "model__conv_blocks" in enriched.columns:
            conv_blocks = enriched["model__conv_blocks"].apply(_normalize_blocks)
            enriched["model__conv_block_count"] = conv_blocks.apply(lambda blocks: len(blocks) if blocks else np.nan)
            enriched["model__total_conv_features"] = conv_blocks.apply(
                lambda blocks: float(np.sum([block.get("features", 0) for block in blocks])) if blocks else np.nan
            )

        if "model__dense_blocks" in enriched.columns:
            dense_blocks = enriched["model__dense_blocks"].apply(_normalize_blocks)
            enriched["model__dense_block_count"] = dense_blocks.apply(lambda blocks: len(blocks) if blocks else np.nan)
            enriched["model__total_dense_units"] = dense_blocks.apply(
                lambda blocks: float(np.sum([block.get("features", 0) for block in blocks])) if blocks else np.nan
            )

        if "metric" in enriched.columns:
            enriched["metric_rank"] = enriched["metric"].rank(ascending=False, method="min")

        return enriched

    def _copy_best_checkpoint(self, frame: pd.DataFrame, metric_name: str, study_root: Path, prefix: str) -> None:
        best_row = frame.sort_values(metric_name, ascending=False).iloc[0]
        checkpoint_path = Path(best_row["checkpoint_path"]) if "checkpoint_path" in best_row else None
        if checkpoint_path is None or not checkpoint_path.exists():
            self._logger.warning("Best checkpoint %s not found. Skipping copy to model registry.", checkpoint_path)
            return
        target = self._config.paths.models_dir / f"{prefix}_best_{study_root.name}.msgpack"
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(checkpoint_path, target)
        self._logger.info("Copied best %s checkpoint to %s", prefix, target)

    def _format_combination_name(self, combination: Mapping[str, Any], index: int) -> str:
        fragments = []
        for key, value in combination.items():
            fragments.append(f"{key}-{self._slugify(value)}")
        suffix = "__".join(fragments)
        return f"combo{index:03d}_{suffix}" if suffix else f"combo{index:03d}"

    @staticmethod
    def _slugify(value: Any) -> str:
        if isinstance(value, (int, float)):
            return str(value).replace(".", "p")
        if isinstance(value, bool):
            return str(value).lower()
        if isinstance(value, str):
            cleaned = value.replace("/", "-")
            return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "-" for ch in cleaned)
        serialized = yaml.safe_dump(value, sort_keys=True)
        digest = hashlib.sha1(serialized.encode("utf-8")).hexdigest()[:10]
        return f"hash-{digest}"
