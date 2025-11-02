"""Automated ablation studies and hyper-parameter search orchestration."""

from __future__ import annotations

import shutil
from dataclasses import asdict, replace
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Mapping

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
            model_name: str,
            model_builder: ModelBuilder,
            base_model_config: Mapping[str, Any],
            base_trainer_config: TrainerConfig,
    ) -> None:
        self._config = project_config
        self._dataset = dataset
        self._model_name = model_name
        self._model_builder = model_builder
        self._base_model_config = asdict(base_model_config) if hasattr(base_model_config,
                                                                       "__dataclass_fields__") else dict(
            base_model_config)
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

        for parameter, values in ablation_cfg.parameters.items():
            for value in values:
                for repeat in range(ablation_cfg.repeats):
                    overrides = {parameter: value}
                    model_overrides, trainer_overrides = self._split_overrides(overrides)
                    run_name = f"{parameter}_{value}_rep{repeat + 1}"
                    self._logger.info(
                        "Ablation run %s (repeat %d) with overrides %s.",
                        run_name,
                        repeat + 1,
                        overrides,
                    )
                    trainer_config = self._prepare_trainer_config(run_name, trainer_overrides, study_root, repeat)
                    model, resolved_config = self._model_builder(self._combine_model_config(model_overrides))
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

        summary, raw_frame = self._build_results_frames(results, metric_name, study_root)
        visualizer = AblationVisualizer(
            AblationVisualizerConfig(output_directory=study_root)
        )
        visualizer.save_summary_table(summary)
        visualizer.save_metric_overview(summary, metric_name)
        self._copy_best_checkpoint(raw_frame, metric_name, study_root, prefix="ablation")
        return study_root

    def run_hyperparameter_search(self) -> Path:
        """Run the configured hyper-parameter grid search."""

        search_cfg = self._config.hyperparameter_search
        metric_name = search_cfg.evaluation_metric
        study_root = self._create_study_root(search_cfg.output_subdir)
        records: list[Dict[str, Any]] = []

        for combination in search_cfg.iter_grid():
            model_overrides, trainer_overrides = self._split_overrides(combination)
            run_name = self._format_combination_name(combination)
            self._logger.info("Hyper-parameter run %s", run_name)
            trainer_config = self._prepare_trainer_config(run_name, trainer_overrides, study_root, repeat_index=0)
            model, resolved_config = self._model_builder(self._combine_model_config(model_overrides))
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
        visualizer.save_ranked_results(frame, "metric")
        if {"trainer__learning_rate", "trainer__weight_decay"}.issubset(frame.columns):
            visualizer.save_metric_heatmap(frame, "metric", "trainer__learning_rate", "trainer__weight_decay")
        self._copy_best_checkpoint(frame, "metric", study_root, prefix="hyperparameter")
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
            if key in self._base_model_config:
                model_overrides[key] = value
            else:
                trainer_overrides[key] = value
        return model_overrides, trainer_overrides

    def _combine_model_config(self, overrides: Mapping[str, Any]) -> Mapping[str, Any]:
        config = {**self._base_model_config}
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

        if "optimizer" in overrides:
            optimizer = replace(optimizer, name=str(overrides["optimizer"]))
        if "weight_decay" in overrides:
            optimizer = replace(optimizer, weight_decay=float(overrides["weight_decay"]))
        if "learning_rate" in overrides:
            scheduler = replace(scheduler, learning_rate=float(overrides["learning_rate"]))
        if "scheduler" in overrides:
            scheduler = replace(scheduler, name=str(overrides["scheduler"]))
        if "batch_size" in overrides:
            config = replace(config, batch_size=int(overrides["batch_size"]))
        if "num_epochs" in overrides:
            config = replace(config, num_epochs=int(overrides["num_epochs"]))

        seed = self._base_trainer_config.seed + repeat_index
        config = replace(config, optimizer=optimizer, scheduler=scheduler, seed=seed, output_dir=output_dir)
        return config

    def _train_model(self, model: Any, trainer_config: TrainerConfig, resolved_config: Any):
        trainer = Trainer(model, trainer_config)
        result = trainer.train(self._dataset)
        self._persist_model_definition(trainer_config.output_dir, resolved_config, trainer_config)
        return result

    def _persist_model_definition(self, output_dir: Path, model_config: Any, trainer_config: TrainerConfig) -> None:
        definition_path = output_dir / self.MODEL_DEFINITION_FILENAME
        model_config_dict = asdict(model_config) if hasattr(model_config, "__dataclass_fields__") else dict(
            model_config)
        payload = {
            "model_name": self._model_name,
            "config": model_config_dict,
            "trainer": trainer_config.to_dict(),
            "loss": asdict(trainer_config.loss),
        }
        with definition_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(payload, handle, sort_keys=False)

    def _extract_metric(self, result, metric_name: str) -> float:
        metrics = result.best_validation_metrics or {}
        value = metrics.get(metric_name)
        if value is None:
            self._logger.warning("Metric '%s' not found in best validation metrics. Falling back to NaN.", metric_name)
            return float("nan")
        return float(value)

    def _build_results_frames(
            self,
            results: Iterable[Dict[str, Any]],
            metric_name: str,
            study_root: Path,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        if not results:
            raise ValueError("No ablation experiments executed. Check the configuration.")
        frame = pd.DataFrame(results)
        raw_path = study_root / "ablation_raw_results.csv"
        frame.to_csv(raw_path, index=False)
        aggregated = frame.groupby(["parameter", "value"], as_index=False)[metric_name].mean()
        return aggregated, frame

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

    def _format_combination_name(self, combination: Mapping[str, Any]) -> str:
        fragments = [f"{key}-{str(value).replace('/', '_')}" for key, value in combination.items()]
        return "_".join(fragments)
