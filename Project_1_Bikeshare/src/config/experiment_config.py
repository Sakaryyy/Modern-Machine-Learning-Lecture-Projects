from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Mapping

import yaml


@dataclass(frozen=True)
class Paths:
    """Container for project paths.

    Attributes
    ----------
    root : Path
        Project root directory.
    raw_dir : Path
        Directory for cached raw data files.
    processed_dir : Path
        Directory for cleaned/processed datasets.
    eda_figures_dir : Path
        Directory for PNG figures.
    eda_tables_dir : Path
        Directory for Excel exports.
    """
    root: Path
    raw_dir: Path
    processed_dir: Path
    eda_figures_dir: Path
    model_figures_dir: Path
    regression_figures_dir: Path
    classification_figures_dir: Path
    eda_tables_dir: Path
    model_tables_dir: Path
    regression_tables_dir: Path
    classification_tables_dir: Path

    def ensure_exists(self) -> None:
        """
        Create all directories if they do not already exist.

        This function is safe to call multiple times.
        """
        for p in (
            self.raw_dir,
            self.processed_dir,
            self.eda_figures_dir,
            self.eda_tables_dir,
            self.model_figures_dir,
            self.regression_figures_dir,
            self.classification_figures_dir,
            self.model_tables_dir,
            self.regression_tables_dir,
            self.classification_tables_dir,
        ):
            p.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class Splits:
    """Configuration for chronological splits.

    Attributes
    ----------
    train_end : str
        ISO date marking the end of training window (inclusive).
    validation_end : str
        ISO date marking the end of validation window (inclusive).
    """
    train_end: str = "2012-06-30"
    validation_end: str = "2012-10-31"

@dataclass(frozen=True)
class FeatureConfig:
    """
    Feature toggles for composing candidate groups and pipelines.

    Attributes
    ----------
    use_atemp : bool
        Include apparent temperature features.
    use_temp : bool
        Include raw temperature features. This project primarily uses `atemp`
    use_workingday : bool
        Include working day indicator as a numeric feature.
    use_humidity : bool
        Include humidity as a numeric feature.
    use_windspeed : bool
        Include wind speed as a numeric feature.
    add_hour_cyclical : bool
        Include hour-of-day cyclical encoding (sin and cos).
    use_season_onehot : bool
        Include season categorical feature mapping. NotImplemented yet.
    use_holiday : bool
        Include holiday indicator. NotImplemented yet.
    use_weathersit_onehot : bool
        Include weather situation category. NotImplemented yet.
    poly_temp_degree : int
        Maximum polynomial degree for temperature features. Values >= 2 introduce
        nonlinear terms like atemp^2, atemp^3, and so on, depending on the builder.
    """
    # Hour encodings
    add_hour_cyclical: bool = True
    hour_fourier_harmonics: int = 2  # if >1, uses hour_fourier_step with this many harmonics
    add_hour_onehot: bool = True
    onehot_drop_first: bool = True

    # Interactions
    add_hour_interactions_with_atemp: bool = False

    # Weather / calendar scalars
    use_atemp: bool = True
    use_temp: bool = False
    use_workingday: bool = False
    use_weekday_onehot: bool = False
    use_month_onehot: bool = False
    use_humidity: bool = False
    use_windspeed: bool = False
    use_season_onehot: bool = True
    season_drop_first: bool = True
    use_holiday: bool = False
    use_weathersit_onehot: bool = True
    weathersit_drop_first: bool = True
    poly_temp_degree: int = 2

@dataclass(frozen=True)
class ExperimentConfig:
    """
    Top-level configuration for an experiment run.

    Attributes
    ----------
    paths : Paths
        Filesystem locations used by the pipeline.
    splits : Splits
        Chronological splitting boundaries.
    features : FeatureConfig
        Feature toggle configuration.

    """
    paths: Paths
    splits: Splits = field(default_factory=Splits)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @staticmethod
    def _inferred_root() -> Path:
        """Return the repository root inferred from this module's location."""
        return Path(__file__).resolve().parents[2]

    @staticmethod
    def _default_paths(root: Path) -> Paths:
        """Create the default directory layout relative to ``root``."""

        return Paths(
            root=root,
            raw_dir=root / "data" / "raw",
            processed_dir=root / "data" / "processed",
            eda_figures_dir=root / "outputs" / "figures" / "eda",
            model_figures_dir=root / "outputs" / "figures" / "model",
            regression_figures_dir=root / "outputs" / "figures" / "model" / "regression",
            classification_figures_dir=root
                                       / "outputs"
                                       / "figures"
                                       / "model"
                                       / "classification",
            eda_tables_dir=root / "outputs" / "tables" / "eda",
            model_tables_dir=root / "outputs" / "tables" / "model",
            regression_tables_dir=root
                                  / "outputs"
                                  / "tables"
                                  / "model"
                                  / "regression",
            classification_tables_dir=root
                                      / "outputs"
                                      / "tables"
                                      / "model"
                                      / "classification",
        )

    @staticmethod
    def default() -> "ExperimentConfig":
        """
        Build a class `ExperimentConfig` anchored at the inferred project root.

        The loader first looks for ``config/experiment.yaml`` under the project
        root. When present, the YAML content is parsed to override the defaults.
        """

        env_path = os.environ.get("EXPERIMENT_CONFIG")
        if env_path:
            return ExperimentConfig.from_file(Path(env_path).expanduser())

        root = ExperimentConfig._inferred_root()
        config_path = root / "config" / "experiment.yaml"
        if config_path.exists():
            return ExperimentConfig.from_file(config_path)
        return ExperimentConfig(paths=ExperimentConfig._default_paths(root))

    @staticmethod
    def from_file(path: Path) -> "ExperimentConfig":
        """Load configuration from a YAML or JSON file."""

        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")

        suffix = path.suffix.lower()
        text = path.read_text()
        if suffix in {".yaml", ".yml"}:
            data = yaml.safe_load(text) or {}
        elif suffix == ".json":
            data = json.loads(text)
        else:
            raise ValueError(
                f"Unsupported configuration format '{suffix}'. Use .yaml, .yml, or .json."
            )

        if not isinstance(data, Mapping):
            raise ValueError("Configuration file must define a mapping at the top level.")

        base_root = path.resolve().parents[1] if len(path.parents) >= 2 else path.resolve().parent
        paths_cfg = data.get("paths", {})
        root = Path(paths_cfg.get("root", base_root))
        if not root.is_absolute():
            root = base_root / root

        def _resolve_path(key: str, default: Path) -> Path:
            raw_value = paths_cfg.get(key)
            if raw_value is None:
                return default
            candidate = Path(raw_value)
            return candidate if candidate.is_absolute() else root / candidate

        default_paths = ExperimentConfig._default_paths(root)
        paths = Paths(
            root=root,
            raw_dir=_resolve_path("raw_dir", default_paths.raw_dir),
            processed_dir=_resolve_path("processed_dir", default_paths.processed_dir),
            eda_figures_dir=_resolve_path("eda_figures_dir", default_paths.eda_figures_dir),
            model_figures_dir=_resolve_path("model_figures_dir", default_paths.model_figures_dir),
            regression_figures_dir=_resolve_path(
                "regression_figures_dir", default_paths.regression_figures_dir
            ),
            classification_figures_dir=_resolve_path(
                "classification_figures_dir", default_paths.classification_figures_dir
            ),
            eda_tables_dir=_resolve_path("eda_tables_dir", default_paths.eda_tables_dir),
            model_tables_dir=_resolve_path("model_tables_dir", default_paths.model_tables_dir),
            regression_tables_dir=_resolve_path(
                "regression_tables_dir", default_paths.regression_tables_dir
            ),
            classification_tables_dir=_resolve_path(
                "classification_tables_dir", default_paths.classification_tables_dir
            ),
        )

        splits_cfg = data.get("splits", {})
        splits = replace(
            Splits(),
            train_end=splits_cfg.get("train_end", Splits.train_end),
            validation_end=splits_cfg.get("validation_end", Splits.validation_end),
        )

        features_cfg = data.get("features", {})
        feature_defaults = FeatureConfig()
        feature_kwargs: dict[str, Any] = {}
        for field_name in feature_defaults.__dataclass_fields__:
            if field_name in features_cfg:
                feature_kwargs[field_name] = features_cfg[field_name]
        features = replace(feature_defaults, **feature_kwargs)

        return ExperimentConfig(paths=paths, splits=splits, features=features)
