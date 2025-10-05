from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

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
    eda_tables_dir: Path
    model_tables_dir: Path

    def ensure_exists(self) -> None:
        """
        Create all directories if they do not already exist.

        This function is safe to call multiple times.
        """
        for p in (self.raw_dir, self.processed_dir, self.eda_figures_dir, self.eda_tables_dir):
            p.mkdir(parents=True, exist_ok=True)

@dataclass(frozen=True)
class Splits:
    """Configuration for chronological splits.

    Attributes
    ----------
    train_end : str
        ISO date marking the end of training window (inclusive).
    validation_end : str
        ISO date marking the end of holdout window (inclusive).
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
    use_season : bool
        Include season categorical feature mapping. NotImplemented yet.
    use_holiday : bool
        Include holiday indicator. NotImplemented yet.
    use_weathersit : bool
        Include weather situation category. NotImplemented yet.
    poly_temp_degree : int
        Maximum polynomial degree for temperature features. Values >= 2 introduce
        nonlinear terms like atemp^2, atemp^3, and so on, depending on the builder.
    """
    # Hour encodings
    add_hour_cyclical: bool = True
    hour_fourier_harmonics: int = 3  # if >1, uses hour_fourier_step with this many harmonics
    add_hour_onehot: bool = True
    onehot_drop_first: bool = True

    # Interactions
    add_hour_interactions_with_atemp: bool = True

    # Weather / calendar scalars
    use_atemp: bool = True
    use_temp: bool = False
    use_workingday: bool = True
    use_weekday_onehot: bool = True
    use_month_onehot: bool = True
    use_humidity: bool = False
    use_windspeed: bool = False
    use_season_onehot: bool = True
    season_drop_first: bool = True
    use_holiday: bool = False
    use_weathersit_onehot: bool = True
    weathersit_drop_first: bool = True
    poly_temp_degree: int = 4

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

    Methods
    -------
    default() -> ExperimentConfig
        Construct a configuration using a project-root inferred from this file's
        location. The resulting object does not create directories automatically.
        Call `cfg.paths.ensure_exists()` before writing outputs if needed.
    """
    paths: Paths
    splits: Splits = field(default_factory=Splits)
    features: FeatureConfig = field(default_factory=FeatureConfig)

    @staticmethod
    def default() -> "ExperimentConfig":
        """
        Build an ExperimentConfig anchored at the inferred project root.

        The root is computed as two levels above this file:
        src/config/experiment_config.py  ->  project root at parents[2].

        Returns
        -------
        ExperimentConfig
            A configuration with reasonable default paths under the project root.
        """
        root = Path(__file__).resolve().parents[2]  # .../Project_1_Bikeshare
        paths = Paths(
            root=root,
            raw_dir=root / "data" / "raw",
            processed_dir=root / "data" / "processed",
            eda_figures_dir=root / "outputs" / "figures" / "eda",
            model_figures_dir=root / "outputs" / "figures" / "model",
            eda_tables_dir=root / "outputs" / "tables" / "eda",
            model_tables_dir=root / "outputs" / "tables" / "model"
        )
        return ExperimentConfig(paths=paths)
