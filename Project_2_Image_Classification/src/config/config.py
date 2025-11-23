"""Centralised configuration utilities for the image classification project.

The module exposes a small hierarchy of dataclasses that capture default paths,
hyper-parameters and experiment settings.  A :class:`ConfigManager` persists the
configuration to disk which allows the command line entry point to reload the
latest settings across application runs.  This keeps command line defaults in
sync with previous executions while still enabling ad-hoc overrides via the
respective flags.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import product
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, MutableMapping

import yaml

from Project_2_Image_Classification.src.utils.logging import get_logger

LOGGER = get_logger(__name__)


@dataclass(slots=True)
class PathsConfig:
    """Directory layout used by the project.

    Parameters
    ----------
    data_dir:
        Directory storing downloaded and processed dataset artefacts.
    outputs_dir:
        Root directory for all generated artefacts (models, figures, reports).
    configs_dir:
        Directory where configuration snapshots are persisted.
    models_dir:
        Folder dedicated to serialised model checkpoints.
    analysis_dir:
        Directory containing reports from hyper-parameter searches and
        ablation studies.
    figures_dir:
        Default location for generated plots that are not tied to a concrete
        training run.
    """

    data_dir: Path
    outputs_dir: Path
    configs_dir: Path
    models_dir: Path
    analysis_dir: Path
    figures_dir: Path

    def ensure_directories(self) -> None:
        """Create all directories if they do not exist already."""

        for path in (
                self.data_dir,
                self.outputs_dir,
                self.configs_dir,
                self.models_dir,
                self.analysis_dir,
                self.figures_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class TrainingDefaults:
    """Default hyper-parameters used for baseline training routines."""

    num_epochs: int = 50
    batch_size: int = 128
    eval_batch_size: int = 256
    log_every: int = 100
    optimizer: str = "adamw"
    nesterov: bool = True
    beta1: float = 0.9
    beta2: float = 0.999
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    scheduler: str = "cosine_decay"
    scheduler_kwargs: Mapping[str, float] = field(default_factory=lambda: {"warmup_steps": 500, "alpha": 0.0})
    loss: str = "cross_entropy"
    metrics: tuple[str, ...] = ("loss", "accuracy")
    use_data_augmentation: bool = True
    label_smoothing: float = 0.05


@dataclass(slots=True)
class BaselineModelDefaults:
    """Starting point for the baseline multi-layer perceptron."""

    hidden_units: int = 512
    activation: str = "relu"
    dropout_rate: float = 0.2
    use_bias: bool = True
    kernel_init: str = "he_normal"
    bias_init: str = "zeros"


@dataclass(slots=True)
class CNNModelDefaults:
    """Defaults for the convolutional architecture used in experiments."""

    conv_blocks: List[Mapping[str, Any]] = field(
        default_factory=lambda: [
            {"features": 32, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.1},
            {"features": 64, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.2},
            {"features": 128, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.3},
        ]
    )
    dense_blocks: List[Mapping[str, Any]] = field(
        default_factory=lambda: [
            {"features": 256, "dropout_rate": 0.5, "activation": "relu"},
            {"features": 128, "dropout_rate": 0.3, "activation": "relu"},
        ]
    )
    classifier_dropout: float = 0.5
    global_average_pooling: bool = True
    classifier_kernel_init: str = "he_normal"
    classifier_bias_init: str = "zeros"
    classifier_use_bias: bool = True


@dataclass(slots=True)
class AblationStudyConfig:
    """Configuration describing the automatic ablation procedure."""

    parameters: Mapping[str, List[Any]] = field(
        default_factory=lambda: {
            "optimizer": ["adamw", "adam", "sgd", "rmsprop"],
            "learning_rate": [1e-3, 5e-3, 1e-2],
            "weight_decay": [0.0, 1e-4, 5e-4],
            "scheduler": [
                "constant",
                "cosine_decay",
                "linear_warmup_cosine_decay",
                "exponential_decay",
            ]
        }
    )
    baseline_model: str = "baseline"
    repeats: int = 1
    metric: str = "validation_accuracy"
    output_subdir: str = "ablation"

    def iter_runs(self) -> Iterator[Mapping[str, Any]]:
        """Yield successive configurations for ablation studies.

        Each iteration isolates a single hyper-parameter change relative to the
        defaults.  For every configured parameter, the generator yields
        dictionaries that only modify one entry at a time.
        """

        for name, values in self.parameters.items():
            for value in values:
                yield {name: value}


@dataclass(slots=True)
class HyperparameterSearchConfig:
    """Configuration describing the hyper-parameter grid search."""

    search_space: Mapping[str, List[Any]] = field(
        default_factory=lambda: {
            "conv_blocks": [
                [
                    {"features": 32, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.1},
                    {"features": 64, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.2},
                    {"features": 128, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.3},
                ],
                [
                    {"features": 64, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.1},
                    {"features": 128, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.2},
                    {"features": 256, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.3},
                    {"features": 256, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.35},
                ],
                [
                    {"features": 32, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.1},
                    {"features": 64, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.2},
                    {"features": 128, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.25},
                    {"features": 256, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.3},
                    {"features": 512, "kernel_size": (3, 3), "pooling_type": "max", "dropout_rate": 0.4},
                ],
            ],
            "dense_blocks": [
                [
                    {"features": 512, "dropout_rate": 0.4, "activation": "relu"},
                    {"features": 256, "dropout_rate": 0.3, "activation": "relu"},
                ],
                [
                    {"features": 512, "dropout_rate": 0.4, "activation": "relu"},
                    {"features": 256, "dropout_rate": 0.3, "activation": "relu"},
                    {"features": 128, "dropout_rate": 0.2, "activation": "relu"},
                ],
            ],
            "learning_rate": [1e-3, 5e-3],
            "weight_decay": [1e-4, 5e-4],
            "optimizer": ["adamw", "rmsprop"],
            "scheduler": ["cosine_decay", "constant"],
        }
    )
    evaluation_metric: str = "validation_accuracy"
    max_combinations: int | None = 120
    output_subdir: str = "hyperparameter_search"

    def iter_grid(self) -> Iterator[Mapping[str, Any]]:
        """Yield combinations."""

        items: List[tuple[str, List[Any]]] = [(name, list(values)) for name, values in self.search_space.items()]
        keys = [name for name, _ in items]
        value_lists = [values for _, values in items]

        total = 1
        for values in value_lists:
            total *= max(len(values), 1)

        if self.max_combinations is not None:
            LOGGER.info(
                "Restricting hyper-parameter search to first %d combinations out of %d.",
                self.max_combinations,
                total,
            )

        for index, combination in enumerate(product(*value_lists), start=1):
            if self.max_combinations is not None and index > self.max_combinations:
                break
            yield dict(zip(keys, combination, strict=True))


@dataclass(slots=True)
class ProjectConfig:
    """Bundle collecting configuration fragments used across the project."""

    paths: PathsConfig
    training: TrainingDefaults = field(default_factory=TrainingDefaults)
    baseline_defaults: BaselineModelDefaults = field(default_factory=BaselineModelDefaults)
    cnn_defaults: CNNModelDefaults = field(default_factory=CNNModelDefaults)
    ablation: AblationStudyConfig = field(default_factory=AblationStudyConfig)
    hyperparameter_search: HyperparameterSearchConfig = field(default_factory=HyperparameterSearchConfig)

    @classmethod
    def default(cls) -> "ProjectConfig":
        """Return the project defaults derived from the repository layout."""

        root = Path(__file__).resolve().parents[2]
        outputs_dir = root / "outputs"
        config = cls(
            paths=PathsConfig(
                data_dir=root / "data",
                outputs_dir=outputs_dir,
                configs_dir=outputs_dir / "configs",
                models_dir=outputs_dir / "models",
                analysis_dir=outputs_dir / "analysis",
                figures_dir=outputs_dir / "figures",
            )
        )
        config.paths.ensure_directories()
        return config

    def to_dict(self) -> Dict[str, Any]:
        """Serialise the configuration to a mapping ready for YAML dumping."""

        def _convert(obj: Any) -> Any:
            if isinstance(obj, Path):
                return str(obj)
            if isinstance(obj, Mapping):
                return {key: _convert(value) for key, value in obj.items()}
            if isinstance(obj, (list, tuple)):
                return [_convert(value) for value in obj]
            if dataclass_is_instance(obj):
                return {key: _convert(value) for key, value in asdict(obj).items()}
            return obj

        data: Dict[str, Any] = {
            "paths": {
                "data_dir": str(self.paths.data_dir),
                "outputs_dir": str(self.paths.outputs_dir),
                "configs_dir": str(self.paths.configs_dir),
                "models_dir": str(self.paths.models_dir),
                "analysis_dir": str(self.paths.analysis_dir),
                "figures_dir": str(self.paths.figures_dir),
            },
            "training": _convert(self.training),
            "baseline_defaults": _convert(self.baseline_defaults),
            "cnn_defaults": _convert(self.cnn_defaults),
            "ablation": _convert(self.ablation),
            "hyperparameter_search": _convert(self.hyperparameter_search),
        }
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ProjectConfig":
        """Construct a :class:`ProjectConfig` from a nested mapping."""

        paths_dict = data.get("paths", {})
        paths = PathsConfig(
            data_dir=Path(paths_dict.get("data_dir", "data")),
            outputs_dir=Path(paths_dict.get("outputs_dir", "outputs")),
            configs_dir=Path(paths_dict.get("configs_dir", "outputs/configs")),
            models_dir=Path(paths_dict.get("models_dir", "outputs/models")),
            analysis_dir=Path(paths_dict.get("analysis_dir", "outputs/analysis")),
            figures_dir=Path(paths_dict.get("figures_dir", "outputs/figures")),
        )

        training = TrainingDefaults(**_pop_mapping(data, "training"))
        baseline = BaselineModelDefaults(**_pop_mapping(data, "baseline_defaults"))
        cnn = CNNModelDefaults(**_pop_mapping(data, "cnn_defaults"))
        ablation = AblationStudyConfig(**_pop_mapping(data, "ablation"))
        hyper_search = HyperparameterSearchConfig(**_pop_mapping(data, "hyperparameter_search"))

        config = cls(
            paths=paths,
            training=training,
            baseline_defaults=baseline,
            cnn_defaults=cnn,
            ablation=ablation,
            hyperparameter_search=hyper_search,
        )
        config.paths.ensure_directories()
        return config


class ConfigManager:
    """Load and persist :class:`ProjectConfig` instances."""

    DEFAULT_FILENAME = "project_config.yaml"

    def __init__(self, config_path: Path | None = None) -> None:
        base_dir = Path(__file__).resolve().parents[2] / "outputs" / "configs"
        if config_path is None:
            config_path = base_dir / self.DEFAULT_FILENAME
        else:
            config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        self._config_path = config_path
        self._logger = get_logger(self.__class__.__name__)

    @property
    def path(self) -> Path:
        """Return the location where the configuration is stored."""

        return self._config_path

    def load(self) -> ProjectConfig:
        """Load the configuration from disk, creating defaults if necessary."""

        if not self._config_path.exists():
            self._logger.info("Configuration file not found. Creating default configuration at %s.", self._config_path)
            config = ProjectConfig.default()
            self.save(config)
            return config

        with self._config_path.open("r", encoding="utf-8") as handle:
            raw_data = yaml.safe_load(handle) or {}
        if not isinstance(raw_data, Mapping):
            raise TypeError("The configuration file must contain a mapping at the top level.")

        config = ProjectConfig.from_dict(raw_data)
        return config

    def save(self, config: ProjectConfig) -> None:
        """Persist ``config`` to disk in YAML format."""

        data = config.to_dict()
        with self._config_path.open("w", encoding="utf-8") as handle:
            yaml.safe_dump(data, handle, sort_keys=False)
        self._logger.info("Saved project configuration to %s", self._config_path)


def dataclass_is_instance(obj: Any) -> bool:
    """Return ``True`` if ``obj`` is an instance of a dataclass."""

    return hasattr(obj, "__dataclass_fields__")


def _pop_mapping(data: Mapping[str, Any], key: str) -> MutableMapping[str, Any]:
    """Return a mutable mapping for ``key`` from ``data`` or an empty dict."""

    value = data.get(key, {})
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"Configuration section '{key}' must be a mapping.")
    return dict(value)
