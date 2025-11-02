"""End-to-end training orchestration for the image classification models."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Mapping, MutableMapping, Sequence

import jax
import jax.numpy as jnp
import optax
import pandas as pd
import yaml
from flax import linen as nn
from flax import serialization
from flax.core import FrozenDict
from flax.training import train_state

from Project_2_Image_Classification.src.data_loading.data_load_and_save import DatasetSplit, PreparedDataset
from Project_2_Image_Classification.src.training_routines.learning_rate_schedulers import LRSchedulerConfig, \
    create_learning_rate_schedule
from Project_2_Image_Classification.src.training_routines.loss_function import LossConfig, LossFunction, \
    resolve_loss_function
from Project_2_Image_Classification.src.training_routines.optimizers import OptimizerConfig, create_optimizer
from Project_2_Image_Classification.src.utils.logging import get_logger
from Project_2_Image_Classification.src.visualization.style import PlotStyleConfig
from Project_2_Image_Classification.src.visualization.training_vis import TrainingVisualizer, TrainingVisualizerConfig

__all__ = [
    "TrainerConfig",
    "TrainingResult",
    "Trainer",
]


@dataclass(slots=True)
class TrainerConfig:
    """Hyper-parameter configuration used by :class:`Trainer`.

    Parameters
    ----------
    output_dir:
        Directory where checkpoints, metrics and visualisations are stored.
    num_epochs:
        Number of epochs to iterate over the training data.
    batch_size:
        Mini-batch size used for optimisation.
    eval_batch_size:
        Batch size used for evaluation.  When ``None`` the training batch size is
        reused.
    seed:
        Base random seed controlling weight initialisation and data shuffling.
    log_every:
        Interval (in optimisation steps) at which training metrics are logged.
    loss:
        Loss configuration passed to :func:`resolve_loss_function`.
    optimizer:
        Configuration selecting the Optax optimiser.
    scheduler:
        Configuration describing the learning-rate schedule.
    evaluate_on_test:
        If ``True`` the test split is evaluated after training has finished.
    style_config:
        Optional plotting style overriding the project defaults for figures.
    metrics:
        Tuple listing the metrics that should be visualised.  The trainer always
        records ``loss`` and ``accuracy``.
    """

    output_dir: Path
    num_epochs: int = 20
    batch_size: int = 128
    eval_batch_size: int | None = None
    seed: int = 0
    log_every: int = 100
    loss: LossConfig = field(default_factory=LossConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: LRSchedulerConfig = field(default_factory=LRSchedulerConfig)
    evaluate_on_test: bool = True
    style_config: PlotStyleConfig | None = None
    metrics: Sequence[str] = ("loss", "accuracy")

    def __post_init__(self) -> None:
        if self.num_epochs <= 0:
            raise ValueError("'num_epochs' must be a positive integer.")
        if self.batch_size <= 0:
            raise ValueError("'batch_size' must be a positive integer.")
        if self.eval_batch_size is not None and self.eval_batch_size <= 0:
            raise ValueError("'eval_batch_size' must be a positive integer when provided.")
        if self.log_every <= 0:
            raise ValueError("'log_every' must be a positive integer.")
        if not isinstance(self.metrics, Sequence) or not self.metrics:
            raise ValueError("'metrics' must be a non-empty sequence of metric identifiers.")

    @property
    def evaluation_batch_size(self) -> int:
        """Return the batch size used for evaluation loops."""

        return self.eval_batch_size or self.batch_size

    @classmethod
    def from_dict(cls, config: Mapping[str, object]) -> "TrainerConfig":
        """Construct a configuration from a nested mapping."""

        data = dict(config)
        if "output_dir" in data and not isinstance(data["output_dir"], Path):
            data["output_dir"] = Path(data["output_dir"])
        if "loss" in data and not isinstance(data["loss"], LossConfig):
            data["loss"] = LossConfig(**data["loss"])
        if "optimizer" in data and not isinstance(data["optimizer"], OptimizerConfig):
            data["optimizer"] = OptimizerConfig(**data["optimizer"])
        if "scheduler" in data and not isinstance(data["scheduler"], LRSchedulerConfig):
            data["scheduler"] = LRSchedulerConfig(**data["scheduler"])
        if "style_config" in data and not isinstance(data["style_config"], PlotStyleConfig):
            data["style_config"] = PlotStyleConfig(**data["style_config"])
        if "metrics" in data and not isinstance(data["metrics"], Sequence):
            raise TypeError("'metrics' must be provided as a sequence of metric names.")
        return cls(**data)

    def to_dict(self) -> Dict[str, object]:
        """Convert the configuration into a serialisable dictionary."""

        def _convert(value: object) -> object:
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
                return [_convert(item) for item in value]
            if isinstance(value, Mapping):
                return {key: _convert(item) for key, item in value.items()}
            if hasattr(value, "__dataclass_fields__"):
                return {key: _convert(item) for key, item in asdict(value).items()}
            return value

        raw = asdict(self)
        return {key: _convert(value) for key, value in raw.items()}


@dataclass(slots=True)
class TrainingResult:
    """Container bundling the artefacts produced during training."""

    state: "TrainingState"
    history: pd.DataFrame
    history_path: Path
    metrics_path: Path
    figure_paths: Dict[str, Path]
    best_validation_metrics: Mapping[str, float] | None
    test_metrics: Mapping[str, float] | None
    checkpoint_path: Path
    trainer_config_path: Path


class TrainingState(train_state.TrainState):
    """Extension of :class:`flax.training.train_state.TrainState` with RNG state."""

    dropout_rng: jax.Array | None = None
    batch_stats: FrozenDict | None = None

    @classmethod
    def create(
            cls,
            *,
            apply_fn: nn.Module,
            params: Mapping[str, jnp.ndarray],
            tx: optax.GradientTransformation,
            **kwargs,
    ) -> "TrainingState":
        opt_state = tx.init(params)
        return cls(
            step=jnp.array(0),
            apply_fn=apply_fn,
            params=params,
            tx=tx,
            opt_state=opt_state,
            **kwargs,
        )


class Trainer:
    """High level training loop coordinating optimisation and evaluation."""

    def __init__(self, model: nn.Module, config: TrainerConfig) -> None:
        self._model = model
        self._config = config
        self._logger = get_logger(self.__class__.__name__)
        self._loss_fn: LossFunction = resolve_loss_function(config.loss)

        self._output_dir = self._config.output_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)
        figures_dir = self._output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        self._visualizer = TrainingVisualizer(
            TrainingVisualizerConfig(
                output_directory=figures_dir,
                style_config=self._config.style_config,
            )
        )
        self._trainer_config_path = self._persist_trainer_config()

        self._logger.info(
            "Trainer initialised (epochs=%d, batch_size=%d, eval_batch_size=%d).",
            self._config.num_epochs,
            self._config.batch_size,
            self._config.evaluation_batch_size,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def train(self, dataset: PreparedDataset) -> TrainingResult:
        """Execute the optimisation loop on ``dataset``."""

        train_split = self._require_split(dataset, "train")
        validation_split = dataset.splits.get("validation")
        test_split = dataset.splits.get("test") if self._config.evaluate_on_test else None

        rng = jax.random.PRNGKey(self._config.seed)
        init_rng, dropout_rng, data_rng = jax.random.split(rng, 3)

        schedule = self._create_schedule(train_split)
        optimizer = create_optimizer(self._config.optimizer, schedule)

        example = jnp.asarray(train_split.images[:1])
        variables = self._model.init(init_rng, example, train=True)
        params = variables["params"]
        batch_stats = variables.get("batch_stats", None)

        state = TrainingState.create(
            apply_fn=self._model.apply,
            params=params,
            tx=optimizer,
            dropout_rng=dropout_rng,
            batch_stats=batch_stats,
        )

        history_records: List[Dict[str, float]] = []
        best_validation_metrics: Dict[str, float] | None = None
        best_validation_score = float("-inf")

        train_step_fn = self._build_train_step()
        eval_step_fn = self._build_eval_step()

        for epoch in range(1, self._config.num_epochs + 1):
            data_rng, epoch_rng = jax.random.split(data_rng)
            state, train_metrics = self._run_training_epoch(
                state,
                train_split,
                epoch_rng,
                train_step_fn,
            )

            lr_step = max(0, int(state.step) - 1)
            record: Dict[str, float] = {
                "epoch": float(epoch),
                "train_loss": train_metrics["loss"],
                "train_accuracy": train_metrics["accuracy"],
                "learning_rate": float(schedule(lr_step)),
            }

            if validation_split is not None:
                validation_metrics = self._evaluate_split(state, validation_split, eval_step_fn)
                if validation_metrics:
                    record.update({f"validation_{name}": value for name, value in validation_metrics.items()})

                    val_accuracy = validation_metrics.get("accuracy")
                    if val_accuracy is not None and val_accuracy > best_validation_score:
                        best_validation_score = val_accuracy
                        best_validation_metrics = dict(validation_metrics)
                else:
                    self._logger.warning(
                        "Validation split yielded no metrics; skipping validation logging for this epoch.")

            history_records.append(record)

            self._logger.info(
                "Epoch %d/%d - train_loss=%.4f train_acc=%.3f%s",
                epoch,
                self._config.num_epochs,
                record["train_loss"],
                record["train_accuracy"],
                f" val_acc={record.get('validation_accuracy', float('nan')):.3f}" if validation_split is not None else "",
            )

        history_df = pd.DataFrame(history_records)
        history_path = self._output_dir / "training_history.xlsx"
        history_df.to_excel(history_path, index=False)
        history_df.to_csv(self._output_dir / "training_history.csv", index=False)
        self._logger.info("Persisted training history to %s", history_path)

        metrics_summary: MutableMapping[str, Mapping[str, float] | None] = {
            "best_validation": best_validation_metrics,
            "final_train": {k: history_records[-1][k] for k in history_records[-1] if k.startswith("train_")},
        }

        test_metrics: Mapping[str, float] | None = None
        if test_split is not None:
            test_metrics = self._evaluate_split(state, test_split, eval_step_fn)
            metrics_summary["test"] = test_metrics
            if test_metrics:
                self._logger.info(
                    "Evaluation on test split - loss=%.4f acc=%.3f",
                    test_metrics.get("loss", float("nan")),
                    test_metrics.get("accuracy", float("nan")),
                )
            else:
                self._logger.warning("Test split evaluation returned no metrics. The split may be empty.")

        metrics_path = self._output_dir / "metrics_summary.json"
        metrics_path.write_text(json.dumps(metrics_summary, indent=2), encoding="utf-8")
        self._logger.info("Stored metrics summary in %s", metrics_path)

        figure_paths = self._visualizer.save_learning_curves(history_df, metrics=self._config.metrics)
        try:
            figure_paths["learning_rate"] = self._visualizer.save_learning_rate_curve(history_df)
        except ValueError:
            self._logger.warning(
                "Learning-rate curve could not be generated because the history lacks a learning_rate column.")

        checkpoint_path = self._save_checkpoint(state)

        return TrainingResult(
            state=state,
            history=history_df,
            history_path=history_path,
            metrics_path=metrics_path,
            figure_paths=figure_paths,
            best_validation_metrics=best_validation_metrics,
            test_metrics=test_metrics,
            checkpoint_path=checkpoint_path,
            trainer_config_path=self._trainer_config_path,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _create_schedule(self, train_split: DatasetSplit):
        steps_per_epoch = math.ceil(train_split.images.shape[0] / self._config.batch_size)
        total_steps = self._config.num_epochs * steps_per_epoch
        return create_learning_rate_schedule(self._config.scheduler, total_steps=total_steps)

    def _persist_trainer_config(self) -> Path:
        """Persist the trainer configuration used for this run."""

        config_path = self._output_dir / "trainer_config.yaml"
        config_path.write_text(yaml.safe_dump(self._config.to_dict(), sort_keys=False), encoding="utf-8")
        self._logger.debug("Stored trainer configuration in %s", config_path)
        return config_path

    def _save_checkpoint(self, state: TrainingState) -> Path:
        """Serialise the trained model parameters to disk."""

        checkpoint_dir = self._output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_path = checkpoint_dir / "final_params.msgpack"
        checkpoint_path.write_bytes(serialization.to_bytes(state.params))
        metadata = {"step": int(jax.device_get(state.step))}
        (checkpoint_dir / "checkpoint_meta.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        self._logger.info("Saved model checkpoint to %s", checkpoint_path)
        return checkpoint_path

    def _build_train_step(self):
        loss_fn = self._loss_fn

        def train_step(state: TrainingState, batch: Mapping[str, jnp.ndarray]
                       ) -> tuple[TrainingState, Dict[str, float]]:
            dropout_rng, new_dropout_rng = jax.random.split(state.dropout_rng)

            has_batch_stats = state.batch_stats is not None

            def loss_with_logits(params: Mapping[str, jnp.ndarray]) -> tuple[jnp.ndarray, jnp.ndarray]:
                variables = {"params": params}

                if has_batch_stats:
                    variables["batch_stats"] = state.batch_stats

                if has_batch_stats:
                    logits, new_model_state = state.apply_fn(
                        variables,
                        batch["images"],
                        train=True,
                        rngs={"dropout": dropout_rng},
                        mutable=["batch_stats"],
                    )
                else:
                    # no BN → no mutable
                    logits = state.apply_fn(
                        variables,
                        batch["images"],
                        train=True,
                        rngs={"dropout": dropout_rng},
                    )
                    new_model_state = {}

                loss = loss_fn(logits, batch["labels"])
                return loss, (logits, new_model_state)

            (loss, (logits, new_model_state)), grads = jax.value_and_grad(
                loss_with_logits, has_aux=True
            )(state.params)
            state = state.apply_gradients(grads=grads)
            updates = {"dropout_rng": new_dropout_rng}

            if has_batch_stats:
                updates["batch_stats"] = new_model_state["batch_stats"]

            state = state.replace(**updates)

            metrics = self._compute_metrics(logits, batch["labels"], loss)
            return state, metrics

        return jax.jit(train_step)

    def _build_eval_step(self):
        loss_fn = self._loss_fn

        def eval_step(state: TrainingState, batch: Mapping[str, jnp.ndarray]) -> Dict[str, float]:
            variables = {"params": state.params}
            if state.batch_stats is not None:
                variables["batch_stats"] = state.batch_stats

            logits = state.apply_fn(
                variables,
                batch["images"],
                train=False,
            )
            loss = loss_fn(logits, batch["labels"])
            return self._compute_metrics(logits, batch["labels"], loss)

        return jax.jit(eval_step)

    def _run_training_epoch(
            self,
            state: TrainingState,
            split: DatasetSplit,
            rng: jax.Array,
            train_step_fn,
    ) -> tuple[TrainingState, Dict[str, float]]:
        metrics: List[Dict[str, float]] = []
        batch_iter = self._iterate_batches(
            split,
            batch_size=self._config.batch_size,
            shuffle=True,
            rng=rng,
        )
        for step, batch in enumerate(batch_iter, start=1):
            state, batch_metrics = train_step_fn(state, batch)
            metrics.append(batch_metrics)
            if step % self._config.log_every == 0:
                self._logger.debug(
                    "Step %d - loss=%.4f acc=%.3f",
                    int(state.step),
                    batch_metrics["loss"],
                    batch_metrics["accuracy"],
                )
        return state, self._aggregate_metrics(metrics)

    def _evaluate_split(
            self,
            state: TrainingState,
            split: DatasetSplit,
            eval_step_fn,
    ) -> Dict[str, float]:
        metrics: List[Dict[str, float]] = []
        batch_iter = self._iterate_batches(
            split,
            batch_size=self._config.evaluation_batch_size,
            shuffle=False,
            rng=None,
        )
        for batch in batch_iter:
            metrics.append(eval_step_fn(state, batch))
        return self._aggregate_metrics(metrics)

    def _iterate_batches(
            self,
            split: DatasetSplit,
            *,
            batch_size: int,
            shuffle: bool,
            rng: jax.Array | None,
    ) -> Iterator[Dict[str, jnp.ndarray]]:
        images = split.images
        labels = split.labels
        num_samples = images.shape[0]

        if shuffle:
            if rng is None:
                raise ValueError("RNG must be provided when shuffling is enabled.")
            permutation = jax.random.permutation(rng, num_samples)
        else:
            permutation = jnp.arange(num_samples)

        for start in range(0, num_samples, batch_size):
            indices = permutation[start:start + batch_size]
            batch_images = jax.device_put(images[indices])
            batch_labels = jax.device_put(labels[indices])
            yield {
                "images": batch_images,
                "labels": batch_labels,
            }

    def _aggregate_metrics(self, metrics: Iterable[Mapping[str, jnp.ndarray]]) -> Dict[str, float]:
        collected: Dict[str, List[float]] = {}
        for metric in metrics:
            for name, value in metric.items():
                collected.setdefault(name, []).append(float(jax.device_get(value)))

        return {
            name: float(sum(values) / len(values))
            for name, values in collected.items()
            if values
        }

    def _compute_metrics(
            self,
            logits: jnp.ndarray,
            labels: jnp.ndarray,
            loss: jnp.ndarray,
    ) -> Dict[str, jnp.ndarray]:
        predictions = jnp.argmax(logits, axis=-1)
        accuracy = jnp.mean(predictions == labels)
        return {
            "loss": loss,
            "accuracy": accuracy,
        }

    @staticmethod
    def _require_split(dataset: PreparedDataset, split_name: str) -> DatasetSplit:
        try:
            return dataset.splits[split_name]
        except KeyError as exc:
            raise KeyError(f"Dataset does not contain a '{split_name}' split.") from exc
