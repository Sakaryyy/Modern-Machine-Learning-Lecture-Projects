"""Inference utilities for evaluating trained models."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Mapping, MutableMapping, Sequence

import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np
import pandas as pd
import yaml
from flax import linen as nn
from flax import serialization

from Project_2_Image_Classification.src.data_loading.data_load_and_save import PreparedDataset
from Project_2_Image_Classification.src.models import (
    BaselineModelConfig,
    ImageClassifierConfig,
    create_baseline_model,
    create_image_classifier,
)
from Project_2_Image_Classification.src.training_routines.loss_function import LossConfig, resolve_loss_function
from Project_2_Image_Classification.src.utils.helper import log_jax_runtime_info
from Project_2_Image_Classification.src.utils.logging import get_logger
from Project_2_Image_Classification.src.visualization.classification_vis import (
    ClassificationVisualizer,
    ClassificationVisualizerConfig,
)
from Project_2_Image_Classification.src.visualization.style import PlotStyleConfig

Batch = Mapping[str, jnp.ndarray]


@dataclass(slots=True)
class ClassificationConfig:
    """Configuration required to evaluate a trained model."""

    run_directory: Path
    batch_size: int = 256
    output_directory: Path | None = None
    save_predictions: bool = True
    style_config: PlotStyleConfig | None = None

    def resolve_output_directory(self) -> Path:
        """Return the directory that will store evaluation artefacts."""

        directory = self.output_directory or (self.run_directory / "classification")
        directory.mkdir(parents=True, exist_ok=True)
        return directory


class ClassificationRunner:
    """Load a checkpoint, run inference and compute evaluation metrics."""

    MODEL_DEFINITION_FILENAME = "model_definition.yaml"

    def __init__(self, config: ClassificationConfig) -> None:
        self._config = config
        self._logger = get_logger(self.__class__.__name__)
        self._device = log_jax_runtime_info()
        self._output_dir = self._config.resolve_output_directory()
        self._visualizer = ClassificationVisualizer(
            ClassificationVisualizerConfig(
                output_directory=self._output_dir,
                style_config=self._config.style_config,
            )
        )

    def run(self, dataset: PreparedDataset) -> Mapping[str, float]:
        """Evaluate the checkpoint stored in :attr:`ClassificationConfig.run_directory`."""

        definition = self._load_model_definition()
        model_name = definition["model_name"]
        model_config_dict = definition["config"]
        loss_config = LossConfig(**definition.get("loss", {"name": "cross_entropy"}))
        trainer_config = definition.get("trainer", {})
        if "eval_batch_size" in trainer_config and trainer_config["eval_batch_size"] is not None:
            batch_size = int(trainer_config["eval_batch_size"])
        else:
            batch_size = self._config.batch_size

        model = self._instantiate_model(model_name, model_config_dict)
        params = self._load_parameters()
        loss_fn = resolve_loss_function(loss_config)

        test_split = dataset.splits.get("test")
        if test_split is None:
            raise KeyError("Prepared dataset must contain a 'test' split for evaluation.")

        metrics, predictions = self._evaluate(model, params, test_split, batch_size, loss_fn)

        class_names = dataset.metadata.get("class_names") if dataset.metadata else None
        if class_names is None:
            class_names = [str(index) for index in range(int(jnp.max(test_split.labels)) + 1)]

        confusion = self._compute_confusion_matrix(predictions["labels"], predictions["predictions"], len(class_names))
        self._visualizer.save_confusion_matrix(confusion, class_names)
        self._visualizer.save_metrics_table(metrics)

        images_np = np.asarray(test_split.images)
        labels_np = predictions["labels"].astype(int)
        preds_np = predictions["predictions"].astype(int)
        probabilities = predictions.get("probabilities")

        self._visualizer.save_prediction_gallery(
            images_np,
            labels_np,
            preds_np,
            class_names,
            probabilities=probabilities,
        )
        self._visualizer.save_per_class_accuracy(labels_np, preds_np, class_names)
        if probabilities is not None:
            self._visualizer.save_confidence_histogram(probabilities, labels_np, preds_np)

        if self._config.save_predictions:
            self._save_predictions(predictions, class_names)

        metrics_path = self._output_dir / "classification_metrics.json"
        metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        self._logger.info("Classification results persisted to %s", metrics_path)

        return metrics

    def _load_model_definition(self) -> Mapping[str, Any]:
        definition_path = self._config.run_directory / self.MODEL_DEFINITION_FILENAME
        if not definition_path.exists():
            raise FileNotFoundError(
                f"Model definition file '{definition_path}' not found. Training runs must store the definition."
            )
        with definition_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, Mapping):
            raise TypeError("Model definition must be a mapping.")
        return data

    def _instantiate_model(self, model_name: str, model_config_dict: Mapping[str, Any]) -> nn.Module:
        normalized = model_name.lower()
        if normalized == "baseline":
            config = BaselineModelConfig(**model_config_dict)
            self._logger.info("Loaded baseline model for classification with hidden_units=%d", config.hidden_units)
            return create_baseline_model(config)
        if normalized in {"cnn", "image_classifier"}:
            config = ImageClassifierConfig(**model_config_dict)
            self._logger.info(
                "Loaded CNN classifier with %d convolutional blocks and %d dense blocks.",
                len(config.conv_blocks),
                len(config.dense_blocks),
            )
            return create_image_classifier(config)
        raise ValueError(f"Unsupported model '{model_name}' in model definition.")

    def _load_parameters(self) -> Mapping[str, Any]:
        checkpoint_path = self._config.run_directory / "checkpoints" / "final_params.msgpack"
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint file '{checkpoint_path}' not found.")
        params_bytes = checkpoint_path.read_bytes()
        params = serialization.from_bytes(None, params_bytes)
        self._logger.info("Loaded parameters from %s", checkpoint_path)
        return params

    def _evaluate(
            self,
            model: nn.Module,
            params: Mapping[str, Any],
            split,
            batch_size: int,
            loss_fn,
    ) -> tuple[MutableMapping[str, float], Mapping[str, np.ndarray]]:
        batches = self._iterate_batches(split.images, split.labels, batch_size)
        metrics: MutableMapping[str, float] = {"loss": 0.0, "accuracy": 0.0}
        total_samples = 0
        all_predictions: list[int] = []
        all_labels: list[int] = []
        all_probabilities: list[np.ndarray] = []

        @jax.jit
        def forward(batch: Batch) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            logits = model.apply({"params": params}, batch["images"], train=False)
            batch_loss = loss_fn(logits, batch["labels"])
            probs = jnn.softmax(logits, axis=-1)
            return logits, batch_loss, probs

        for batch in batches:
            logits, batch_loss, probs = forward(batch)
            preds = jnp.argmax(logits, axis=-1)
            labels = batch["labels"]

            batch_size_actual = int(labels.shape[0])
            total_samples += batch_size_actual
            metrics["loss"] += float(batch_loss) * batch_size_actual
            metrics["accuracy"] += float(jnp.sum(preds == labels))

            all_predictions.extend(np.asarray(preds).tolist())
            all_labels.extend(np.asarray(labels).tolist())
            all_probabilities.append(np.asarray(probs))

        metrics["loss"] /= total_samples
        metrics["accuracy"] /= total_samples
        probabilities = np.concatenate(all_probabilities, axis=0) if all_probabilities else None
        prediction_dict = {
            "predictions": np.array(all_predictions),
            "labels": np.array(all_labels),
        }
        if probabilities is not None:
            prediction_dict["probabilities"] = probabilities
        return metrics, prediction_dict

    def _iterate_batches(
            self,
            images: jnp.ndarray,
            labels: jnp.ndarray,
            batch_size: int,
    ) -> Iterator[Batch]:
        num_samples = images.shape[0]
        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_images = jax.device_put(images[start:end])
            batch_labels = jax.device_put(labels[start:end])
            yield {"images": batch_images, "labels": batch_labels}

    def _save_predictions(self, predictions: Mapping[str, np.ndarray], class_names: Sequence[str]) -> None:
        path = self._output_dir / "predictions.csv"
        labels = predictions["labels"].astype(int)
        preds = predictions["predictions"].astype(int)
        data = {
            "true_label": [class_names[label] for label in labels],
            "predicted_label": [class_names[label] for label in preds],
            "true_index": labels,
            "predicted_index": preds,
        }
        if "probabilities" in predictions:
            confidences = predictions["probabilities"][np.arange(len(preds)), preds]
            data["confidence"] = confidences
        frame = pd.DataFrame(data)
        frame.to_csv(path, index=False)
        self._logger.info("Saved individual predictions to %s", path)

    def _compute_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray, num_classes: int) -> np.ndarray:
        matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
        for true, pred in zip(labels, predictions, strict=False):
            matrix[int(true), int(pred)] += 1
        return matrix
