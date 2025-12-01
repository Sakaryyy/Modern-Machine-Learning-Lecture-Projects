"""Training and evaluation utilities for the Conway Transformer."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
from flax import serialization

from Project_3_Conways_Game_Of_Life_Transformer.src.config.data_config import DataConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import TransformerConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.training_config import TrainState, TrainingConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.data_functions.data_pipelines import (
    DatasetSplits,
    prepare_gol_dataset,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.model.gol_transformer import (
    GameOfLifeTransformer,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.training.loss import (
    binary_cross_entropy_with_logits,
    log_likelihood_from_logits,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.training.metrics import (
    accuracy_from_logits,
    calibration_curve,
    compute_auc,
    compute_roc_curve,
    negative_log_likelihood_scores,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.logging import get_logger
from Project_3_Conways_Game_Of_Life_Transformer.src.visualization.plotting_utils import (
    plot_calibration_curve,
    plot_multiple_roc_curves,
    plot_training_curves,
    set_scientific_plot_style,
)

LOGGER = get_logger(__name__)


def data_loader(
        inputs: np.ndarray, targets: np.ndarray, batch_size: int, rng: np.random.Generator
) -> Iterable[Dict[str, np.ndarray]]:
    """Yield shuffled mini-batches as dictionaries."""

    num_samples = inputs.shape[0]
    indices = np.arange(num_samples)
    rng.shuffle(indices)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        idx = indices[start:end]
        yield {"x": inputs[idx], "y": targets[idx]}


def create_train_state(
        rng: jax.random.PRNGKey,
        model: GameOfLifeTransformer,
        input_shape: Tuple[int, int, int],
        config: TrainingConfig
) -> TrainState:
    """Initialize model parameters and optimizer state.

    Parameters
    ----------
    rng : jax.random.PRNGKey
        JAX random key for parameter initialization and dropout.
    model : GameOfLifeTransformer
        Flax module instance to be trained.
    input_shape : tuple of int
        Shape of a single input batch, excluding the batch dimension.
        For example (height, width) for Game of Life grids.
    config : TrainingConfig
        Training hyperparameters.

    Returns
    -------
    state : TrainState
        Initialized train state with parameters and optimizer.
    """
    # Dummy batch for initialization
    dummy_batch = jnp.zeros((1,) + input_shape, dtype=jnp.int32)

    params = model.init(rng, dummy_batch, train=True)["params"]
    tx = optax.adam(config.learning_rate)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx)


def make_train_and_eval_step(model: GameOfLifeTransformer):
    """Create jitted train and eval step functions for the given model.

    This helper closes over `model.apply` to keep the signatures clean.

    Parameters
    ----------
    model : GameOfLifeTransformer
        Model to be used in the steps.

    Returns
    -------
    train_step : callable
        Function (state, batch, rng) -> (new_state, metrics).
    eval_step : callable
        Function (state, batch) -> metrics.
    """

    def train_step(
            state: TrainState,
            batch: Dict[str, np.ndarray],
            rng: jax.random.PRNGKey,
    ) -> Tuple[TrainState, Dict[str, jnp.ndarray]]:
        """Single training step.

        Parameters
        ----------
        state : TrainState
            Current train state with parameters and optimizer state.
        batch : dict
            Batch dictionary with keys "x" and "y" as np.ndarray or jnp.ndarray.
        rng : jax.random.PRNGKey
            Random key used for dropout inside the model.

        Returns
        -------
        new_state : TrainState
            Updated train state.
        metrics : dict
            Dictionary with scalar loss and accuracy.
        """
        x = jnp.array(batch["x"])
        y = jnp.array(batch["y"])

        def loss_fn(params):
            logits = state.apply_fn(
                {"params": params},
                x,
                train=True,
                rngs={"dropout": rng},
            )
            loss = binary_cross_entropy_with_logits(logits, y)
            return loss, logits

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, logits), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        acc = accuracy_from_logits(logits, y)
        return new_state, {"loss": loss, "accuracy": acc}

    def eval_step(
            state: TrainState,
            batch: Dict[str, np.ndarray],
    ) -> Dict[str, jnp.ndarray]:
        """Single evaluation step without gradient updates.

        Parameters
        ----------
        state : TrainState
            Current train state with parameters.
        batch : dict
            Batch dictionary with keys "x" and "y".

        Returns
        -------
        metrics : dict
            Dictionary with scalar loss and accuracy.
        """
        x = jnp.array(batch["x"])
        y = jnp.array(batch["y"])

        logits = state.apply_fn(
            {"params": state.params},
            x,
            train=False,
        )
        loss = binary_cross_entropy_with_logits(logits, y)
        acc = accuracy_from_logits(logits, y)
        return {"loss": loss, "accuracy": acc, "logits": logits}

    return jax.jit(train_step), jax.jit(eval_step)


def _save_checkpoint(output_dir: Path, state: TrainState, epoch: int) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.msgpack"
    ckpt_path.write_bytes(serialization.to_bytes(state.params))
    LOGGER.info("Saved checkpoint to %s", ckpt_path)
    return ckpt_path


def _save_configs(output_dir: Path, data_cfg: DataConfig, model_cfg: TransformerConfig,
                  train_cfg: TrainingConfig) -> None:
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    for name, cfg in {"data": data_cfg, "model": model_cfg, "training": train_cfg}.items():
        cfg_path = output_dir / "configs" / f"{name}.json"
        cfg_path.write_text(json.dumps(asdict(cfg), indent=2))
        LOGGER.info("Saved %s configuration to %s", name, cfg_path)


def _record_history(output_dir: Path, history: Dict[str, list]) -> None:
    df = pd.DataFrame(history)
    csv_path = output_dir / "training_history.csv"
    xlsx_path = output_dir / "training_history.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    LOGGER.info("Persisted training metrics to %s and %s", csv_path, xlsx_path)


def train_and_evaluate(
        data_cfg: DataConfig,
        model_cfg: TransformerConfig,
        train_cfg: TrainingConfig,
        output_root: Path,
        seed: int = 0,
) -> Tuple[TrainState, DatasetSplits, Path]:
    """Full training pipeline including plots and artefact saving."""

    output_root.mkdir(parents=True, exist_ok=True)
    _save_configs(output_root, data_cfg, model_cfg, train_cfg)

    set_scientific_plot_style()

    rng = np.random.default_rng(seed)
    splits = prepare_gol_dataset(data_cfg)

    model = GameOfLifeTransformer(config=model_cfg)
    key = jax.random.PRNGKey(seed)
    state = create_train_state(key, model, (data_cfg.height, data_cfg.width), train_cfg)
    train_step, eval_step = make_train_and_eval_step(model)

    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}

    for epoch in range(1, train_cfg.num_epochs + 1):
        key, epoch_key = jax.random.split(key)
        batches = data_loader(splits.x_train, splits.y_train, train_cfg.batch_size, rng)
        epoch_losses, epoch_accs = [], []

        for batch in batches:
            epoch_key, step_key = jax.random.split(epoch_key)
            key, step_key = jax.random.split(key)
            state, metrics = train_step(state, batch, step_key)
            epoch_losses.append(metrics["loss"])
            epoch_accs.append(metrics["accuracy"])

        train_loss = float(jnp.stack(epoch_losses).mean())
        train_acc = float(jnp.stack(epoch_accs).mean())

        val_batches = data_loader(splits.x_val, splits.y_val, train_cfg.batch_size, rng)
        val_loss_list, val_acc_list = [], []
        for batch in val_batches:
            metrics = eval_step(state, batch)
            val_loss_list.append(metrics["loss"])
            val_acc_list.append(metrics["accuracy"])

        val_loss = float(jnp.stack(val_loss_list).mean())
        val_acc = float(jnp.stack(val_acc_list).mean())

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        LOGGER.info(
            "Epoch %03d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        _save_checkpoint(output_root, state, epoch)

    plot_training_curves(history, title="Training diagnostics", save_path=output_root / "training_curves.png")
    _record_history(output_root, history)

    # Test evaluation
    test_batches = data_loader(splits.x_test, splits.y_test, train_cfg.batch_size, rng)
    test_loss_list, test_acc_list = [], []
    logits_all = []
    for batch in test_batches:
        metrics = eval_step(state, batch)
        test_loss_list.append(metrics["loss"])
        test_acc_list.append(metrics["accuracy"])
        logits_all.append(np.array(metrics["logits"]))

    test_loss = float(jnp.stack(test_loss_list).mean())
    test_acc = float(jnp.stack(test_acc_list).mean())
    LOGGER.info("Test loss=%.4f accuracy=%.4f", test_loss, test_acc)

    logits_concat = np.concatenate(logits_all, axis=0)
    probs = jax.nn.sigmoid(logits_concat)
    bin_centers, empirical_freq = calibration_curve(probs, splits.y_test)
    plot_calibration_curve(probs, splits.y_test, save_path=output_root / "calibration_curve.png")

    calib_df = pd.DataFrame({"bin_center": bin_centers, "empirical_freq": empirical_freq})
    calib_df.to_csv(output_root / "calibration_curve.csv", index=False)

    # Optional anomaly evaluation
    if data_cfg.anomaly_detection and splits.labels_test is not None:
        log_likelihoods = log_likelihood_from_logits(jnp.array(logits_concat), jnp.array(splits.y_test))
        scores = negative_log_likelihood_scores(np.array(log_likelihoods))
        _, fpr, tpr = compute_roc_curve(scores=scores, labels=splits.labels_test)
        auc = compute_auc(fpr, tpr)
        LOGGER.info("Anomaly ROC AUC=%.4f", auc)
        plot_multiple_roc_curves({f"H{data_cfg.height}W{data_cfg.width}": (fpr, tpr)},
                                 save_path=output_root / "roc_curve.png")
        pd.DataFrame({"fpr": fpr, "tpr": tpr}).to_csv(output_root / "roc_curve.csv", index=False)

    metrics_path = output_root / "summary.json"
    metrics_path.write_text(json.dumps({"test_loss": test_loss, "test_accuracy": test_acc}, indent=2))
    LOGGER.info("Wrote summary to %s", metrics_path)

    return state, splits, output_root


def load_params(checkpoint_path: Path, model: GameOfLifeTransformer, input_shape: Tuple[int, int]) -> TrainState:
    """Load a saved parameter file into a fresh TrainState."""

    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1,) + input_shape, dtype=jnp.int32)
    variables = model.init(rng, dummy, train=False)
    params = serialization.from_bytes(variables["params"], checkpoint_path.read_bytes())
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.0))


def generate_predictions(
        checkpoint_path: Path,
        model_cfg: TransformerConfig,
        inputs: np.ndarray,
        batch_size: int,
) -> np.ndarray:
    """Run a trained model in generation/evaluation mode."""

    model = GameOfLifeTransformer(config=model_cfg)
    state = load_params(checkpoint_path, model, (inputs.shape[1], inputs.shape[2]))

    def forward(params, x_batch):
        logits = state.apply_fn({"params": params}, x_batch, train=False)
        return jax.nn.sigmoid(logits)

    forward_jit = jax.jit(forward)

    num_samples = inputs.shape[0]
    outputs = np.empty_like(inputs, dtype=np.float32)
    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        x_batch = jnp.array(inputs[start:end])
        probs = forward_jit(state.params, x_batch)
        outputs[start:end] = np.array(probs)

    return outputs
