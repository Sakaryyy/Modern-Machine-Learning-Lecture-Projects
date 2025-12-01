"""Training and evaluation utilities for the Conway Transformer."""

import contextlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pandas as pd
from flax import serialization

from Project_3_Conways_Game_Of_Life_Transformer.src.config.data_config import DataConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import TransformerConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.training_config import TrainState, TrainingConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.data_functions.data_pipelines import (
    DatasetSplits,
    generate_deterministic_pairs,
    generate_stochastic_pairs,
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
    compute_brier_score,
    compute_roc_curve,
    negative_log_likelihood_scores,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.backend import log_jax_runtime_info
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.logging import get_logger
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.parameter_analysis import (
    count_parameters,
    estimate_attention_activation_memory,
    estimate_attention_flops,
    estimate_parameter_memory,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.visualization.plotting_utils import (
    plot_calibration_curve,
    plot_grid_difference,
    plot_grid_pair_examples,
    plot_grid_triplet,
    plot_multiple_roc_curves,
    plot_training_curves,
    set_scientific_plot_style,
)

LOGGER = get_logger(__name__)


def data_loader(
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        rng: np.random.Generator | None,
        shuffle: bool = True,
) -> Iterable[Dict[str, np.ndarray]]:
    """Yield mini-batches as dictionaries without materializing copies.

    Parameters
    ----------
    inputs : np.ndarray
        Array of input states.
    targets : np.ndarray
        Array of target states aligned with ``inputs``.
    batch_size : int
        Number of samples per batch.
    rng : np.random.Generator or None
        Random generator used for shuffling. Must be provided when
        ``shuffle`` is True.
    shuffle : bool, optional
        If True shuffle indices before batching. Validation and test
        loaders set this to False to keep deterministic ordering.

    Yields
    ------
    dict
        Dictionary with keys ``"x"`` and ``"y"`` containing batch views.
    """

    num_samples = inputs.shape[0]
    if shuffle:
        if rng is None:
            raise ValueError("rng must be provided when shuffle=True")
        indices = rng.permutation(num_samples)
    else:
        indices = np.arange(num_samples)

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        idx = indices[start:end]
        yield {"x": inputs[idx], "y": targets[idx]}


def _sample_density_schedule(num_samples: int, data_cfg: DataConfig, rng: np.random.Generator) -> np.ndarray:
    """Mirror the training density sampling strategy for new grids."""

    if data_cfg.density_range is not None:
        low, high = data_cfg.density_range
        return rng.uniform(low=low, high=high, size=(num_samples,)).astype(np.float32)
    return np.full((num_samples,), fill_value=float(data_cfg.density), dtype=np.float32)


def _build_learning_rate_schedule(config: TrainingConfig, total_steps: int) -> optax.Schedule:
    """Construct a warmup + decay schedule according to the config."""

    decay_steps = config.decay_steps or total_steps
    decay_steps = max(decay_steps, 1)
    warmup_steps = min(config.warmup_steps, decay_steps)

    if config.lr_schedule == "cosine":
        schedule = optax.warmup_cosine_decay_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            warmup_steps=warmup_steps,
            decay_steps=decay_steps,
            end_value=config.learning_rate * config.min_lr_ratio,
        )
    elif config.lr_schedule == "linear":
        schedule = optax.warmup_polynomial_schedule(
            init_value=0.0,
            peak_value=config.learning_rate,
            end_value=config.learning_rate * config.min_lr_ratio,
            power=1.0,
            transition_steps=decay_steps,
            warmup_steps=warmup_steps,
        )
    else:
        schedule = optax.constant_schedule(config.learning_rate)

    return schedule


def _build_optimizer(config: TrainingConfig, total_steps: int) -> Tuple[
    Callable[[int], float], optax.GradientTransformation]:
    """Create optimizer and schedule with optional gradient clipping."""

    lr_schedule = _build_learning_rate_schedule(config, total_steps)

    if config.optimizer == "adam":
        base_opt = optax.adam(
            learning_rate=lr_schedule,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
        )
    elif config.optimizer == "sgd":
        base_opt = optax.sgd(
            learning_rate=lr_schedule,
            momentum=config.beta1,
            nesterov=True,
        )
    else:
        base_opt = optax.adamw(
            learning_rate=lr_schedule,
            b1=config.beta1,
            b2=config.beta2,
            eps=config.eps,
            weight_decay=config.weight_decay,
        )

    transforms = []
    if config.max_grad_norm is not None:
        transforms.append(optax.clip_by_global_norm(config.max_grad_norm))
    transforms.append(base_opt)

    tx = optax.chain(*transforms)
    return lr_schedule, tx


def create_train_state(
        rng: jax.random.PRNGKey,
        model: GameOfLifeTransformer,
        input_shape: Tuple[int, int, int],
        config: TrainingConfig,
        steps_per_epoch: int,
) -> Tuple[TrainState, Callable[[int], float]]:
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
    total_steps = max(1, steps_per_epoch * config.num_epochs)
    lr_schedule, tx = _build_optimizer(config, total_steps)

    return TrainState.create(apply_fn=model.apply, params=params, tx=tx), lr_schedule


def make_train_and_eval_step(model: GameOfLifeTransformer, train_cfg: TrainingConfig):
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
            if train_cfg.l2_reg > 0.0:
                l2_term = sum(jnp.sum(jnp.square(p)) for p in jtu.tree_leaves(params))
                param_count = sum(p.size for p in jtu.tree_leaves(params))
                loss = loss + train_cfg.l2_reg * l2_term / jnp.maximum(param_count, 1)
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


def _evaluate_dataset(
        state: TrainState,
        eval_step: Callable[[TrainState, Dict[str, np.ndarray]], Dict[str, jnp.ndarray]],
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
) -> Tuple[float, float, np.ndarray]:
    """Run evaluation over a dataset and aggregate metrics."""

    eval_batches = data_loader(
        inputs,
        targets,
        batch_size,
        rng=None,
        shuffle=False,
    )

    loss_list, acc_list = [], []
    logits_all = []
    for batch in eval_batches:
        metrics = eval_step(state, batch)
        loss_list.append(metrics["loss"])
        acc_list.append(metrics["accuracy"])
        logits_all.append(np.array(metrics["logits"]))

    loss_value = float(jnp.stack(loss_list).mean())
    acc_value = float(jnp.stack(acc_list).mean())
    logits_concat = np.concatenate(logits_all, axis=0)
    return loss_value, acc_value, logits_concat


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
        cfg_path.write_text(json.dumps(asdict(cfg), indent=2, default=str))
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

    preferred_device = log_jax_runtime_info()
    device_context = (
        jax.default_device(preferred_device)
        if hasattr(jax, "default_device")
        else contextlib.nullcontext()
    )

    with device_context:
        splits = prepare_gol_dataset(data_cfg)

        LOGGER.info(
            "Loaded dataset with shapes train=%s val=%s test=%s",
            splits.x_train.shape,
            splits.x_val.shape,
            splits.x_test.shape,
        )

        example_pairs_path = output_root / "example_pairs.png"
        plot_grid_pair_examples(
            inputs=splits.x_train[: min(6, splits.x_train.shape[0])],
            targets=splits.y_train[: min(6, splits.y_train.shape[0])],
            save_path=example_pairs_path,
            title="Example Conway transitions (train split)",
        )
        LOGGER.info("Saved example grid transitions to %s", example_pairs_path)

        model = GameOfLifeTransformer(config=model_cfg)
        key = jax.random.PRNGKey(seed)
        steps_per_epoch = max(1, int(np.ceil(splits.x_train.shape[0] / train_cfg.batch_size)))
        state, lr_schedule = create_train_state(
            key, model, (data_cfg.height, data_cfg.width), train_cfg, steps_per_epoch
        )
        train_step, eval_step = make_train_and_eval_step(model, train_cfg)

        num_params = count_parameters(state.params)
        param_mem_mb = estimate_parameter_memory(num_params)
        attn_layer_mb, attn_total_mb = estimate_attention_activation_memory(
            batch_size=train_cfg.batch_size,
            height=data_cfg.height,
            width=data_cfg.width,
            config=model_cfg,
        )
        attn_flops = estimate_attention_flops(
            batch_size=train_cfg.batch_size,
            height=data_cfg.height,
            width=data_cfg.width,
            config=model_cfg,
        )
        analysis = {
            "num_parameters": num_params,
            "parameter_memory_mb": param_mem_mb,
            "attention_memory_per_layer_mb": attn_layer_mb,
            "attention_memory_total_mb": attn_total_mb,
            "attention_flops": attn_flops,
            "device": str(preferred_device),
        }
        analysis_path = output_root / "analysis" / "parameter_report.json"
        analysis_path.parent.mkdir(parents=True, exist_ok=True)
        analysis_path.write_text(json.dumps(analysis, indent=2))
        LOGGER.info(
            "Parameter report saved to %s (params=%d, memory=%.2f MB)",
            analysis_path,
            num_params,
            param_mem_mb,
        )

        history = {
            "train_loss": [],
            "train_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "learning_rate": [],
            "train_val_gap": [],
        }
        best_val_acc = -np.inf
        best_val_loss = float("inf")
        best_checkpoint = None

        for epoch in range(1, train_cfg.num_epochs + 1):
            key, epoch_key = jax.random.split(key)
            epoch_rng = np.random.default_rng(seed + epoch)
            batches = data_loader(
                splits.x_train,
                splits.y_train,
                train_cfg.batch_size,
                rng=epoch_rng,
                shuffle=True,
            )
            epoch_losses, epoch_accs = [], []

            for batch in batches:
                epoch_key, step_key = jax.random.split(epoch_key)
                state, metrics = train_step(state, batch, step_key)
                epoch_losses.append(metrics["loss"])
                epoch_accs.append(metrics["accuracy"])

            train_loss = float(jnp.stack(epoch_losses).mean())
            train_acc = float(jnp.stack(epoch_accs).mean())

            val_loss, val_acc, _ = _evaluate_dataset(
                state, eval_step, splits.x_val, splits.y_val, train_cfg.batch_size
            )
            lr_value = float(lr_schedule(int(state.step)))

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["learning_rate"].append(lr_value)
            history["train_val_gap"].append(train_acc - val_acc)

            LOGGER.info(
                (
                    "Epoch %03d train_loss=%.4f train_acc=%.4f val_loss=%.4f "
                    "val_acc=%.4f lr=%.6f"
                ),
                epoch,
                train_loss,
                train_acc,
                val_loss,
                val_acc,
                lr_value,
            )

            ckpt_path = _save_checkpoint(output_root, state, epoch)
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_val_loss = val_loss
                best_checkpoint = ckpt_path

        if best_checkpoint is not None:
            LOGGER.info(
                "Best validation accuracy=%.4f loss=%.4f saved at %s",
                best_val_acc,
                best_val_loss,
                best_checkpoint,
            )

        training_plot_path = output_root / "training_curves.png"
        plot_training_curves(history, title="Training diagnostics", save_path=training_plot_path)
        LOGGER.info("Saved training curves to %s", training_plot_path)
        _record_history(output_root, history)

        # Test evaluation
        test_loss, test_acc, logits_concat = _evaluate_dataset(
            state, eval_step, splits.x_test, splits.y_test, train_cfg.batch_size
        )
        LOGGER.info("Test loss=%.4f accuracy=%.4f", test_loss, test_acc)

        probs = jax.nn.sigmoid(logits_concat)
        bin_centers, empirical_freq = calibration_curve(probs, splits.y_test)
        calibration_path = output_root / "calibration_curve.png"
        plot_calibration_curve(probs, splits.y_test, save_path=calibration_path)
        calib_df = pd.DataFrame({"bin_center": bin_centers, "empirical_freq": empirical_freq})
        calib_df.to_csv(output_root / "calibration_curve.csv", index=False)
        LOGGER.info("Saved calibration diagnostics to %s", calibration_path)

        # Example prediction plots on the test set
        example_pred_path = output_root / "test_prediction_triplet.png"
        example_diff_path = output_root / "test_prediction_difference.png"
        sample_probs = np.array(probs[0])
        plot_grid_triplet(
            splits.x_test[0],
            splits.y_test[0],
            sample_probs,
            save_path=example_pred_path,
            title="Test prediction example",
        )
        plot_grid_difference(
            splits.y_test[0],
            (sample_probs > 0.5).astype(int),
            save_path=example_diff_path,
        )

        log_likelihoods = log_likelihood_from_logits(jnp.array(logits_concat), jnp.array(splits.y_test))
        test_nll = float(-log_likelihoods.mean())
        test_brier = compute_brier_score(np.asarray(probs), np.asarray(splits.y_test))

        roc_summary = None
        if data_cfg.anomaly_detection and splits.labels_test is not None:
            scores = negative_log_likelihood_scores(np.array(log_likelihoods))
            thresholds, fpr, tpr = compute_roc_curve(scores=scores, labels=splits.labels_test)
            auc = compute_auc(fpr, tpr)
            LOGGER.info("Anomaly ROC AUC=%.4f", auc)
            roc_path = output_root / "roc_curve.png"
            plot_multiple_roc_curves({f"H{data_cfg.height}W{data_cfg.width}": (fpr, tpr)},
                                     save_path=roc_path)
            pd.DataFrame({"threshold": thresholds, "fpr": fpr, "tpr": tpr}).to_csv(
                output_root / "roc_curve.csv", index=False
            )
            LOGGER.info("Saved ROC curve to %s", roc_path)
            roc_summary = {"auc": auc}

        generalization_summary = None
        if train_cfg.eval_larger_lattice and not data_cfg.anomaly_detection:
            gen_height = train_cfg.larger_height or data_cfg.height * 2
            gen_width = train_cfg.larger_width or data_cfg.width * 2
            gen_rng = np.random.default_rng(seed + 2024)
            if train_cfg.generalization_density is not None:
                densities = np.full(
                    (train_cfg.num_generalization_samples,),
                    float(train_cfg.generalization_density),
                    dtype=np.float32,
                )
            else:
                densities = _sample_density_schedule(train_cfg.num_generalization_samples, data_cfg, gen_rng)

            if data_cfg.stochastic:
                gen_inputs, gen_targets, _ = generate_stochastic_pairs(
                    num_samples=train_cfg.num_generalization_samples,
                    height=gen_height,
                    width=gen_width,
                    densities=densities,
                    p=data_cfg.p_stochastic,
                    rng=gen_rng,
                )
            else:
                gen_inputs, gen_targets, _ = generate_deterministic_pairs(
                    num_samples=train_cfg.num_generalization_samples,
                    height=gen_height,
                    width=gen_width,
                    densities=densities,
                    rng=gen_rng,
                )

            gen_loss, gen_acc, gen_logits = _evaluate_dataset(
                state, eval_step, gen_inputs, gen_targets, train_cfg.batch_size
            )
            gen_probs = jax.nn.sigmoid(gen_logits)
            gen_plot_path = output_root / "generalization_example.png"
            plot_grid_triplet(
                gen_inputs[0],
                gen_targets[0],
                np.array(gen_probs[0]),
                save_path=gen_plot_path,
                title=f"Generalisation on {gen_height}x{gen_width}",
            )

            generalization_summary = {
                "height": gen_height,
                "width": gen_width,
                "num_samples": train_cfg.num_generalization_samples,
                "loss": gen_loss,
                "accuracy": gen_acc,
                "density_mean": float(np.mean(densities)),
            }
            LOGGER.info(
                "Generalisation eval on %dx%d: loss=%.4f acc=%.4f",
                gen_height,
                gen_width,
                gen_loss,
                gen_acc,
            )

        metrics_path = output_root / "summary.json"
        summary_payload = {
            "test_loss": test_loss,
            "test_accuracy": test_acc,
            "test_brier_score": test_brier,
            "test_negative_log_likelihood": test_nll,
            "density_means": {
                "train": float(np.mean(splits.densities_train)) if splits.densities_train is not None else None,
                "val": float(np.mean(splits.densities_val)) if splits.densities_val is not None else None,
                "test": float(np.mean(splits.densities_test)) if splits.densities_test is not None else None,
            },
        }
        if roc_summary is not None:
            summary_payload.update(roc_summary)
        if generalization_summary is not None:
            summary_payload["generalization"] = generalization_summary
        metrics_path.write_text(json.dumps(summary_payload, indent=2))
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
