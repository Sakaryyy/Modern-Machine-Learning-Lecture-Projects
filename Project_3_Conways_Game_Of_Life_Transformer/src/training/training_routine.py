"""Training and evaluation utilities for the Conway Transformer."""

import contextlib
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable, Dict, Iterable, Tuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import optax
import pandas as pd
from flax import serialization
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from Project_3_Conways_Game_Of_Life_Transformer.src.config.data_config import DataConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import ModelConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.training_config import TrainState, TrainingConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.data_functions.conway_rules import conway_step_periodic
from Project_3_Conways_Game_Of_Life_Transformer.src.data_functions.data_pipelines import (
    DatasetSplits,
    generate_deterministic_pairs,
    generate_stochastic_pairs,
    prepare_gol_dataset,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.model.gol_transformer import (
    GameOfLifeModel,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.training.loss import (
    binary_cross_entropy_with_logits,
    balanced_binary_cross_entropy_with_logits,
    log_likelihood_from_logits,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.training.metrics import (
    accuracy_from_logits,
    balanced_accuracy_from_logits,
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
from Project_3_Conways_Game_Of_Life_Transformer.src.utils.rule_analysis import (
    RuleCategory,
    RuleMetrics,
    summarise_rule_accuracy,
)
from Project_3_Conways_Game_Of_Life_Transformer.src.visualization.plotting_utils import (
    plot_autoregressive_rollout,
    plot_calibration_curve,
    plot_grid_difference,
    plot_confusion_overview,
    plot_grid_pair_examples,
    plot_grid_triplet,
    plot_grid_triplet_array,
    plot_multiple_roc_curves,
    plot_performance_by_neighbor_count,
    plot_rule_diagnostics,
    plot_rule_probability_distributions,
    plot_training_curves,
    set_scientific_plot_style,
)

LOGGER = get_logger(__name__)


def conway_step_periodic_jax(grid: jnp.ndarray) -> jnp.ndarray:
    """JAX implementation of the deterministic Conway update."""
    padded = jnp.pad(grid, ((0, 0), (1, 1), (1, 1)), mode="wrap")
    neighbor_sum = (
            padded[:, 0:-2, 0:-2]
            + padded[:, 0:-2, 1:-1]
            + padded[:, 0:-2, 2:]
            + padded[:, 1:-1, 0:-2]
            + padded[:, 1:-1, 2:]
            + padded[:, 2:, 0:-2]
            + padded[:, 2:, 1:-1]
            + padded[:, 2:, 2:]
    )
    alive = grid == 1
    born = (neighbor_sum == 3) & (~alive)
    survive = alive & ((neighbor_sum == 2) | (neighbor_sum == 3))
    return jnp.where(born | survive, 1, 0).astype(jnp.int32)


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


def _describe_rule_setup(data_cfg: DataConfig) -> str:
    """Return a human-readable description of the rule configuration."""

    if data_cfg.anomaly_detection:
        return (
            "Stochastic mix 23/3 vs 35/3 "
            f"(normal p={data_cfg.p_stochastic:.2f}, anomaly p={data_cfg.p_anomaly:.2f})"
        )
    if data_cfg.stochastic:
        return f"Stochastic mix 23/3 vs 35/3 (p={data_cfg.p_stochastic:.2f})"
    return "Deterministic Conway 23/3"


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
        model: GameOfLifeModel,
        input_shape: Tuple[int, int, int],
        config: TrainingConfig,
        steps_per_epoch: int,
) -> Tuple[TrainState, Callable[[int], float]]:
    """Initialize model parameters and optimizer state.

    Parameters
    ----------
    rng : jax.random.PRNGKey
        JAX random key for parameter initialization and dropout.
    model : GameOfLifeModel
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


def make_train_and_eval_step(
        model: GameOfLifeModel,
        train_cfg: TrainingConfig,
        data_cfg: DataConfig,
):
    """Create jitted train and eval step functions for the given model.

    This helper closes over `model.apply` to keep the signatures clean.

    Parameters
    ----------
    model : GameOfLifeModel
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

        rollout_steps = train_cfg.rollout_steps if not data_cfg.stochastic else 1

        def loss_fn(params):
            total_loss = 0.0
            acc_list = []
            bal_acc_list = []
            current = x
            current_true = x

            for step in range(rollout_steps):
                logits = state.apply_fn(
                    {"params": params},
                    current,
                    train=True,
                    rngs={"dropout": rng},
                )

                if data_cfg.stochastic:
                    targets = y
                else:
                    current_true = conway_step_periodic_jax(current_true)
                    targets = current_true

                if train_cfg.balance_loss:
                    loss = balanced_binary_cross_entropy_with_logits(
                        logits,
                        targets,
                        max_pos_weight=train_cfg.max_pos_weight,
                    )
                else:
                    loss = binary_cross_entropy_with_logits(logits, targets)

                total_loss = total_loss + loss
                acc_list.append(accuracy_from_logits(logits, targets))
                bal_acc_list.append(balanced_accuracy_from_logits(logits, targets))

                preds = (jax.nn.sigmoid(logits) >= 0.5).astype(jnp.int32)
                current = jax.lax.stop_gradient(preds)

            loss = total_loss / rollout_steps
            acc = jnp.stack(acc_list).mean()
            bal_acc = jnp.stack(bal_acc_list).mean()

            if train_cfg.l2_reg > 0.0:
                l2_term = sum(jnp.sum(jnp.square(p)) for p in jtu.tree_leaves(params))
                param_count = sum(p.size for p in jtu.tree_leaves(params))
                loss = loss + train_cfg.l2_reg * l2_term / jnp.maximum(param_count, 1)
            return loss, (acc, bal_acc)

        grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss, (acc, bal_acc)), grads = grad_fn(state.params)
        new_state = state.apply_gradients(grads=grads)
        return new_state, {"loss": loss, "accuracy": acc, "balanced_accuracy": bal_acc}

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

        rollout_steps = train_cfg.rollout_steps if not data_cfg.stochastic else 1
        total_loss = 0.0
        acc_list = []
        bal_acc_list = []
        current = x
        current_true = x
        logits = None
        first_logits = None

        for _ in range(rollout_steps):
            logits = state.apply_fn(
                {"params": state.params},
                current,
                train=False,
            )
            if first_logits is None:
                first_logits = logits

            if data_cfg.stochastic:
                targets = y
            else:
                current_true = conway_step_periodic_jax(current_true)
                targets = current_true

            if train_cfg.balance_loss:
                loss = balanced_binary_cross_entropy_with_logits(
                    logits,
                    targets,
                    max_pos_weight=train_cfg.max_pos_weight,
                )
            else:
                loss = binary_cross_entropy_with_logits(logits, targets)

            total_loss = total_loss + loss
            acc_list.append(accuracy_from_logits(logits, targets))
            bal_acc_list.append(balanced_accuracy_from_logits(logits, targets))

            preds = (jax.nn.sigmoid(logits) >= 0.5).astype(jnp.int32)
            current = jax.lax.stop_gradient(preds)

        loss = total_loss / rollout_steps
        acc = jnp.stack(acc_list).mean()
        bal_acc = jnp.stack(bal_acc_list).mean()
        return {
            "loss": loss,
            "accuracy": acc,
            "balanced_accuracy": bal_acc,
            "logits": first_logits,
        }

    return jax.jit(train_step), jax.jit(eval_step)


def _evaluate_dataset(
        state: TrainState,
        eval_step: Callable[[TrainState, Dict[str, np.ndarray]], Dict[str, jnp.ndarray]],
        inputs: np.ndarray,
        targets: np.ndarray,
        batch_size: int,
        desc: str | None = None,
) -> Tuple[float, float, float, np.ndarray]:
    """Run evaluation over a dataset and aggregate metrics with progress bars."""

    eval_batches = data_loader(
        inputs,
        targets,
        batch_size,
        rng=None,
        shuffle=False,
    )

    loss_list, acc_list, bal_acc_list = [], [], []
    logits_all = []
    iterator = tqdm(
        eval_batches,
        total=int(np.ceil(inputs.shape[0] / batch_size)),
        desc=desc,
        leave=False,
        dynamic_ncols=True,
    )
    for batch in iterator:
        metrics = eval_step(state, batch)
        loss_list.append(metrics["loss"])
        acc_list.append(metrics["accuracy"])
        bal_acc_list.append(metrics["balanced_accuracy"])
        logits_all.append(np.array(metrics["logits"]))

    loss_value = float(jnp.stack(loss_list).mean())
    acc_value = float(jnp.stack(acc_list).mean())
    bal_acc_value = float(jnp.stack(bal_acc_list).mean())
    logits_concat = np.concatenate(logits_all, axis=0)
    return loss_value, acc_value, bal_acc_value, logits_concat


def _save_checkpoint(output_dir: Path, state: TrainState, epoch: int) -> Path:
    ckpt_dir = output_dir / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"epoch_{epoch:03d}.msgpack"
    ckpt_path.write_bytes(serialization.to_bytes(state.params))
    LOGGER.info("Saved checkpoint to %s", ckpt_path)
    return ckpt_path


def _save_configs(output_dir: Path, data_cfg: DataConfig, model_cfg: ModelConfig,
                  train_cfg: TrainingConfig) -> None:
    (output_dir / "configs").mkdir(parents=True, exist_ok=True)
    for name, cfg in {"data": data_cfg, "model": model_cfg, "training": train_cfg}.items():
        cfg_path = output_dir / "configs" / f"{name}.json"
        cfg_path.write_text(json.dumps(asdict(cfg), indent=2, default=str))
        LOGGER.info("Saved %s configuration to %s", name, cfg_path)


def _load_config_from_artifacts(run_dir: Path, name: str, cls):
    cfg_path = run_dir / "configs" / f"{name}.json"
    if cfg_path.exists():
        LOGGER.info("Loading %s configuration from %s", name, cfg_path)
        return cls(**json.loads(cfg_path.read_text()))
    LOGGER.warning("No saved %s configuration found at %s; using defaults", name, cfg_path)
    return None


def load_run_artifact_configs(
        run_dir: Path,
) -> Tuple[DataConfig | None, ModelConfig | None, TrainingConfig | None]:
    """Load saved configs adjacent to a checkpoint directory when available."""

    data_cfg = _load_config_from_artifacts(run_dir, "data", DataConfig)
    model_cfg = _load_config_from_artifacts(run_dir, "model", ModelConfig)
    train_cfg = _load_config_from_artifacts(run_dir, "training", TrainingConfig)
    return data_cfg, model_cfg, train_cfg


def _save_rule_report(output_dir: Path, prefix: str, summary: Dict[RuleCategory, RuleMetrics]) -> Path:
    report = {
        rule.name.lower(): {
            "total": metrics.total,
            "correct": metrics.correct,
            "accuracy": metrics.accuracy,
        }
        for rule, metrics in summary.items()
    }
    report_path = output_dir / f"{prefix}_rule_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    LOGGER.info("Saved %s rule adherence report to %s", prefix, report_path)
    return report_path


def analyse_rule_adherence(
        inputs: np.ndarray,
        targets: np.ndarray,
        probs: np.ndarray,
        output_dir: Path,
        prefix: str,
) -> Dict[RuleCategory, RuleMetrics]:
    """Aggregate how well the model applies Conway's rules across a dataset."""

    LOGGER.info("Analysing Conway rule adherence for %s set...", prefix)
    predictions_binary = (probs > 0.5).astype(int)
    aggregate: Dict[RuleCategory, RuleMetrics] = {
        rule: RuleMetrics(total=0, correct=0) for rule in RuleCategory
    }

    iterator = tqdm(
        range(inputs.shape[0]),
        desc=f"{prefix.capitalize()} rule analysis",
        leave=False,
    )
    for idx in iterator:
        per_rule = summarise_rule_accuracy(inputs[idx], predictions_binary[idx])
        for rule, metrics in per_rule.items():
            aggregate[rule].add_counts(metrics.total, metrics.correct)

    _save_rule_report(output_dir, prefix, aggregate)

    diag_path = output_dir / f"{prefix}_rule_diagnostics.png"
    plot_rule_diagnostics(
        x=inputs[0],
        y_true=targets[0],
        y_prob=probs[0],
        title=f"{prefix.capitalize()} rule view",
        save_path=diag_path,
    )

    plot_rule_probability_distributions(
        inputs=inputs[:5000],  # Subsample for speed if needed
        probs=probs[:5000],
        title=f"{prefix.capitalize()} Rule Confidence",
        save_path=output_dir / f"{prefix}_rule_prob_dist.png"
    )

    plot_performance_by_neighbor_count(
        inputs=inputs,
        targets=targets,
        probs=probs,
        title=f"{prefix.capitalize()} Performance by Density",
        save_path=output_dir / f"{prefix}_neighbor_performance.png"
    )

    LOGGER.info("Saved %s rule diagnostic plot to %s", prefix, diag_path)
    return aggregate


def _record_history(output_dir: Path, history: Dict[str, list]) -> None:
    df = pd.DataFrame(history)
    csv_path = output_dir / "training_history.csv"
    xlsx_path = output_dir / "training_history.xlsx"
    df.to_csv(csv_path, index=False)
    df.to_excel(xlsx_path, index=False)
    LOGGER.info("Persisted training metrics to %s and %s", csv_path, xlsx_path)


def _choose_example_indices(num_available: int, num_requested: int, rng: np.random.Generator) -> np.ndarray:
    """Select a random subset of indices for visualisation."""

    if num_available == 0 or num_requested <= 0:
        return np.array([], dtype=int)

    num = int(min(num_available, num_requested))
    return rng.permutation(num_available)[:num]


def train_and_evaluate(
        data_cfg: DataConfig,
        model_cfg: ModelConfig,
        train_cfg: TrainingConfig,
        output_root: Path,
        seed: int = 0,
) -> Tuple[TrainState, DatasetSplits, Path]:
    """Full training pipeline including plots and artefact saving."""

    output_root.mkdir(parents=True, exist_ok=True)

    # Encourage size generalisation by preferring relative positional signals
    if train_cfg.eval_larger_lattice:
        if not model_cfg.use_relative_position_bias:
            LOGGER.warning(
                "Enabling relative position bias to support lattice-size generalisation."
            )
            model_cfg = replace(model_cfg, use_relative_position_bias=True)
        if model_cfg.use_coord_features:
            LOGGER.warning(
                "Disabling absolute coordinate features to avoid overfitting to a fixed grid size."
            )
            model_cfg = replace(model_cfg, use_coord_features=False)

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

        diagnostics_dir = output_root / "diagnostics"
        diagnostics_dir.mkdir(parents=True, exist_ok=True)
        examples_dir = output_root / "examples"
        examples_dir.mkdir(parents=True, exist_ok=True)

        example_rng = np.random.default_rng(seed + 7481)

        example_pairs_path = examples_dir / "train_pairs.png"
        train_pair_indices = _choose_example_indices(
            num_available=splits.x_train.shape[0],
            num_requested=min(6, splits.x_train.shape[0]),
            rng=example_rng,
        )
        plot_grid_pair_examples(
            inputs=splits.x_train[train_pair_indices],
            targets=splits.y_train[train_pair_indices],
            save_path=example_pairs_path,
            title="Example Conway transitions (train split)",
        )
        LOGGER.info("Saved example grid transitions to %s", example_pairs_path)

        model = GameOfLifeModel(config=model_cfg)
        key = jax.random.PRNGKey(seed)
        steps_per_epoch = max(1, int(np.ceil(splits.x_train.shape[0] / train_cfg.batch_size)))
        state, lr_schedule = create_train_state(
            key, model, (data_cfg.height, data_cfg.width), train_cfg, steps_per_epoch
        )
        if data_cfg.stochastic and train_cfg.rollout_steps > 1:
            LOGGER.warning(
                "Stochastic data detected: rollout_steps>1 is ignored because multi-step targets are undefined."
            )
        train_step, eval_step = make_train_and_eval_step(model, train_cfg, data_cfg)

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
            "train_balanced_accuracy": [],
            "val_loss": [],
            "val_accuracy": [],
            "val_balanced_accuracy": [],
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
            epoch_losses, epoch_accs, epoch_bal_accs = [], [], []

            with logging_redirect_tqdm():
                for batch in tqdm(
                        batches,
                        total=steps_per_epoch,
                        desc=f"Epoch {epoch}/{train_cfg.num_epochs} [train]",
                        leave=False,
                        dynamic_ncols=True,
                ):
                    epoch_key, step_key = jax.random.split(epoch_key)
                    state, metrics = train_step(state, batch, step_key)
                    epoch_losses.append(metrics["loss"])
                    epoch_accs.append(metrics["accuracy"])
                    epoch_bal_accs.append(metrics["balanced_accuracy"])

            train_loss = float(jnp.stack(epoch_losses).mean())
            train_acc = float(jnp.stack(epoch_accs).mean())
            train_bal_acc = float(jnp.stack(epoch_bal_accs).mean())

            val_loss, val_acc, val_bal_acc, _ = _evaluate_dataset(
                state, eval_step, splits.x_val, splits.y_val, train_cfg.batch_size, desc=f"Epoch {epoch} [val]"
            )
            lr_value = float(lr_schedule(int(state.step)))

            history["train_loss"].append(train_loss)
            history["train_accuracy"].append(train_acc)
            history["train_balanced_accuracy"].append(train_bal_acc)
            history["val_loss"].append(val_loss)
            history["val_accuracy"].append(val_acc)
            history["val_balanced_accuracy"].append(val_bal_acc)
            history["learning_rate"].append(lr_value)
            history["train_val_gap"].append(train_bal_acc - val_bal_acc)

            LOGGER.info(
                (
                    "Epoch %03d train_loss=%.4f train_bal_acc=%.4f val_loss=%.4f "
                    "val_bal_acc=%.4f lr=%.6f"
                ),
                epoch,
                train_loss,
                train_bal_acc,
                val_loss,
                val_bal_acc,
                lr_value,
            )

            ckpt_path = _save_checkpoint(output_root, state, epoch)
            if val_bal_acc > best_val_acc:
                best_val_acc = val_bal_acc
                best_val_loss = val_loss
                best_checkpoint = ckpt_path

        if best_checkpoint is not None:
            LOGGER.info(
                "Best validation balanced accuracy=%.4f loss=%.4f saved at %s",
                best_val_acc,
                best_val_loss,
                best_checkpoint,
            )

        training_plot_path = diagnostics_dir / "training_curves.png"
        plot_training_curves(history, title="Training diagnostics", save_path=training_plot_path)
        LOGGER.info("Saved training curves to %s", training_plot_path)
        _record_history(output_root, history)

        # Test evaluation
        test_loss, test_acc, test_bal_acc, logits_concat = _evaluate_dataset(
            state, eval_step, splits.x_test, splits.y_test, train_cfg.batch_size, desc="Testing"
        )
        LOGGER.info(
            "Test loss=%.4f accuracy=%.4f balanced_accuracy=%.4f",
            test_loss,
            test_acc,
            test_bal_acc,
        )

        probs = jax.nn.sigmoid(logits_concat)
        rule_label = _describe_rule_setup(data_cfg)
        bin_centers, empirical_freq, bin_counts = calibration_curve(probs, splits.y_test)
        calibration_path = diagnostics_dir / "calibration_curve.png"
        plot_calibration_curve(probs, splits.y_test, save_path=calibration_path)
        calib_df = pd.DataFrame(
            {"bin_center": bin_centers, "empirical_freq": empirical_freq, "bin_count": bin_counts}
        )
        calib_df.to_csv(diagnostics_dir / "calibration_curve.csv", index=False)
        LOGGER.info("Saved calibration diagnostics to %s", calibration_path)

        rule_summary = analyse_rule_adherence(
            inputs=splits.x_test,
            targets=splits.y_test,
            probs=np.asarray(probs),
            output_dir=output_root,
            prefix="test",
        )

        # Example prediction plots on the test set
        test_example_dir = examples_dir / "test"
        test_example_dir.mkdir(parents=True, exist_ok=True)
        num_examples = int(min(12, splits.x_test.shape[0]))
        for idx in range(num_examples):
            sample_probs = np.array(probs[idx])
            plot_grid_triplet(
                splits.x_test[idx],
                splits.y_test[idx],
                sample_probs,
                save_path=test_example_dir / f"test_prediction_{idx:03d}.png",
                title=f"Test prediction example {idx}",
            )
            plot_grid_difference(
                splits.y_test[idx],
                (sample_probs > 0.5).astype(int),
                save_path=test_example_dir / f"test_prediction_{idx:03d}_difference.png",
            )

        plot_grid_triplet_array(
            inputs=splits.x_test,
            targets=splits.y_test,
            probs=np.array(probs),
            title="Test prediction overview",
            rule_label=rule_label,
            threshold=0.5,
            max_examples=num_examples,
            include_difference=True,
            rng=example_rng,
            save_path=test_example_dir / "test_prediction_overview.png",
        )
        plot_confusion_overview(
            y_true_batch=splits.y_test,
            y_prob_batch=np.array(probs),
            rule_label=rule_label,
            threshold=0.5,
            save_path=test_example_dir / "test_confusion_overview.png",
            title="True/false positive and negative summary (test)",
        )

        def model_step_wrapper(x_batch):
            logits = state.apply_fn({"params": state.params}, x_batch, train=False)
            return np.array(jax.nn.sigmoid(logits))

        plot_autoregressive_rollout(
            initial_state=splits.x_test[0],
            model_step_fn=model_step_wrapper,
            true_step_fn=conway_step_periodic,
            steps=8,
            save_path=diagnostics_dir / "test_rollout.png",
            title="Autoregressive Stability (Test Set)"
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
            roc_path = diagnostics_dir / "roc_curve.png"
            roc_label = f"H{data_cfg.height}W{data_cfg.width} ({rule_label})"
            plot_multiple_roc_curves({roc_label: (fpr, tpr, auc)}, save_path=roc_path)
            pd.DataFrame({"threshold": thresholds, "fpr": fpr, "tpr": tpr}).to_csv(
                diagnostics_dir / "roc_curve.csv", index=False
            )
            LOGGER.info("Saved ROC curve to %s", roc_path)
            roc_summary = {"auc": auc}

        heldout_summary = None
        if not data_cfg.stochastic and not data_cfg.anomaly_detection:
            heldout_rng = np.random.default_rng(seed + 1337)
            heldout_densities = _sample_density_schedule(
                train_cfg.num_generalization_samples, data_cfg, heldout_rng
            )
            heldout_inputs, heldout_targets, _ = generate_deterministic_pairs(
                num_samples=train_cfg.num_generalization_samples,
                height=data_cfg.height,
                width=data_cfg.width,
                densities=heldout_densities,
                rng=heldout_rng,
            )
            heldout_loss, heldout_acc, heldout_bal_acc, heldout_logits = _evaluate_dataset(
                state,
                eval_step,
                heldout_inputs,
                heldout_targets,
                train_cfg.batch_size,
                desc="Held-out deterministic eval",
            )
            heldout_probs = jax.nn.sigmoid(heldout_logits)

            heldout_dir = examples_dir / "heldout"
            heldout_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(min(4, heldout_inputs.shape[0])):
                plot_grid_triplet(
                    heldout_inputs[idx],
                    heldout_targets[idx],
                    np.array(heldout_probs[idx]),
                    save_path=heldout_dir / f"heldout_prediction_{idx:03d}.png",
                    title=f"Held-out deterministic example {idx}",
                )

            num_examples = int(min(8, heldout_inputs.shape[0]))
            plot_grid_triplet_array(
                inputs=heldout_inputs,
                targets=heldout_targets,
                probs=np.array(heldout_probs),
                title="Held-out deterministic examples",
                rule_label=_describe_rule_setup(data_cfg),
                threshold=0.5,
                max_examples=num_examples,
                include_difference=True,
                rng=example_rng,
                save_path=heldout_dir / "heldout_prediction_overview.png",
            )

            heldout_summary = {
                "loss": heldout_loss,
                "accuracy": heldout_acc,
                "balanced_accuracy": heldout_bal_acc,
                "density_mean": float(np.mean(heldout_densities)),
            }
            LOGGER.info(
                "Held-out deterministic evaluation: loss=%.4f acc=%.4f",
                heldout_loss,
                heldout_acc,
            )

        generalization_summary = None
        if train_cfg.eval_larger_lattice:
            gen_height = train_cfg.larger_height or data_cfg.height * 2
            gen_width = train_cfg.larger_width or data_cfg.width * 2
            gen_rng = np.random.default_rng(seed + 61453)
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

            gen_loss, gen_acc, gen_bal_acc, gen_logits = _evaluate_dataset(
                state, eval_step, gen_inputs, gen_targets, train_cfg.batch_size
            )
            gen_probs = jax.nn.sigmoid(gen_logits)
            gen_example_dir = examples_dir / "generalization"
            gen_example_dir.mkdir(parents=True, exist_ok=True)
            for idx in range(min(3, gen_inputs.shape[0])):
                plot_grid_triplet(
                    gen_inputs[idx],
                    gen_targets[idx],
                    np.array(gen_probs[idx]),
                    save_path=gen_example_dir / f"generalization_{idx:03d}.png",
                    title=f"Generalisation on {gen_height}x{gen_width} (example {idx})",
                )
            num_examples = int(min(9, gen_inputs.shape[0]))
            plot_grid_triplet_array(
                inputs=gen_inputs,
                targets=gen_targets,
                probs=np.array(gen_probs),
                title=f"Generalisation on {gen_height}x{gen_width}",
                rule_label=_describe_rule_setup(data_cfg),
                threshold=0.5,
                max_examples=num_examples,
                include_difference=True,
                rng=example_rng,
                save_path=gen_example_dir / "generalization_overview.png",
            )

            plot_autoregressive_rollout(
                initial_state=gen_inputs[0],
                model_step_fn=model_step_wrapper,
                true_step_fn=conway_step_periodic,
                steps=8,
                save_path=diagnostics_dir / "generalization_rollout.png",
                title=f"Rollout Stability on {gen_height}x{gen_width}"
            )

            generalization_rules = analyse_rule_adherence(
                inputs=gen_inputs,
                targets=gen_targets,
                probs=np.asarray(gen_probs),
                output_dir=output_root,
                prefix="generalization",
            )

            generalization_summary = {
                "height": gen_height,
                "width": gen_width,
                "num_samples": train_cfg.num_generalization_samples,
                "loss": gen_loss,
                "accuracy": gen_acc,
                "balanced_accuracy": gen_bal_acc,
                "density_mean": float(np.mean(densities)),
                "rule_adherence": {
                    rule.name.lower(): {
                        "total": metrics.total,
                        "correct": metrics.correct,
                        "accuracy": metrics.accuracy,
                    }
                    for rule, metrics in generalization_rules.items()
                },
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
            "test_balanced_accuracy": test_bal_acc,
            "test_brier_score": test_brier,
            "test_negative_log_likelihood": test_nll,
            "density_means": {
                "train": float(np.mean(splits.densities_train)) if splits.densities_train is not None else None,
                "val": float(np.mean(splits.densities_val)) if splits.densities_val is not None else None,
                "test": float(np.mean(splits.densities_test)) if splits.densities_test is not None else None,
            },
            "rule_adherence": {
                rule.name.lower(): {
                    "total": metrics.total,
                    "correct": metrics.correct,
                    "accuracy": metrics.accuracy,
                }
                for rule, metrics in rule_summary.items()
            },
        }
        if roc_summary is not None:
            summary_payload.update(roc_summary)
        if heldout_summary is not None:
            summary_payload["heldout_deterministic"] = heldout_summary
        if generalization_summary is not None:
            summary_payload["generalization"] = generalization_summary
        metrics_path.write_text(json.dumps(summary_payload, indent=2))
        LOGGER.info("Wrote summary to %s", metrics_path)

    return state, splits, output_root


def load_params(checkpoint_path: Path, model: GameOfLifeModel, input_shape: Tuple[int, int]) -> TrainState:
    """Load a saved parameter file into a fresh TrainState."""

    rng = jax.random.PRNGKey(0)
    dummy = jnp.zeros((1,) + input_shape, dtype=jnp.int32)
    variables = model.init(rng, dummy, train=False)
    params = serialization.from_bytes(variables["params"], checkpoint_path.read_bytes())
    return TrainState.create(apply_fn=model.apply, params=params, tx=optax.adam(0.0))


def generate_predictions(
        checkpoint_path: Path,
        model_cfg: ModelConfig | None,
        inputs: np.ndarray,
        batch_size: int,
) -> np.ndarray:
    """Run a trained model in generation/evaluation mode."""

    run_dir = checkpoint_path.parent.parent
    saved_cfg = _load_config_from_artifacts(run_dir, "model", ModelConfig)
    active_cfg = saved_cfg or model_cfg or ModelConfig()
    LOGGER.info("Using model configuration: %s", active_cfg)

    model = GameOfLifeModel(config=active_cfg)
    state = load_params(checkpoint_path, model, (inputs.shape[1], inputs.shape[2]))

    def forward(params, x_batch):
        logits = state.apply_fn({"params": params}, x_batch, train=False)
        return jax.nn.sigmoid(logits)

    forward_jit = jax.jit(forward)

    num_samples = inputs.shape[0]
    outputs = np.empty_like(inputs, dtype=np.float32)
    for start in tqdm(
            range(0, num_samples, batch_size), desc="Generating", leave=False, dynamic_ncols=True
    ):
        end = min(start + batch_size, num_samples)
        x_batch = jnp.array(inputs[start:end])
        probs = forward_jit(state.params, x_batch)
        outputs[start:end] = np.array(probs)

    return outputs
