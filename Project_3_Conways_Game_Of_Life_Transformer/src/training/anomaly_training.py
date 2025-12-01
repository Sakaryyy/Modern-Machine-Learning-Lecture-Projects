from typing import Tuple, Dict

import jax
import jax.numpy as jnp
import numpy as np

from Project_3_Conways_Game_Of_Life_Transformer.src.config.anomaly_config import AnomalyExperimentConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.model_config import TransformerConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.config.training_config import TrainingConfig
from Project_3_Conways_Game_Of_Life_Transformer.src.model.gol_transformer import GameOfLifeTransformer
from Project_3_Conways_Game_Of_Life_Transformer.src.training.loss import compute_log_likelihoods_dataset
from Project_3_Conways_Game_Of_Life_Transformer.src.training.metrics import negative_log_likelihood_scores, \
    compute_roc_curve
from Project_3_Conways_Game_Of_Life_Transformer.src.training.training_routine import make_train_and_eval_step, \
    create_train_state
from Project_3_Conways_Game_Of_Life_Transformer.src.visualization.plotting_utils import plot_multiple_roc_curves


def run_anomaly_experiment_for_size(
        model_config: TransformerConfig,
        exp_config: AnomalyExperimentConfig,
        seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run training and anomaly detection for a single lattice size.

    This function trains the GameOfLifeTransformer on stochastic
    dynamics with bias parameter p_train, then evaluates anomaly
    detection performance on a test set that mixes normal samples
    (p_normal) and anomalous samples (p_anomaly). It returns the ROC
    curve and the raw scores and labels.

    Parameters
    ----------
    model_config : TransformerConfig
        Configuration of the transformer model.
    exp_config : AnomalyExperimentConfig
        Experiment configuration including lattice size, data sizes, and
        stochastic parameters.
    seed : int, optional
        Seed for JAX and NumPy random number generation.

    Returns
    -------
    fpr : np.ndarray
        False positive rates for the ROC curve.
    tpr : np.ndarray
        True positive rates for the ROC curve.
    scores : np.ndarray
        Anomaly scores for all test samples (negative log likelihood).
    labels : np.ndarray
        Binary labels for all test samples (0 normal, 1 anomaly).
    """
    height = exp_config.height
    width = exp_config.width

    # Numpy RNG for data
    np_rng = np.random.default_rng(seed=seed)

    # Training and validation data for p_train
    train_inputs, train_targets = generate_stochastic_dataset(
        num_samples=exp_config.num_train,
        height=height,
        width=width,
        p=exp_config.p_train,
        rng=np_rng,
    )
    val_inputs, val_targets = generate_stochastic_dataset(
        num_samples=exp_config.num_val,
        height=height,
        width=width,
        p=exp_config.p_train,
        rng=np_rng,
    )

    # Model and train state
    model = GameOfLifeTransformer(config=model_config)
    train_cfg = TrainingConfig(
        learning_rate=exp_config.learning_rate,
        num_epochs=exp_config.num_epochs,
        batch_size=exp_config.batch_size,
    )

    key = jax.random.PRNGKey(seed)
    state = create_train_state(
        rng=key,
        model=model,
        input_shape=(height, width),
        config=train_cfg,
    )
    train_step, eval_step = make_train_and_eval_step(model)

    # Training loop
    for epoch in range(train_cfg.num_epochs):
        # Training epoch
        train_batches = data_loader(
            train_inputs,
            train_targets,
            batch_size=train_cfg.batch_size,
            rng=np_rng,
        )
        epoch_loss = []
        epoch_acc = []

        for batch in train_batches:
            key, step_key = jax.random.split(key)
            state, metrics = train_step(state, batch, step_key)
            epoch_loss.append(metrics["loss"])
            epoch_acc.append(metrics["accuracy"])

        train_loss = float(jnp.stack(epoch_loss).mean())
        train_acc = float(jnp.stack(epoch_acc).mean())

        # Validation epoch
        val_batches = data_loader(
            val_inputs,
            val_targets,
            batch_size=train_cfg.batch_size,
            rng=np_rng,
        )
        val_loss_list = []
        val_acc_list = []

        for batch in val_batches:
            metrics = eval_step(state, batch)
            val_loss_list.append(metrics["loss"])
            val_acc_list.append(metrics["accuracy"])

        val_loss = float(jnp.stack(val_loss_list).mean())
        val_acc = float(jnp.stack(val_acc_list).mean())

        print(
            f"[{height}x{width}] Epoch {epoch + 1:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    # Test data for anomaly detection (90 percent normal, 10 percent anomalous)
    test_inputs, test_targets, test_labels = generate_anomaly_dataset(
        num_samples=exp_config.num_test,
        height=height,
        width=width,
        p_normal=exp_config.p_normal,
        p_anomaly=exp_config.p_anomaly,
        anomaly_fraction=exp_config.anomaly_fraction,
        rng=np_rng,
    )

    # Log likelihoods and anomaly scores
    log_likelihoods = compute_log_likelihoods_dataset(
        state=state,
        inputs=test_inputs,
        targets=test_targets,
        batch_size=exp_config.batch_size,
    )
    scores = negative_log_likelihood_scores(log_likelihoods)

    # ROC curve
    _, fpr, tpr = compute_roc_curve(scores=scores, labels=test_labels)

    return fpr, tpr, scores, test_labels


def run_all_anomaly_experiments() -> None:
    """Run anomaly experiments for multiple lattice sizes and plot ROC curves.

    This function is an example driver that executes the anomaly
    detection pipeline for lattice sizes 16x16, 32x32, and 64x64, and
    visualizes the tradeoff between true positive rate and false
    positive rate in ROC curves.
    """
    # Shared model configuration; you can tune this further
    model_config = TransformerConfig(
        d_model=64,
        num_heads=4,
        num_layers=3,
        mlp_dim=128,
        dropout_rate=0.1,
        use_local_attention=True,
        window_radius=1,
    )

    sizes = [(16, 16), (32, 32), (64, 64)]
    roc_results: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    for height, width in sizes:
        exp_config = AnomalyExperimentConfig(
            height=height,
            width=width,
            num_train=20000,
            num_val=4000,
            num_test=5000,
            p_train=0.8,
            p_normal=0.8,
            p_anomaly=0.6,
            anomaly_fraction=0.1,
            batch_size=64 if height <= 32 else 16,
            num_epochs=20,
            learning_rate=1e-3,
        )

        fpr, tpr, _, _ = run_anomaly_experiment_for_size(
            model_config=model_config,
            exp_config=exp_config,
            seed=0,
        )
        label = f"{height}x{width}"
        roc_results[label] = (fpr, tpr)

    plot_multiple_roc_curves(
        roc_dict=roc_results,
        title="ROC curves for anomaly detection at p=0.8 vs p=0.6",
        save_path="roc_curves_gol_stochastic.png",
    )
