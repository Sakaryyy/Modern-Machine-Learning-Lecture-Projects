"""
Training loop for the Game of Life transformer in JAX + Flax.
"""

from typing import Dict, Tuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from Project_3_Conways_Game_Of_Life_Transformer.src.config.training_config import TrainingConfig, TrainState
from Project_3_Conways_Game_Of_Life_Transformer.src.model.gol_transformer import TransformerConfig, \
    GameOfLifeTransformer
from Project_3_Conways_Game_Of_Life_Transformer.src.training.loss import binary_cross_entropy_with_logits
from Project_3_Conways_Game_Of_Life_Transformer.src.training.metrics import accuracy_from_logits
from Project_3_Conways_Game_Of_Life_Transformer.src.visualization.plotting_utils import set_scientific_plot_style, \
    plot_training_curves


def conway_step_periodic(grid: np.ndarray) -> np.ndarray:
    """Compute one deterministic Conway Game of Life step with periodic BC.

    Parameters
    ----------
    grid : np.ndarray
        Binary array of shape (height, width) with entries in {0, 1}.

    Returns
    -------
    next_grid : np.ndarray
        Binary array of the same shape with the next state.
    """
    h, w = grid.shape
    # Use numpy here for simplicity, you can convert to jax.numpy if desired
    padded = np.pad(grid, 1, mode="wrap")

    # Sum over the 8 neighbors
    neighbor_sum = (
            padded[0:h, 0:w]
            + padded[0:h, 1:w + 1]
            + padded[0:h, 2:w + 2]
            + padded[1:h + 1, 0:w]
            + padded[1:h + 1, 2:w + 2]
            + padded[2:h + 2, 0:w]
            + padded[2:h + 2, 1:w + 1]
            + padded[2:h + 2, 2:w + 2]
    )

    # Apply Conway rules
    alive = grid == 1
    birth = (neighbor_sum == 3) & (~alive)
    survive = ((neighbor_sum == 2) | (neighbor_sum == 3)) & alive
    next_grid = np.where(birth | survive, 1, 0).astype(np.int32)
    return next_grid


def create_train_state(
        rng: jax.random.PRNGKey,
        model: GameOfLifeTransformer,
        input_shape: Tuple[int, int, int],
        config: TrainingConfig,
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

    return TrainState.create(
        apply_fn=model.apply,
        params=params,
        tx=tx,
    )


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

        metrics = {"loss": loss, "accuracy": acc}
        return new_state, metrics

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
        return {"loss": loss, "accuracy": acc}

    # Jit both steps for efficiency
    return jax.jit(train_step), jax.jit(eval_step)


def train_model() -> None:
    """Example end to end training routine for 16x16 Game of Life.

    This function shows how to put together the model, dataset, and
    training loop. Adapt it for your project setup, logging and
    evaluation.
    """
    # Configuration
    height = 16
    width = 16
    num_train = 10000
    num_val = 2000

    model_config = TransformerConfig(
        d_model=64,
        num_heads=4,
        num_layers=3,
        mlp_dim=128,
        dropout_rate=0.1,
        use_local_attention=True,
        window_radius=1,
    )
    train_config = TrainingConfig(
        learning_rate=1e-3,
        num_epochs=20,
        batch_size=64,
    )

    # Data generation
    np_rng = np.random.default_rng(seed=0)
    train_inputs, train_targets = generate_gol_dataset(
        num_samples=num_train,
        height=height,
        width=width,
        rng=np_rng,
    )
    val_inputs, val_targets = generate_gol_dataset(
        num_samples=num_val,
        height=height,
        width=width,
        rng=np_rng,
    )

    # Model and state
    model = GameOfLifeTransformer(config=model_config)
    key = jax.random.PRNGKey(42)
    state = create_train_state(
        rng=key,
        model=model,
        input_shape=(height, width),
        config=train_config,
    )
    train_step, eval_step = make_train_and_eval_step(model)

    history = {
        "train_loss": [],
        "train_accuracy": [],
        "val_loss": [],
        "val_accuracy": [],
    }

    # Training loop
    for epoch in range(train_config.num_epochs):
        # New RNG for this epoch
        key, subkey = jax.random.split(key)

        # Training
        train_batches = data_loader(
            train_inputs,
            train_targets,
            batch_size=train_config.batch_size,
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

        # Evaluation on full validation set in a few batches
        val_batches = data_loader(
            val_inputs,
            val_targets,
            batch_size=train_config.batch_size,
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

        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_acc)

        print(
            f"Epoch {epoch + 1:03d} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

    set_scientific_plot_style()
    plot_training_curves(
        history,
        title=f"Training curves {height}x{width}",
        save_path=f"training_curves_{height}x{width}.png",
    )
