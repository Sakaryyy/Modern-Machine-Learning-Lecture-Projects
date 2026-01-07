from dataclasses import dataclass
from typing import Optional

from flax.training import train_state


@dataclass
class TrainingConfig:
    """Hyperparameter container for training.

    Attributes
    ----------
    learning_rate : float
        Peak learning rate for the optimizer.
    num_epochs : int
        Number of epochs to train.
    batch_size : int
        Batch size used for training.
    optimizer : str
        Optimizer choice. One of {"adam", "adamw", "sgd"}.
    weight_decay : float
        Weight decay (L2) used by decoupled optimizers.
    beta1 : float
        Beta1 momentum for Adam based optimizers.
    beta2 : float
        Beta2 momentum for Adam based optimizers.
    eps : float
        Numerical stability epsilon for Adam based optimizers.
    lr_schedule : str
        Learning rate schedule. One of {"constant", "cosine", "linear"}.
    warmup_steps : int
        Number of warmup steps before reaching peak learning rate.
    decay_steps : int or None
        Total steps used for the decay schedule. If None defaults to the
        total number of training steps.
    min_lr_ratio : float
        Final learning rate expressed as a fraction of the peak value.
    max_grad_norm : float or None
        Clip gradients by global norm when provided.
    l2_reg : float
        Additional L2 regularisation strength applied to all parameters
        in the loss function.
    eval_larger_lattice : bool
        Whether to automatically evaluate on a larger lattice size
        after training to probe generalisation.
    larger_height : int or None
        Optional explicit height for the larger lattice evaluation. If
        None, a scaled height based on the training data is used.
    larger_width : int or None
        Optional explicit width for the larger lattice evaluation. If
        None, a scaled width based on the training data is used.
    num_generalization_samples : int
        Number of samples to generate for the larger lattice evaluation.
    generalization_density : float or None
        Optional fixed density for the larger lattice evaluation. If
        None the density sampling strategy from the training data is
        reused.
    """

    learning_rate: float = 5e-4
    num_epochs: int = 100
    batch_size: int = 64
    optimizer: str = "adamw"
    weight_decay: float = 1e-4
    beta1: float = 0.9
    beta2: float = 0.999
    eps: float = 1e-8
    lr_schedule: str = "cosine"
    warmup_steps: int = 2000
    decay_steps: Optional[int] = None
    min_lr_ratio: float = 0.05
    max_grad_norm: Optional[float] = 1.0
    l2_reg: float = 0.0
    eval_larger_lattice: bool = True
    larger_height: Optional[int] = None
    larger_width: Optional[int] = None
    num_generalization_samples: int = 1000
    generalization_density: Optional[float] = None


class TrainState(train_state.TrainState):
    """Extended train state that can also hold extra variables if needed."""
    pass
