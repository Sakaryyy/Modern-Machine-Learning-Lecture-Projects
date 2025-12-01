from dataclasses import dataclass

from flax.training import train_state


@dataclass
class TrainingConfig:
    """Hyperparameter container for training.

    Attributes
    ----------
    learning_rate : float
        Learning rate for the optimizer.
    num_epochs : int
        Number of epochs to train.
    batch_size : int
        Batch size used for training.
    """

    learning_rate: float = 1e-3
    num_epochs: int = 20
    batch_size: int = 64


class TrainState(train_state.TrainState):
    """Extended train state that can also hold extra variables if needed."""
    pass
