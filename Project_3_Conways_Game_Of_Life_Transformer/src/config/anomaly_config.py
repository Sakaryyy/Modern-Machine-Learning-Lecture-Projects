from dataclasses import dataclass


@dataclass
class AnomalyExperimentConfig:
    """Configuration for an anomaly detection experiment.

    Attributes
    ----------
    height : int
        Lattice height.
    width : int
        Lattice width.
    num_train : int
        Number of training samples for stochastic dynamics.
    num_val : int
        Number of validation samples for monitoring training.
    num_test : int
        Number of test samples for anomaly detection evaluation.
    p_train : float
        Bias parameter p used for training, for example 0.8.
    p_normal : float
        Bias parameter p for normal test data, usually equal to
        ``p_train``.
    p_anomaly : float
        Bias parameter p for anomalous test data, for example 0.6.
    anomaly_fraction : float
        Fraction of test samples that are anomalous, for example 0.1.
    batch_size : int
        Batch size for training and evaluation.
    num_epochs : int
        Number of epochs for training.
    learning_rate : float
        Learning rate for Adam.
    """

    height: int
    width: int
    num_train: int = 100000
    num_val: int = 4000
    num_test: int = 5000
    p_train: float = 0.8
    p_normal: float = 0.8
    p_anomaly: float = 0.6
    anomaly_fraction: float = 0.1
    batch_size: int = 64
    num_epochs: int = 100
    learning_rate: float = 4e-5
