import jax

from Project_3_Conways_Game_Of_Life_Transformer.src.utils.logging import get_logger


def log_jax_runtime_info() -> jax.Device:
    """
    Log JAX backend and device information and return the preferred device.

    Returns
    -------
    jax.Device
        The device we will use for training. Preference order is:
        first GPU if available, otherwise the first CPU device.
    """
    logger = get_logger("DEVICE")
    backend = jax.default_backend()
    gpus = jax.devices("gpu")
    cpus = jax.devices("cpu")

    if gpus:
        dev = gpus[0]
        chosen = f"GPU: {dev.device_kind}"
    else:
        dev = cpus[0]
        chosen = f"CPU: {dev.device_kind}"

    logger.info(f"JAX default backend: {backend}")
    logger.info(f"JAX devices (gpu={len(gpus)}, cpu={len(cpus)}). Using {chosen}.")
    return dev
