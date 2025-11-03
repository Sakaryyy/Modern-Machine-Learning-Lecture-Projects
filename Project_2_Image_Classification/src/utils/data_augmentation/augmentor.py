"""Data augmentation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import jax
import jax.numpy as jnp
import numpy as np
import numpy.typing as npt

from Project_2_Image_Classification.src.utils.logging import get_logger

__all__ = [
    "DataAugmentationConfig",
    "ImageAugmenter",
]


@dataclass(slots=True)
class DataAugmentationConfig:
    """Configuration describing how images are augmented before training.

    Parameters
    ----------
    enabled:
        If ``True`` the augmentation pipeline is executed.  When ``False`` the
        images are returned unchanged.
    horizontal_flip_prob:
        Probability of flipping an image along the horizontal axis.
    vertical_flip_prob:
        Probability of flipping an image along the vertical axis.  Vertical
        flips are disabled by default as they can be detrimental for CIFAR-10.
    random_crop_padding:
        Number of pixels used for reflection padding before extracting a crop.
    rotation_degrees:
        Maximum absolute rotation angle in degrees.  The actual angle is drawn
        uniformly from ``[-rotation_degrees, rotation_degrees]`` whenever a
        rotation is applied.
    rotation_probability:
        Probability that a sampled image will undergo a random rotation.
    brightness_delta:
        Maximum additive brightness change applied to each image.  A value of
        ``0.2`` implies that the offset is sampled from ``[-0.2, 0.2]``.
    contrast_delta:
        Range controlling multiplicative contrast adjustments.  The resulting
        factor is drawn from ``[1 - contrast_delta, 1 + contrast_delta]``.
    saturation_delta:
        Range controlling multiplicative saturation adjustments.  A value of
        ``0.3`` allows factors in ``[0.7, 1.3]``.
    color_jitter_probability:
        Probability of applying brightness/contrast/saturation jitter to a
        sample.
    gaussian_noise_std:
        Standard deviation of Gaussian noise added to selected samples.
    gaussian_noise_probability:
        Probability that Gaussian noise is injected into a sample.
    cutout_probability:
        Probability of applying the CutOut regularisation technique.
    cutout_size:
        Edge length of the square CutOut mask measured in pixels.
    cutout_fill:
        Optional fill value used inside the CutOut mask.  When ``None`` the
        image mean is used which preserves overall brightness.
    use_random_crop / use_horizontal_flip / ...:
        Boolean switches enabling or disabling the corresponding augmentation
        independent of probability values.  This allows fine grained control
        via configuration files or command line flags.
    """

    enabled: bool = True
    horizontal_flip_prob: float = 0.5
    vertical_flip_prob: float = 0.0
    random_crop_padding: int = 4
    rotation_degrees: float = 15.0
    rotation_probability: float = 0.25
    brightness_delta: float = 0.15
    contrast_delta: float = 0.2
    saturation_delta: float = 0.2
    color_jitter_probability: float = 0.8
    gaussian_noise_std: float = 0.01
    gaussian_noise_probability: float = 0.3
    cutout_probability: float = 0.25
    cutout_size: int = 8
    cutout_fill: float | None = None
    use_random_crop: bool = True
    use_horizontal_flip: bool = True
    use_vertical_flip: bool = False
    use_rotation: bool = True
    use_color_jitter: bool = True
    use_gaussian_noise: bool = True
    use_cutout: bool = True

    def __post_init__(self) -> None:
        for name, value in {
            "horizontal_flip_prob": self.horizontal_flip_prob,
            "vertical_flip_prob": self.vertical_flip_prob,
            "rotation_probability": self.rotation_probability,
            "color_jitter_probability": self.color_jitter_probability,
            "gaussian_noise_probability": self.gaussian_noise_probability,
            "cutout_probability": self.cutout_probability,
        }.items():
            if not 0.0 <= value <= 1.0:
                msg = f"'{name}' must lie within [0, 1]."
                raise ValueError(msg)
        if self.random_crop_padding < 0:
            raise ValueError("'random_crop_padding' must be non-negative.")
        if self.rotation_degrees < 0:
            raise ValueError("'rotation_degrees' must be non-negative.")
        if self.brightness_delta < 0:
            raise ValueError("'brightness_delta' must be non-negative.")
        if self.contrast_delta < 0:
            raise ValueError("'contrast_delta' must be non-negative.")
        if self.saturation_delta < 0:
            raise ValueError("'saturation_delta' must be non-negative.")
        if self.gaussian_noise_std < 0:
            raise ValueError("'gaussian_noise_std' must be non-negative.")
        if self.cutout_size <= 0:
            raise ValueError("'cutout_size' must be a positive integer.")
        for flag_name in (
                "use_random_crop",
                "use_horizontal_flip",
                "use_vertical_flip",
                "use_rotation",
                "use_color_jitter",
                "use_gaussian_noise",
                "use_cutout",
        ):
            if not isinstance(getattr(self, flag_name), bool):
                raise TypeError(f"'{flag_name}' must be a boolean flag.")

class ImageAugmenter:
    """Apply stochastic image augmentation using differentiable ``jax`` ops."""

    _LUMA_WEIGHTS: npt.NDArray[np.float32] = np.array([0.2989, 0.5870, 0.1140], dtype=np.float32)

    def __init__(self, config: DataAugmentationConfig | Mapping[str, object]) -> None:
        if isinstance(config, Mapping):
            config = DataAugmentationConfig(**config)
        self._config = config
        self._logger = get_logger(self.__class__.__name__)
        self._has_logged_runtime_shape = False
        self._log_pipeline_summary()

    def __call__(self, rng: jax.Array, images: jnp.ndarray) -> jnp.ndarray:
        """Return augmented ``images`` sampled according to ``rng``."""

        if not self._config.enabled:
            return images

        key = rng
        augmented = images

        if self._config.use_random_crop and self._config.random_crop_padding > 0:
            key, subkey = jax.random.split(key)
            augmented = self._random_crop(subkey, augmented)

        if self._config.use_horizontal_flip and self._config.horizontal_flip_prob > 0:
            key, subkey = jax.random.split(key)
            augmented = self._random_flip(subkey, augmented, axis=1, probability=self._config.horizontal_flip_prob)

        if self._config.vertical_flip_prob > 0:
            key, subkey = jax.random.split(key)
            augmented = self._random_flip(subkey, augmented, axis=0, probability=self._config.vertical_flip_prob)

        if (
                self._config.use_rotation
                and self._config.rotation_probability > 0
                and self._config.rotation_degrees > 0
        ):
            key, subkey = jax.random.split(key)
            augmented = self._random_rotate(subkey, augmented)

        if self._config.use_color_jitter and self._config.color_jitter_probability > 0:
            key, subkey = jax.random.split(key)
            augmented = self._color_jitter(subkey, augmented)

        if (
                self._config.use_gaussian_noise
                and self._config.gaussian_noise_probability > 0
                and self._config.gaussian_noise_std > 0
        ):
            key, subkey = jax.random.split(key)
            augmented = self._gaussian_noise(subkey, augmented)

        if self._config.use_cutout and self._config.cutout_probability > 0:
            key, subkey = jax.random.split(key)
            augmented = self._cutout(subkey, augmented)

        augmented = jnp.clip(augmented, 0.0, 1.0)
        if not self._has_logged_runtime_shape:
            self._logger.info(
                "Augmentation processed batch with shape %s -> %s (sample count preserved).",
                tuple(images.shape),
                tuple(augmented.shape),
            )
            self._has_logged_runtime_shape = True
        return augmented

    # ------------------------------------------------------------------
    # Individual augmentation primitives
    # ------------------------------------------------------------------
    def _random_flip(self, rng: jax.Array, images: jnp.ndarray, *, axis: int, probability: float) -> jnp.ndarray:
        decisions = jax.random.bernoulli(rng, probability, (images.shape[0],))

        def flip(image: jnp.ndarray, decision: jnp.ndarray) -> jnp.ndarray:
            return jax.lax.cond(decision, lambda x: jnp.flip(x, axis=axis), lambda x: x, image)

        return jax.vmap(flip)(images, decisions)

    def _random_crop(self, rng: jax.Array, images: jnp.ndarray) -> jnp.ndarray:
        padding = self._config.random_crop_padding
        if padding <= 0:
            return images

        padded = jnp.pad(
            images,
            ((0, 0), (padding, padding), (padding, padding), (0, 0)),
            mode="reflect",
        )
        batch_size, height, width, channels = images.shape
        max_offset = 2 * padding
        key_y, key_x = jax.random.split(rng)
        offsets_y = jax.random.randint(key_y, (batch_size,), 0, max_offset + 1)
        offsets_x = jax.random.randint(key_x, (batch_size,), 0, max_offset + 1)

        def crop(image: jnp.ndarray, offset_y: jnp.ndarray, offset_x: jnp.ndarray) -> jnp.ndarray:
            return jax.lax.dynamic_slice(image, (offset_y, offset_x, 0), (height, width, channels))

        return jax.vmap(crop)(padded, offsets_y, offsets_x)

    def _random_rotate(self, rng: jax.Array, images: jnp.ndarray) -> jnp.ndarray:
        batch_size = images.shape[0]
        key_mask, key_angles = jax.random.split(rng)

        mask = jax.random.bernoulli(
            key_mask,
            self._config.rotation_probability,
            (batch_size,),
        )
        angles = jax.random.uniform(
            key_angles,
            (batch_size,),
            minval=-self._config.rotation_degrees,
            maxval=self._config.rotation_degrees,
        )

        def rotate_one(img, ang, apply_):
            return jax.lax.cond(
                apply_,
                lambda args: _rotate_image(args[0], args[1]),
                lambda args: args[0],
                (img, ang),
            )

        return jax.vmap(rotate_one)(images, angles, mask)

    def _color_jitter(self, rng: jax.Array, images: jnp.ndarray) -> jnp.ndarray:
        batch_size = images.shape[0]
        key_mask, key_brightness, key_contrast, key_saturation = jax.random.split(rng, 4)
        mask = jax.random.bernoulli(key_mask, self._config.color_jitter_probability, (batch_size,))
        brightness_offsets = jax.random.uniform(
            key_brightness,
            (batch_size,),
            minval=-self._config.brightness_delta,
            maxval=self._config.brightness_delta,
        )
        contrast_factors = jax.random.uniform(
            key_contrast,
            (batch_size,),
            minval=1.0 - self._config.contrast_delta,
            maxval=1.0 + self._config.contrast_delta,
        )
        saturation_factors = jax.random.uniform(
            key_saturation,
            (batch_size,),
            minval=1.0 - self._config.saturation_delta,
            maxval=1.0 + self._config.saturation_delta,
        )

        luma = jnp.asarray(self._LUMA_WEIGHTS, dtype=images.dtype)

        def jitter(
                image: jnp.ndarray,
                brightness: jnp.ndarray,
                contrast: jnp.ndarray,
                saturation: jnp.ndarray,
                apply: jnp.ndarray,
        ) -> jnp.ndarray:
            def _apply(args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
                img, b_off, c_fac, s_fac = args
                result = img + b_off.reshape(1, 1, 1)
                mean = jnp.mean(result, axis=(0, 1), keepdims=True)
                result = (result - mean) * c_fac + mean
                gray = jnp.tensordot(result, luma, axes=([-1], [0]))
                gray = gray[..., None]
                result = gray + (result - gray) * s_fac
                return result

            return jax.lax.cond(apply, _apply, lambda args: args[0], (image, brightness, contrast, saturation))

        augmented = jax.vmap(jitter)(images, brightness_offsets, contrast_factors, saturation_factors, mask)
        return augmented

    def _log_pipeline_summary(self) -> None:
        """Log a human readable summary of the augmentation pipeline."""

        if not self._config.enabled:
            self._logger.info("Data augmentation disabled; batches pass through unchanged.")
            return

        steps: list[str] = []
        if self._config.use_random_crop and self._config.random_crop_padding > 0:
            steps.append(f"Random crop (padding={self._config.random_crop_padding})")
        if self._config.use_horizontal_flip and self._config.horizontal_flip_prob > 0:
            steps.append(f"Horizontal flip (p={self._config.horizontal_flip_prob:.2f})")
        if self._config.use_vertical_flip and self._config.vertical_flip_prob > 0:
            steps.append(f"Vertical flip (p={self._config.vertical_flip_prob:.2f})")
        if (
                self._config.use_rotation
                and self._config.rotation_probability > 0
                and self._config.rotation_degrees > 0
        ):
            steps.append(
                f"Rotation (±{self._config.rotation_degrees:.1f}° with p={self._config.rotation_probability:.2f})"
            )
        if self._config.use_color_jitter and self._config.color_jitter_probability > 0:
            steps.append(
                (
                    "Color jitter (Δbrightness≤"
                    f"{self._config.brightness_delta:.2f}, Δcontrast≤{self._config.contrast_delta:.2f}, "
                    f"Δsaturation≤{self._config.saturation_delta:.2f}, p={self._config.color_jitter_probability:.2f})"
                )
            )
        if (
                self._config.use_gaussian_noise
                and self._config.gaussian_noise_probability > 0
                and self._config.gaussian_noise_std > 0
        ):
            steps.append(
                f"Gaussian noise (std={self._config.gaussian_noise_std:.3f}, p={self._config.gaussian_noise_probability:.2f})"
            )
        if self._config.use_cutout and self._config.cutout_probability > 0:
            steps.append(
                f"CutOut (size={self._config.cutout_size}, p={self._config.cutout_probability:.2f})"
            )

        if not steps:
            self._logger.info(
                "Augmentation enabled but no stochastic operations are active; batches remain unchanged."
            )
            return

        self._logger.info(
            "Augmentation pipeline initialised with %d step(s).", len(steps)
        )
        for step_description in steps:
            self._logger.info(" • %s", step_description)
        self._logger.info(
            "Augmentation preserves the number of samples per batch; logging actual shapes on first use."
        )

    def _gaussian_noise(self, rng: jax.Array, images: jnp.ndarray) -> jnp.ndarray:
        key_mask, key_noise = jax.random.split(rng)
        mask = jax.random.bernoulli(key_mask, self._config.gaussian_noise_probability, (images.shape[0],))
        noise = jax.random.normal(key_noise, images.shape) * self._config.gaussian_noise_std
        mask = mask.astype(images.dtype).reshape((-1, 1, 1, 1))
        return images + noise * mask

    def _cutout(self, rng: jax.Array, images: jnp.ndarray) -> jnp.ndarray:
        batch_size, height, width, _ = images.shape
        half = self._config.cutout_size // 2
        key_mask, key_y, key_x = jax.random.split(rng, 3)
        mask = jax.random.bernoulli(key_mask, self._config.cutout_probability, (batch_size,))
        centers_y = jax.random.randint(key_y, (batch_size,), 0, height)
        centers_x = jax.random.randint(key_x, (batch_size,), 0, width)
        fill_value = self._config.cutout_fill

        def apply_cutout(image: jnp.ndarray, centre_y: jnp.ndarray, centre_x: jnp.ndarray,
                         apply: jnp.ndarray) -> jnp.ndarray:
            def _cut(args: tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]) -> jnp.ndarray:
                img, y_c, x_c = args
                y1 = jnp.clip(y_c - half, 0, height)
                y2 = jnp.clip(y_c + half, 0, height)
                x1 = jnp.clip(x_c - half, 0, width)
                x2 = jnp.clip(x_c + half, 0, width)

                ys = jnp.arange(height)[:, None]
                xs = jnp.arange(width)[None, :]
                region = (ys >= y1) & (ys < y2) & (xs >= x1) & (xs < x2)
                region = region[..., None]

                if fill_value is None:
                    value = jnp.mean(img)
                else:
                    value = jnp.asarray(fill_value, dtype=img.dtype)

                return jnp.where(region, value, img)

            return jax.lax.cond(apply, _cut, lambda args: args[0], (image, centre_y, centre_x))

        return jax.vmap(apply_cutout)(images, centers_y, centers_x, mask)


def _bilinear_sample(image: jnp.ndarray, x: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Bilinear sampler for a single image.

    Parameters
    ----------
    image : (H, W) or (H, W, C)
    x, y  : (H, W) source coordinates in image space

    Returns
    -------
    sampled : same shape as `image`
    """
    h, w = image.shape[:2]

    x0 = jnp.floor(x).astype(jnp.int32)
    x1 = x0 + 1
    y0 = jnp.floor(y).astype(jnp.int32)
    y1 = y0 + 1

    x0 = jnp.clip(x0, 0, w - 1)
    x1 = jnp.clip(x1, 0, w - 1)
    y0 = jnp.clip(y0, 0, h - 1)
    y1 = jnp.clip(y1, 0, h - 1)

    # weights
    wx = x - x0.astype(x.dtype)
    wy = y - y0.astype(y.dtype)

    def _gather(ix, iy):
        return image[iy, ix]  # will be (H, W, C) or (H, W)

    Ia = _gather(x0, y0)
    Ib = _gather(x0, y1)
    Ic = _gather(x1, y0)
    Id = _gather(x1, y1)

    wa = (1.0 - wx) * (1.0 - wy)
    wb = (1.0 - wx) * wy
    wc = wx * (1.0 - wy)
    wd = wx * wy

    # broadcast over channels automatically
    return (wa[..., None] if image.ndim == 3 else wa) * Ia + \
        (wb[..., None] if image.ndim == 3 else wb) * Ib + \
        (wc[..., None] if image.ndim == 3 else wc) * Ic + \
        (wd[..., None] if image.ndim == 3 else wd) * Id


def _rotate_image(image: jnp.ndarray, angle_deg: jnp.ndarray) -> jnp.ndarray:
    """
    Rotate a single image around its center by `angle_deg` degrees.

    image: (H, W, C) or (H, W)
    """
    h, w = image.shape[:2]
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0

    theta = jnp.deg2rad(angle_deg)
    cos_t = jnp.cos(theta)
    sin_t = jnp.sin(theta)

    # target grid
    ys, xs = jnp.meshgrid(jnp.arange(h), jnp.arange(w), indexing="ij")
    xs = xs.astype(jnp.float32) - cx
    ys = ys.astype(jnp.float32) - cy

    x_src = cos_t * xs + sin_t * ys + cx
    y_src = -sin_t * xs + cos_t * ys + cy

    return _bilinear_sample(image, x_src, y_src)
