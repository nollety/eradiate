import numpy as np
from numpy.typing import ArrayLike

from ._util import ensure_array


def square_to_uniform_disk_concentric(sample: ArrayLike) -> ArrayLike:
    """
    Low-distortion concentric square to disk mapping.

    Parameter ``sample`` (array-like):
        A (N, 2) array of sample values. The function tries to be flexible with
        arrays with
    """
    # Matches Mitsuba implementation

    sample = ensure_array(sample)
    if sample.ndim < 2:
        sample = sample.reshape((sample.size // 2, 2))
    if sample.ndim > 2 or sample.shape[1] != 2:
        raise ValueError(f"array must be of shape (N, 2), got {sample.shape}")

    x: ArrayLike = 2.0 * sample[..., 0] - 1.0
    y: ArrayLike = 2.0 * sample[..., 1] - 1.0

    is_zero = np.logical_and(x == 0.0, y == 0.0)
    quadrant_1_or_3 = np.abs(x) < np.abs(y)

    r = np.where(quadrant_1_or_3, y, x)
    rp = np.where(quadrant_1_or_3, x, y)

    phi = np.empty_like(r)
    phi[~is_zero] = 0.25 * np.pi * (rp[~is_zero] / r[~is_zero])
    phi[quadrant_1_or_3] = 0.5 * np.pi - phi[quadrant_1_or_3]
    phi[is_zero] = 0.0

    s, c = np.sin(phi), np.cos(phi)
    return np.vstack((r * c, r * s)).T


def square_to_uniform_hemisphere(sample: ArrayLike) -> ArrayLike:
    """
    Uniformly sample a vector on the unit hemisphere with respect to solid angles.
    """
    # Matches Mitsuba implementation

    sample = ensure_array(sample)
    if sample.ndim < 2:
        sample = sample.reshape((sample.size // 2, 2))
    if sample.ndim > 2 or sample.shape[1] != 2:
        raise ValueError(f"array must be of shape (N, 2), got {sample.shape}")

    p = square_to_uniform_disk_concentric(sample)
    z = 1.0 - np.multiply(p, p).sum(axis=1)
    p *= np.sqrt(z + 1.0).reshape((len(p), 1))
    return np.vstack((p[..., 0], p[..., 1], z)).T


def uniform_hemisphere_to_square(v: ArrayLike) -> ArrayLike:
    """
    Inverse of the mapping square_to_uniform_hemisphere.
    """
    # Matches Mitsuba implementation

    v = ensure_array(v)
    if v.ndim < 2:
        v = v.reshape((v.size // 3, 3))
    if v.ndim > 2 or v.shape[1] != 3:
        raise ValueError(f"array must be of shape (N, 3), got {v.shape}")

    p = v[..., 0:2]
    return uniform_disk_to_square_concentric(
        p / np.sqrt(v[..., 2] + 1.0).reshape((len(p), 1))
    )


def uniform_disk_to_square_concentric(p: ArrayLike) -> ArrayLike:
    """
    Inverse of the mapping square_to_uniform_disk_concentric.
    """
    # Matches Mitsuba implementation

    p = ensure_array(p)
    if p.ndim < 2:
        p = p.reshape((p.size // 2, 2))
    if p.ndim > 2 or p.shape[1] != 2:
        raise ValueError(f"array must be of shape (N, 2), got {p.shape}")

    quadrant_0_or_2 = np.abs(p[..., 0]) > np.abs(p[..., 1])
    r_sign = np.where(quadrant_0_or_2, p[..., 0], p[..., 1])
    r = np.copysign(np.linalg.norm(p, axis=-1), r_sign)

    phi = np.arctan2(p[..., 1] * np.sign(r_sign), p[..., 0] * np.sign(r_sign))

    t = 4.0 / np.pi * phi
    t = np.where(quadrant_0_or_2, t, 2.0 - t) * r

    a = np.where(quadrant_0_or_2, r, t)
    b = np.where(quadrant_0_or_2, t, r)

    return np.vstack(((a + 1.0) * 0.5, (b + 1.0) * 0.5)).T
