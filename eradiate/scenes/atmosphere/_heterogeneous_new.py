import struct
import tempfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import MutableMapping

import attr
import numpy as np
import xarray as xr

from eradiate.contexts import KernelDictContext, SpectralContext

from ._core import Atmosphere
from ..phase import PhaseFunction
from ... import validators
from ..._attrs import documented, parse_docs
from ...units import unit_context_kernel as uck
from ...units import unit_registry as ureg


def write_binary_grid3d(filename, values):
    """
    Write volume data to a binary file so that a ``gridvolume`` kernel plugin can be
    instantiated with that file.

    Parameter ``filename`` (path-like):
        File name.

    Parameter ``values`` (:class:`~numpy.ndarray` or :class:`~xarray.DataArray`):
        Data array to output to the volume data file. This array must 3 or 4
        dimensions (x, y, z, spectrum). If the array is 3-dimensional, it will
        automatically be assumed to have only one spectral channel.
    """
    if isinstance(values, xr.DataArray):
        values = values.values

    if not isinstance(values, np.ndarray):
        raise TypeError(
            f"unsupported data type {type(values)} "
            f"(expected numpy array or xarray DataArray)"
        )

    if values.ndim not in {3, 4}:
        raise ValueError(
            f"'values' must have 3 or 4 dimensions " f"(got shape {values.shape})"
        )

    # note: this is an exact copy of the function write_binary_grid3d from
    # https://github.com/mitsuba-renderer/mitsuba-data/blob/master/tests/scenes/participating_media/create_volume_data.py

    with open(filename, "wb") as f:
        f.write(b"V")
        f.write(b"O")
        f.write(b"L")
        f.write(np.uint8(3).tobytes())  # Version
        f.write(np.int32(1).tobytes())  # type
        f.write(np.int32(values.shape[0]).tobytes())  # size
        f.write(np.int32(values.shape[1]).tobytes())
        f.write(np.int32(values.shape[2]).tobytes())
        if values.ndim == 3:
            f.write(np.int32(1).tobytes())  # channels
        else:
            f.write(np.int32(values.shape[3]).tobytes())  # channels
        f.write(np.float32(0.0).tobytes())  # bbox
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(0.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(np.float32(1.0).tobytes())
        f.write(values.ravel().astype(np.float32).tobytes())


def read_binary_grid3d(filename):
    """Reads a volume data binary file.

    Parameter ``filename`` (str):
        File name.

    Returns → :class:`~numpy.ndarray`:
        Values.
    """

    with open(filename, "rb") as f:
        file_content = f.read()
        _shape = struct.unpack("iii", file_content[8:20])  # shape of the values array
        _num = np.prod(np.array(_shape))  # number of values
        values = np.array(struct.unpack("f" * _num, file_content[48:]))
        # file_type = struct.unpack("ccc", file_content[:3]),
        # version = struct.unpack("B", file_content[3:4]),
        # type = struct.unpack("i", file_content[4:8]),
        # channels = struct.unpack("i", file_content[20:24]),
        # bbox = struct.unpack("ffffff", file_content[24:48]),

    return values


@parse_docs
@attr.s
class HeterogeneousAtmosphereNew(Atmosphere, ABC):
    """
    Heterogeneous atmosphere scene element [:factorykey:`heterogeneous`].

    This class builds a one-dimensional heterogeneous atmosphere.
    It expands as a ``heterogeneous`` kernel plugin, which takes as parameters
    a set of paths to volume data files.
    """

    albedo_fname = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(Path),
            validator=attr.validators.optional(validators.is_file),
        ),
        doc="Path to the single scattering albedo volume data file. If "
        "``None``, a value will be created when the file will be "
        "requested.",
        type="path-like or None",
        default="None",
    )

    sigma_t_fname = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(Path),
            validator=attr.validators.optional(validators.is_file),
        ),
        doc="Path to the extinction coefficient volume data file. If ``None``, "
        "a value will be created when the file will be requested.",
        type="path-like or None",
        default="None",
    )

    @albedo_fname.validator
    @sigma_t_fname.validator
    def _albedo_fname_and_sigma_t_fname_validator(instance, attribute, value):
        if (
            instance.width == "auto"
            and instance.albedo_fname is not None
            and instance.sigma_t_fname is not None
        ):
            raise ValueError(
                "'albedo_fname' and 'sigma_t_fname' cannot be set when 'width' is set to 'auto'"
            )
        if instance.toa_altitude == "auto" and value is not None:
            raise ValueError(
                "'albedo_fname' and 'sigma_t_fname' cannot be set when toa_altitude is set to 'auto'"
            )

    cache_dir = documented(
        attr.ib(default=None, converter=attr.converters.optional(Path)),
        doc="Path to a cache directory where volume data files will be "
        "created. If ``None``, a temporary cache directory will be used.",
        type="path-like or None",
        default="None",
    )

    def __attrs_post_init__(self):
        # Prepare cache directory in case we'd need it
        if self.cache_dir is None:
            self.cache_dir = Path(tempfile.mkdtemp())
        else:
            self.cache_dir.mkdir(parents=True, exist_ok=True)

    @abstractmethod
    def get_levels(self) -> ureg.Quantity:
        """Return the altitude levels."""
        pass

    @abstractmethod
    def get_phase(self, spectral_ctx: SpectralContext) -> PhaseFunction:
        """Return the phase function."""
        pass

    @abstractmethod
    def get_sigma_a(self, spectral_ctx: SpectralContext) -> ureg.Quantity:
        """Return the absorption coefficient."""
        pass

    @abstractmethod
    def get_sigma_s(self, spectral_ctx: SpectralContext) -> ureg.Quantity:
        """Return the scattering coefficient."""
        pass

    def get_albedo(self, spectral_ctx: SpectralContext) -> ureg.Quantity:
        """Return the albedo."""
        with np.errstate(divide="ignore", invalid="ignore"):
            sigma_s = self.get_sigma_s(spectral_ctx)
            sigma_t = self.get_sigma_t(spectral_ctx)
            return np.where(sigma_t != 0.0, sigma_s / sigma_t, 0.0)  # broadcast 0.0

    def get_sigma_t(self, spectral_ctx: SpectralContext) -> ureg.Quantity:
        """Return the extinction coefficient."""
        return self.get_sigma_a(spectral_ctx) + self.get_sigma_s(spectral_ctx)

    def kernel_width(self, ctx: KernelDictContext = None) -> ureg.Quantity:
        if self.width == "auto":
            auto_computed = 10.0 / self.get_sigma_s(ctx.spectral_ctx).min()
            auto_prescribed = ureg.Quantity(1e3, "km")
            return min(auto_computed, auto_prescribed)
        else:
            return self.width

    def kernel_phase(self, ctx: KernelDictContext = None) -> MutableMapping:
        return self.get_phase(ctx.spectral_ctx).kernel_dict(ctx=ctx)

    def kernel_media(self, ctx: KernelDictContext = None) -> MutableMapping:
        from mitsuba.core import ScalarTransform4f

        k_width = self.kernel_width(ctx).m_as(uck.get("length"))
        k_height = self.kernel_height(ctx).m_as(uck.get("length"))
        k_offset = self.kernel_offset(ctx).m_as(uck.get("length"))

        # First, transform the [0, 1]^3 cube to the right dimensions
        trafo = ScalarTransform4f(
            [
                [k_width, 0.0, 0.0, -0.5 * k_width],
                [0.0, k_width, 0.0, -0.5 * k_width],
                [0.0, 0.0, k_height + k_offset, -k_offset],
                [0.0, 0.0, 0.0, 1.0],
            ]
        )

        # Create volume data files
        self.make_volume_data(spectral_ctx=ctx.spectral_ctx)

        # if isinstance(self.phase(ctx.spectral_ctx), BlendPhaseFunction):
        #    make_phase_volume_data(spectral_ctx=ctx.spectral_ctx)

        return {
            f"medium_{self.id}": {
                "type": "heterogeneous",
                f"phase_{self.id}": self.kernel_phase(ctx)[
                    self.get_phase(ctx.spectral_ctx).id
                ],
                "sigma_t": {
                    "type": "gridvolume",
                    "filename": str(self.sigma_t_fname),
                    "to_world": trafo,
                },
                "albedo": {
                    "type": "gridvolume",
                    "filename": str(self.albedo_fname),
                    "to_world": trafo,
                },
            }
        }

    def kernel_shapes(self, ctx=None) -> MutableMapping:
        from mitsuba.core import ScalarTransform4f

        if ctx.ref:
            medium = {"type": "ref", "id": f"medium_{self.id}"}
        else:
            medium = self.kernel_media(ctx=None)[f"medium_{self.id}"]

        k_length = uck.get("length")
        k_width = self.kernel_width(ctx).m_as(k_length)
        k_height = self.kernel_height(ctx).m_as(k_length)
        k_offset = self.kernel_offset(ctx).m_as(k_length)

        return {
            f"shape_{self.id}": {
                "type": "cube",
                "to_world": ScalarTransform4f(
                    [
                        [0.5 * k_width, 0.0, 0.0, 0.0],
                        [0.0, 0.5 * k_width, 0.0, 0.0],
                        [0.0, 0.0, 0.5 * k_height, 0.5 * k_height - k_offset],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ),
                "bsdf": {"type": "null"},
                "interior": medium,
            }
        }

    def make_volume_data(self, spectral_ctx=None):
        """
        Create volume data files for requested fields.

        Parameter ``fields`` (str or list[str] or None):
            If str, field for which to create volume data file. If list,
            fields for which to create volume data files. If ``None``,
            all supported fields are processed (``{"albedo", "sigma_t"}``).
            Default: ``None``.

        Parameter ``spectral_ctx`` (:class:`.SpectralContext`):
            A spectral context data structure containing relevant spectral
            parameters (*e.g.* wavelength in monochromatic mode).
        """
        quantities = {"albedo": "albedo", "sigma_t": "collision_coefficient"}
        for field in quantities:
            field_quantity = getattr(self, f"get_{field}")(spectral_ctx)
            if field_quantity.ndim == 1:
                field_quantity = field_quantity[np.newaxis, np.newaxis, :]

            field_fname = getattr(self, f"{field}_fname")
            if field_fname is None:
                field_fname = self.cache_dir / f"{field}.vol"
                setattr(self, f"{field}_fname", field_fname)

            write_binary_grid3d(
                field_fname,
                field_quantity.m_as(uck.get(quantities[field])),
            )
