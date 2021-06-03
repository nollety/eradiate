"""
Particle layers
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Callable, Union

import attr
import numpy as np
import pint
import pinttr
import xarray as xr
from pinttr.util import units_compatible
from scipy.stats import expon, norm

from .. import path_resolver
from .._attrs import documented, parse_docs
from .._factory import BaseFactory
from ..contexts import SpectralContext
from ..units import to_quantity
from ..units import unit_context_config as ucc
from ..units import unit_registry as ureg
from ..validators import is_positive


@parse_docs
@attr.s
class VerticalDistribution(ABC):
    r"""
    An abstract base class for particles vertical distributions.

    Vertical distributions help define particle layers.

    The particle layer is split into a number of divisions (sub-layers),
    wherein the particles fraction is evaluated.

    The vertical distribution is normalised so that:

    .. math::
        \sum_i f_i = 1

    where :math:`f_i` is the particles fraction in the layer division :math:`i`.
    """
    bottom = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=pinttr.validators.has_compatible_units,
        ),
        doc="Layer bottom altitude.\nUnit-enabled field (default: ucc[length]).",
        type="float",
    )
    top = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=pinttr.validators.has_compatible_units,
        ),
        doc="Layer top altitude.\nUnit-enabled field (default: ucc[length]).",
        type="float",
    )

    def __attrs_post_init__(self):
        if self.bottom >= self.top:
            raise ValueError("bottom altitude must be lower than top altitude")

    @classmethod
    def from_dict(cls, d: dict) -> VerticalDistribution:
        """Initialise a :class:`VerticalDistribution` from a dictionary."""
        return cls(**d)

    @property
    @abstractmethod
    def fractions(self) -> Callable:
        """Returns a callable that evaluates the particles fractions in the
        layer, given an array of altitude values."""
        pass

    @staticmethod
    def _normalise(x: np.ndarray) -> np.ndarray:
        """
        Scale the values so that their sum is 1.

        Parameter ``x`` (array):
            Values to normalise.

        Returns → array:
            Normalised array.
        """
        _norm = np.sum(x)
        if _norm > 0.0:
            return x / _norm
        else:
            raise ValueError(f"Cannot normalise fractions because the norm is " f"0.")


class VerticalDistributionFactory(BaseFactory):
    """
    This factory constructs objects whose classes are derived from
    :class:`VerticalDistribution`

    .. admonition:: Registered factory members
       :class: hint

       .. factorytable::
          :factory: VerticalDistributionFactory
    """

    _constructed_type = VerticalDistribution
    registry = {}


@VerticalDistributionFactory.register("uniform")
@parse_docs
@attr.s
class Uniform(VerticalDistribution):
    r"""
    Uniform vertical distribution.

    The uniform probability distribution function is:

    .. math::
        f(z) = \frac{1}{z_{\rm top} - z_{\rm bottom}}, \quad
        z \in [z_{\rm top}, z_{\rm bottom}]

    where :math:`z_{\rm top}` and :math:`z_{\rm bottom}` are the layer top and bottom
    altitudes, respectively.
    """

    @property
    def fractions(self) -> Callable:
        def eval(z: pint.Quantity) -> np.ndarray:
            """
            Parameter ``z`` (:class:`pint.Quantity`):
                Altitude values.

            Return → array:
                Particles fractions.
            """
            if (self.bottom <= z).all() and (z <= self.top).all():
                x = z.magnitude
                return self._normalise(np.ones(len(x)))
            else:
                raise ValueError(
                    f"Altitude values do not lie between layer "
                    f"bottom ({self.bottom}) and top ({self.top}) "
                    f"altitudes. Got {z}."
                )

        return eval


@VerticalDistributionFactory.register("exponential")
@parse_docs
@attr.s
class Exponential(VerticalDistribution):
    r"""
    Exponential vertical distribution.

    The exponential probability distribution function is:

    .. math::
        f(z) = \lambda  \exp \left( -\lambda z \right)

    where :math:`\lambda` is the rate parameter and :math:`z` is the altitude.
    """
    rate = documented(
        pinttr.ib(
            units=ucc.deferred("collision_coefficient"),
            default=None,
            converter=attr.converters.optional(
                pinttr.converters.to_units(ucc.deferred("collision_coefficient"))
            ),
            validator=attr.validators.optional(pinttr.validators.has_compatible_units),
        ),
        doc="Rate parameter of the exponential distribution. If ``None``, "
        "set to the inverse of the layer thickness.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="``None``",
    )

    def __attrs_post_init__(self):
        if self.rate is None:
            self.rate = 1.0 / (self.top - self.bottom)

    @property
    def fractions(self) -> Callable:
        def eval(z: pint.Quantity) -> np.ndarray:
            if (self.bottom <= z).all() and (z <= self.top).all():
                x = z.magnitude
                loc = self.bottom.to(z.units).magnitude
                scale = (1.0 / self.rate).to(z.units).magnitude
                f = expon.pdf(x=x, loc=loc, scale=scale)
                return self._normalise(f)
            else:
                raise ValueError(
                    f"Altitude values do not lie between layer "
                    f"bottom ({self.bottom}) and top ({self.top}) "
                    f"altitudes. Got {z}."
                )

        return eval


@VerticalDistributionFactory.register("gaussian")
@parse_docs
@attr.s
class Gaussian(VerticalDistribution):
    r"""
    Gaussian vertical distribution.

    The Gaussian probability distribution function is:

    .. math::
        f(z) = \frac{1}{2 \pi \sigma}
        \exp{\left[
            -\frac{1}{2}
            \left( \frac{z - \mu}{\sigma} \right)^2
        \right]}

    where :math:`\mu` is the mean of the distribution and :math:`\sigma` is
    the standard deviation of the distribution.
    """
    mean = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            default=None,
            converter=attr.converters.optional(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=attr.validators.optional(pinttr.validators.has_compatible_units),
        ),
        doc="Mean (expectation) of the distribution. "
        "If ``None``, set to the middle of the layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="``None``",
    )
    std = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            default=None,
            converter=attr.converters.optional(
                pinttr.converters.to_units(ucc.deferred("length"))
            ),
            validator=attr.validators.optional(pinttr.validators.has_compatible_units),
        ),
        doc="Standard deviation of the distribution. If ``None``, set to one "
        "sixth of the layer thickness so that half the layer thickness "
        "equals three standard deviations.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="``None``",
    )

    def __attrs_post_init__(self):
        if self.mean is None:
            self.mean = (self.bottom + self.top) / 2.0
        if self.std is None:
            self.std = (self.top - self.bottom) / 6.0

    @property
    def fractions(self) -> Callable:
        def eval(z: pint.Quantity) -> np.ndarray:
            if (self.bottom <= z).all() and (z <= self.top).all():
                x = z.magnitude
                loc = self.mean.to(z.units).magnitude
                scale = self.std.to(z.units).magnitude
                f = norm.pdf(x=x, loc=loc, scale=scale)
                return self._normalise(f)
            else:
                raise ValueError(
                    f"Altitude values do not lie between layer "
                    f"bottom ({self.bottom}) and top ({self.top}) "
                    f"altitudes. Got {z}."
                )

        return eval


@VerticalDistributionFactory.register("array")
@parse_docs
@attr.s
class Array(VerticalDistribution):
    """
    Flexible vertical distribution specified either by an array of values,
    or by a :class:`~xarray.DataArray` object.
    """

    values = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(np.array),
            validator=attr.validators.optional(attr.validators.instance_of(np.ndarray)),
        ),
        doc="Particles fractions values on a regular altitude mesh starting "
        "from the layer bottom and stopping at the layer top altitude.",
        type="array",
        default="``None``",
    )
    data_array = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(xr.DataArray),
            validator=attr.validators.optional(
                attr.validators.instance_of(xr.DataArray)
            ),
        ),
        doc="Particles vertical distribution data array. Fraction as a function"
        " of altitude (``z``).",
        type=":class:`xarray.DataArray`",
        default="``None``",
    )

    @data_array.validator
    def check(self, attribute, value):
        if value is not None:
            if not "z" in value.coords:
                raise ValueError("Attribute 'data_array' must have a 'z' " "coordinate")
            else:
                try:
                    units = ureg.Unit(value.z.units)
                    if not units_compatible(units, ureg.Unit("m")):
                        raise ValueError(
                            f"Coordinate 'z' of attribute "
                            f"'data_array' must have units"
                            f"compatible with m^-1 (got {units})."
                        )
                except AttributeError:
                    raise ValueError(
                        "Coordinate 'z' of attribute 'data_array' " "must have units."
                    )

    method = documented(
        attr.ib(
            default="linear", converter=str, validator=attr.validators.instance_of(str)
        ),
        doc="Method to interpolate the data. This parameter is passed to "
        ":meth:`xarray.DataArray.interp`.",
        type="str",
        default='``"linear"``',
    )

    def __attrs_post_init__(self):
        if self.values is None and self.data_array is None:
            raise ValueError("You must specify 'values' or 'data_array'.")
        elif self.values is not None and self.data_array is not None:
            raise ValueError(
                "You cannot specify both 'values' and " "'data_array' simultaneously."
            )
        elif self.data_array is None:
            self.data_array = xr.DataArray(
                data=self.values,
                coords={
                    "z": (
                        "z",
                        np.linspace(
                            start=self.bottom.to("m").magnitude,
                            stop=self.top.to("m").magnitude,
                            num=len(self.values),
                        ),
                        {"units": "m"},
                    )
                },
                dims=["z"],
            )
        elif self.data_array is not None:
            min_z = to_quantity(self.data_array.z.min(keep_attrs=True))
            if min_z < self.bottom:
                raise ValueError(
                    f"Minimum altitude value in data_array "
                    f"({min_z}) is smaller than bottom altitude "
                    f"({self.bottom})."
                )

            max_z = to_quantity(self.data_array.z.max(keep_attrs=True))
            if max_z > self.top:
                raise ValueError(
                    f"Minimum altitude value in data_array "
                    f"({max_z}) is smaller than top altitude"
                    f"({self.top})."
                )

    @property
    def fractions(self) -> Callable:
        def eval(z: pint.Quantity) -> np.ndarray:
            x = z.to(self.data_array.z.units).magnitude
            f = self.data_array.interp(
                coords={"z": x}, method=self.method, kwargs=dict(fill_value=0.0)
            )
            return self._normalise(f.values)

        return eval


@parse_docs
@attr.s
class ParticleLayer:
    """
    1D particle layer.

    The particle layer has a vertical extension specified by a bottom altitude
    (``bottom``) and a top altitude (``top``).
    Inside the layer, the particles are distributed according to a vertical
    distribution (``vert_dist``).
    See :class:`.VerticalDistribution` for the available distribution types
    and corresponding parameters.
    The particle layer is itself divided into a number of (sub-)layers
    (``n_layers``) to allow the description of the particles number variations
    with altitude.
    The total number of particles in the layer is adjusted so that the
    particle layer's optical thickness at 550 nm meet a specified value
    (``tau_550``).
    The particles radiative properties are specified by a data set
    (``dataset``).
    """

    bottom = documented(
        pinttr.ib(
            default=ureg.Quantity(0.0, ureg.km),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
            units=ucc.deferred("length"),
        ),
        doc="Bottom altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length])",
        type="float",
        default="0 km",
    )

    top = documented(
        pinttr.ib(
            units=ucc.deferred("length"),
            default=ureg.Quantity(1.0, ureg.km),
            converter=pinttr.converters.to_units(ucc.deferred("length")),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Top altitude of the particle layer.\n"
        "\n"
        "Unit-enabled field (default: ucc[length]).",
        type="float",
        default="1 km.",
    )

    vert_dist = documented(
        attr.ib(
            default={"type": "uniform"},
            validator=attr.validators.instance_of((dict, VerticalDistribution)),
        ),
        doc="Particles vertical distribution.",
        type="dict or :class:`VerticalDistribution`",
        default=":class:`Uniform`",
    )

    tau_550 = documented(
        pinttr.ib(
            units=ucc.deferred("dimensionless"),
            default=ureg.Quantity(0.2, ureg.dimensionless),
            validator=[
                is_positive,
                pinttr.validators.has_compatible_units,
            ],
        ),
        doc="Extinction optical thickness at the wavelength of 550 nm.\n"
        "\n"
        "Unit-enabled field (default: ucc[dimensionless]).",
        type="float",
        default="0.2",
    )

    n_layers = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(int),
            validator=attr.validators.optional(attr.validators.instance_of(int)),
        ),
        doc="Number of layers inside the particle layer.\n"
        "If ``None``, ``n_layers`` is set to a different value based on the "
        "vertical distribution type (see table below).\n"
        "\n"
        ".. list-table::\n"
        "   :widths: 1 1\n"
        "   :header-rows: 1\n"
        "\n"
        "   * - Vertical distribution type\n"
        "     - Number of layers\n"
        "   * - :class:`Uniform`\n"
        "     - 1\n"
        "   * - :class:`Exponential`\n"
        "     - 8\n"
        "   * - :class:`Gaussian`\n"
        "     - 16\n"
        "   * - :class:`Array`\n"
        "     - 32\n"
        "\n",
        type="int",
        default="``None``",
    )

    dataset = documented(
        attr.ib(
            default=path_resolver.resolve("tests/radprops/rtmom_aeronet_desert.nc"),
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc="Particles radiative properties data set path.",
        type="str",
    )

    def __attrs_post_init__(self):
        # update the keys 'bottom' and 'top' in vertical distribution config
        if isinstance(self.vert_dist, dict):
            d = self.vert_dist
            d.update({"bottom": self.bottom, "top": self.top})
            self.vert_dist = VerticalDistributionFactory.convert(d)

        # determine layers number based on vertical distribution type
        if self.n_layers is None:
            if isinstance(self.vert_dist, Uniform):
                self.n_layers = 1
            elif isinstance(self.vert_dist, Exponential):
                self.n_layers = 8
            elif isinstance(self.vert_dist, Gaussian):
                self.n_layers = 16
            elif isinstance(self.vert_dist, Array):
                self.n_layers = 32

    @property
    def z_layer(self) -> pint.Quantity:
        """
        Compute the layer altitude mesh within the layer.

        The layer altitude mesh corresponds to a regular level altitude mesh
        from the layer's bottom altitude to the layer's top altitude with
        a number of points specified by ``n_layer``.

        Returns → :class:`~pint.Quantity`:
            Layer altitude mesh.
        """
        bottom = self.bottom.to("km").magnitude
        top = self.top.to("km").magnitude
        z_level = np.linspace(start=bottom, stop=top, num=self.n_layers + 1)
        z_layer = (z_level[:-1] + z_level[1:]) / 2.0
        return ureg.Quantity(z_layer, "km")

    @property
    def fractions(self) -> np.ndarray:
        """
        Compute the particles fractions in the layer.

        Returns → :class:`~numpy.ndarray`:
            Particles fractions.
        """
        return self.vert_dist.fractions(self.z_layer)

    def eval_phase(self, spectral_ctx: SpectralContext) -> xr.DataArray:
        """
        Return phase function.

        The phase function is represented by a :class:`~xarray.DataArray` with
        a :math:`\\mu` (``mu``) coordinate for the scattering angle cosine
        (:math:`\\mu \\in [-1, 1]`).

        Returns → :class:`xarray.DataArray`:
            Phase function.
        """
        ds = xr.open_dataset(self.dataset)
        return (
            ds.phase.sel(i=0)
            .sel(j=0)
            .interp(w=spectral_ctx.wavelength.magnitude, kwargs=dict(bounds_error=True))
        )

    def eval_albedo(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate albedo given a spectral context.

        Returns → :class:`pint.Quantity`:
            Particle layer albedo.
        """
        wavelength = spectral_ctx.wavelength.magnitude
        ds = xr.open_dataset(self.dataset)
        interpolated_albedo = ds.albedo.interp(w=wavelength)
        albedo = to_quantity(interpolated_albedo)
        albedo_array = albedo * np.ones(self.n_layers)
        return albedo_array

    def eval_sigma_t(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate extinction coefficient given a spectral context.

        Returns → :class:`pint.Quantity`:
            Particle layer extinction coefficient.
        """
        wavelength = spectral_ctx.wavelength.magnitude
        ds = xr.open_dataset(self.dataset)
        interpolated_sigma_t = ds.sigma_t.interp(w=wavelength)
        sigma_t = to_quantity(interpolated_sigma_t)
        sigma_t_array = sigma_t * self.fractions
        normalised_sigma_t_array = self._normalise_to_tau(
            ki=sigma_t_array.magnitude,
            dz=(self.top - self.bottom) / self.n_layers,
            tau=self.tau_550,
        )
        return normalised_sigma_t_array

    def eval_sigma_a(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        Returns → :class:`pint.Quantity`:
            Particle layer absorption coefficient.
        """
        return self.eval_sigma_t(spectral_ctx) - self.eval_sigma_a(spectral_ctx)

    def eval_sigma_s(self, spectral_ctx: SpectralContext) -> pint.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.

        Returns → :class:`pint.Quantity`:
            Particle layer scattering coefficient.
        """
        return self.eval_sigma_t(spectral_ctx) * self.eval_albedo(spectral_ctx)

    @classmethod
    def from_dict(cls, d: dict) -> ParticleLayer:
        """
        Initialise a :class:`ParticleLayer` from a dictionary."""
        return cls(**d)

    @classmethod
    def convert(cls, value: Union[ParticleLayer, dict]):
        """
        Object converter method.

        If ``value`` is a dictionary, this method forwards it to
        :meth:`from_dict`. Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return cls.from_dict(value)

        return value

    def to_dataset(self, spectral_ctx: SpectralContext) -> xr.Dataset:
        """
        Return a dataset that holds the radiative properties of the
        particle layer.

        Returns → :class:`xarray.Dataset`:
            Particle layer radiative properties dataset.
        """
        phase = self.eval_phase(spectral_ctx)
        sigma_t = self.eval_sigma_t(spectral_ctx)
        albedo = self.eval_albedo(spectral_ctx)
        z_layer = self.z_layer
        wavelength = spectral_ctx.wavelength
        return xr.Dataset(
            data_vars={
                "phase": (
                    ("w", "mu"),
                    np.atleast_2d(phase.values),
                    dict(
                        standard_name="scattering_phase_function",
                        long_name="scattering phase function",
                        units="",
                    ),
                ),
                "sigma_t": (
                    ("w", "z_layer"),
                    np.atleast_2d(sigma_t.magnitude),
                    dict(
                        standard_name="extinction_coefficient",
                        long_name="extinction coefficient",
                        units=sigma_t.units,
                    ),
                ),
                "albedo": (
                    ("w", "z_layer"),
                    np.atleast_2d(albedo.magnitude),
                    dict(
                        standard_name="albedo",
                        long_name="albedo",
                        units=albedo.units,
                    ),
                ),
            },
            coords={
                "z_layer": (
                    "z_layer",
                    z_layer.magnitude,
                    dict(
                        standard_name="layer_altitude",
                        long_name="layer altitude",
                        units=z_layer.units,
                    ),
                ),
                "w": (
                    "w",
                    [wavelength.magnitude],
                    dict(
                        standard_name="wavelength",
                        long_name="wavelength",
                        units=wavelength.units,
                    ),
                ),
                "mu": (
                    "mu",
                    phase.mu.values,
                    dict(
                        standard_name="scattering_angle_cosine",
                        long_name="scattering angle cosine",
                        units="",
                    ),
                ),
            },
        )

    @staticmethod
    @ureg.wraps(ret="km^-1", args=("", "km", ""), strict=False)
    def _normalise_to_tau(ki: np.ndarray, dz: np.ndarray, tau: float) -> np.ndarray:
        r"""
        Normalise extinction coefficient values :math:`k_i` so that:

        .. math::

            \sum_i k_i \Delta z = \tau_{550}

        where :math:`tau` is the particle layer optical thickness.

        Parameter ``ki`` (array):
            Dimensionless extinction coefficients values [].

        Parameter ``dz`` (array):
            Layer divisions thickness [km].

        Parameter ``tau`` (float):
            Layer optical thickness (dimensionless).

        Returns → array:
            Normalised extinction coefficients.
        """
        return ki * tau / (np.sum(ki) * dz)
