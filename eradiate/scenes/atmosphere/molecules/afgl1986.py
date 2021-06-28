import attr
import numpy as np
import pinttr
import xarray as xr

from . import MolecularAtmosphere
from .._core import AtmosphereFactory
from ...phase import RayleighPhaseFunction
from .... import data
from ...._attrs import documented, parse_docs
from ....contexts import SpectralContext
from ....data.absorption_spectra import Absorber, Engine, find_dataset
from ....radprops.absorption import compute_sigma_a
from ....radprops.rad_profile import make_dataset
from ....radprops.rayleigh import compute_sigma_s_air
from ....thermoprops import afgl1986
from ....thermoprops.util import (
    compute_scaling_factors,
    interpolate,
    rescale_concentration,
)
from ....units import to_quantity
from ....units import unit_context_config as ucc
from ....units import unit_registry as ureg

_AFGL1986_MODELS = [
    "tropical",
    "midlatitude_summer",
    "midlatitude_winter",
    "subarctic_summer",
    "subarctic_winter",
    "us_standard",
]


@AtmosphereFactory.register("afgl1986")
@parse_docs
@attr.s
class AFGL1986MolecularAtmosphere(MolecularAtmosphere):
    """
    Molecular atmosphere based on the AFGL (1986) atmospheric thermophysical
    properties profiles :cite:`Anderson1986AtmosphericConstituentProfiles`.

    :cite:`Anderson1986AtmosphericConstituentProfiles` defines six models,
    listed in the table below.

    .. list-table:: AFGL (1986) atmospheric thermophysical properties profiles models
       :widths: 2 4 4
       :header-rows: 1

       * - Model number
         - Model identifier
         - Model name
       * - 1
         - ``tropical``
         - Tropic (15N Annual Average)
       * - 2
         - ``midlatitude_summer``
         - Mid-Latitude Summer (45N July)
       * - 3
         - ``midlatitude_winter``
         - Mid-Latitude Winter (45N Jan)
       * - 4
         - ``subarctic_summer``
         - Sub-Arctic Summer (60N July)
       * - 5
         - ``subarctic_winter``
         - Sub-Arctic Winter (60N Jan)
       * - 6
         - ``us_standard``
         - U.S. Standard (1976)

    .. attention::
        The original altitude mesh specified by
        :cite:`Anderson1986AtmosphericConstituentProfiles` is a piece-wise
        regular altitude mesh with an altitude step of 1 km from 0 to 25 km,
        2.5 km from 25 km to 50 km and 5 km from 50 km to 120 km.
        Since the Eradiate kernel only supports regular altitude mesh, the
        original atmospheric thermophysical properties profiles were
        interpolated on the regular altitude mesh with an altitude step of 1 km
        from 0 to 120 km.

    Although the altitude meshes of the interpolated
    :cite:`Anderson1986AtmosphericConstituentProfiles` profiles is fixed,
    this class lets you define a custom altitude mesh (regular or irregular).

    .. admonition:: Example
        :class: example

        .. code:: python

            import numpy as np
            from eradiate import unit_registry as ureg

            AFGL1986MolecularAtmosphere(
                levels=np.array([0., 5., 10., 25., 50., 100]) * ureg.km
            )

        In this example, the :cite:`Anderson1986AtmosphericConstituentProfiles`
        profile is truncated at the height of 100 km.

    All six models include the following six absorbing molecular species:
    H2O, CO2, O3, N2O, CO, CH4 and O2.
    The concentrations of these species in the atmosphere is fixed by
    :cite:`Anderson1986AtmosphericConstituentProfiles`.
    However, this class allows you to rescale the concentrations of each
    individual molecular species to custom concentration values.
    Custom concentrations can be provided in different units.

    .. admonition:: Example
        :class: example

        .. code:: python

            from eradiate import unit_registry as ureg

            AFGL1986MolecularAtmosphere(
                concentrations={
                    "H2O": ureg.Quantity(15 , "kg/m^2"),
                    "CO2": 420 * ureg.dimensionless,
                    "O3": 350 * ureg.dobson_unit,
                }
            )
    """

    model = documented(
        attr.ib(
            default="us_standard",
            converter=str,
            validator=attr.validators.instance_of(str),
        ),
        doc=(
            "AFGL (1986) atmospheric thermophysical properties profile model "
            "identifier in [``'tropical'``, ``'midlatitude_summer'``, "
            "``'midlatitude_winter'``, ``'subarctic_summer'``, "
            "``'subarctic_winter'``, ``'us_standard'``.]"
        ),
        type="str",
    )

    @model.validator
    def _validate_model(self, attribute, value):
        if value not in _AFGL1986_MODELS:
            raise ValueError(
                f"{attribute} should be in {_AFGL1986_MODELS} " f"(got {value})."
            )

    levels = documented(
        pinttr.ib(
            factory=lambda: np.arange(0.0, 120.01, 1.0) * ureg.km,
            units=ucc.deferred("length"),
        ),
        doc="Level altitudes. Default is a regular mesh from 0 to 120 km with "
        "1 km layer size.\n"
        "\n"
        "Unit-enabled field (ucc[length]).",
        type="array",
        default="range(0, 121) km",
    )

    concentrations = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(dict),
            validator=attr.validators.optional(attr.validators.instance_of(dict)),
        ),
        doc="Mapping of species and concentration. "
        "For more information about rescaling process and the supported "
        "concentration units, refer to the documentation of "
        ":func:`~eradiate.thermoprops.util.compute_scaling_factors`.",
        type="dict",
    )

    has_absorption = documented(
        attr.ib(
            default=True,
            converter=bool,
            validator=attr.validators.instance_of(bool),
        ),
        doc="Absorption switch. If ``True``, the absorption coefficient is "
        "computed. Else, the absorption coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    has_scattering = documented(
        attr.ib(
            default=True,
            converter=bool,
            validator=attr.validators.instance_of(bool),
        ),
        doc="Scattering switch. If ``True``, the scattering coefficient is "
        "computed. Else, the scattering coefficient is not computed and "
        "instead set to zero.",
        type="bool",
        default="True",
    )

    absorption_data_sets = documented(
        attr.ib(
            default=None,
            converter=attr.converters.optional(dict),
            validator=attr.validators.optional(attr.validators.instance_of(dict)),
        ),
        doc="Mapping of species and absorption data set files paths. If "
        "``None``, the default absorption data sets are used to compute "
        "the absorption coefficient. If not ``None``, the absorption data "
        "set files whose paths are provided in the mapping will be used to "
        "compute the absorption coefficient. If the mapping does not "
        "include all species from the AFGL (1986) atmospheric "
        "thermophysical profile, the default data sets will be used to "
        "compute the absorption coefficient of the corresponding species.",
        type="dict",
    )

    def eval_sigma_a(self, spectral_ctx: SpectralContext) -> ureg.Quantity:
        """
        Evaluate absorption coefficient given a spectral context.

        .. note:: Extrapolate to zero when wavelength, pressure and/or
           temperature are out of bounds.
        """
        if self.has_absorption:
            wavelength = spectral_ctx.wavelength

            p = self.thermoprops.p
            t = self.thermoprops.t
            n = self.thermoprops.n
            mr = self.thermoprops.mr

            sigma_a = np.full(mr.shape, np.nan)
            absorbers = [
                Absorber.CH4,
                Absorber.CO,
                Absorber.CO2,
                Absorber.H2O,
                Absorber.N2O,
                Absorber.O2,
                Absorber.O3,
            ]
            if self.absorption_data_sets is None:
                self.absorption_data_sets = {}

            for i, absorber in enumerate(absorbers):
                n_absorber = n * mr.sel(species=absorber.value)

                if absorber.value in self.absorption_data_sets:
                    sigma_a_absorber = self._compute_sigma_a_absorber_from_data_set(
                        path=self.absorption_data_sets[absorber.value],
                        wavelength=wavelength,
                        n_absorber=n_absorber,
                        p=p,
                        t=t,
                    )
                else:
                    sigma_a_absorber = self._auto_compute_sigma_a_absorber(
                        wavelength=wavelength,
                        absorber=absorber,
                        n_absorber=n_absorber,
                        p=p,
                        t=t,
                    )

                sigma_a[i, :] = sigma_a_absorber.m_as("km^-1")

            sigma_a = np.sum(sigma_a, axis=0)

            return ureg.Quantity(sigma_a, "km^-1")
        else:
            return ureg.Quantity(np.zeros(self.thermoprops.z_layer.size), "km^-1")

    def eval_sigma_s(self, spectral_ctx: SpectralContext) -> ureg.Quantity:
        """
        Evaluate scattering coefficient given a spectral context.
        """
        if self.has_scattering:
            return compute_sigma_s_air(
                wavelength=spectral_ctx.wavelength,
                number_density=ureg.Quantity(
                    self.thermoprops.n.values, self.thermoprops.n.units
                ),
            )
        else:
            return ureg.Quantity(np.zeros(self.thermoprops.z_layer.size), "km^-1")

    @property
    def thermoprops(self) -> xr.Dataset:
        """
        Return the atmosphere thermophysical properties.

        Returns → :class:`~xarray.Dataset`:
            Atmosphere thermophysical properties.
        """
        ds = afgl1986.make_profile(model_id=self.model)
        if self.levels is not None:
            ds = interpolate(ds=ds, z_level=self.levels, conserve_columns=True)

        if self.concentrations is not None:
            factors = compute_scaling_factors(ds=ds, concentration=self.concentrations)
            ds = rescale_concentration(ds=ds, factors=factors)

        return ds

    @property
    def radprops(self, spectral_ctx: SpectralContext = None) -> xr.Dataset:
        """
        Return the atmosphere radiative properties.

        Returns → :class:`~xarray.Dataset`:
            Radiative properties dataset.
        """
        return make_dataset(
            wavelength=spectral_ctx.wavelength,
            z_level=to_quantity(self.thermoprops().z_level),
            z_layer=to_quantity(self.thermoprops().z_layer),
            sigma_a=self.sigma_a(spectral_ctx),
            sigma_s=self.sigma_s(spectral_ctx),
        )

    def get_levels(self) -> ureg.Quantity:
        return self.levels

    def get_phase(self, spectral_ctx: SpectralContext):
        return RayleighPhaseFunction(id="phase_atmosphere")

    def get_sigma_a(self, spectral_ctx: SpectralContext = None) -> ureg.Quantity:
        return self.eval_sigma_a(spectral_ctx)

    def get_sigma_s(self, spectral_ctx: SpectralContext = None) -> ureg.Quantity:
        return self.eval_sigma_s(spectral_ctx)

    @staticmethod
    def _auto_compute_sigma_a_absorber(
        wavelength: ureg.Quantity,
        absorber: Absorber,
        n_absorber: ureg.Quantity,
        p: ureg.Quantity,
        t: ureg.Quantity,
    ) -> ureg.Quantity:
        """
        Compute the absorption coefficient using the predefined absorption
        data sets.
        """
        # ! This method is never tested because it requires large absorption
        # data sets to be downloaded
        wavenumber = 1.0 / wavelength
        try:
            dataset_id = find_dataset(
                wavenumber=wavenumber,
                absorber=absorber,
                engine=Engine.SPECTRA,
            )
            dataset = data.open(category="absorption_spectrum", id=dataset_id)
            sigma_a_absorber = compute_sigma_a(
                ds=dataset,
                wl=wavelength,
                p=p.values,
                t=t.values,
                n=n_absorber.values,
                fill_values=dict(
                    w=0.0, pt=0.0
                ),  # extrapolate to zero along wavenumber and pressure and temperature dimensions
            )
        except ValueError:  # no data at current wavelength/wavenumber
            sigma_a_absorber = ureg.Quantity(np.zeros(len(p)), "km^-1")

        return sigma_a_absorber

    @staticmethod
    def _compute_sigma_a_absorber_from_data_set(
        path: str,
        wavelength: ureg.Quantity,
        n_absorber: ureg.Quantity,
        p: ureg.Quantity,
        t: ureg.Quantity,
    ) -> ureg.Quantity:

        """
        Compute the absorption coefficient using a custom absorption data set
        file.

        Parameter ``path`` (str):
            Path to data set file.

        Parameter ``wavelength`` (:class:`~pint.Quantity`):
            Wavelength  [nm].

        Parameter ``n_absorber`` (:class:`~pint.Quantity`):
            Absorber number density [m^-3].

        Parameter ``p`` (:class:`~pint.Quantity`):
            Pressure [Pa].

        Parameter ``t`` (:class:`~pint.Quantity`):
            Temperature [K].

        Returns → :class:`~pint.Quantity`:
            Absorption coefficient [km^-1].
        """
        dataset = xr.open_dataset(path)
        return compute_sigma_a(
            ds=dataset,
            wl=wavelength,
            p=p.values,
            t=t.values,
            n=n_absorber.values,
            fill_values=dict(
                w=0.0, pt=0.0
            ),  # extrapolate to zero along wavenumber and pressure and temperature dimensions
        )
