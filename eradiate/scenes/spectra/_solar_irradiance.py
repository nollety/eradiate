import attr
import numpy as np

import eradiate
from ... import data
from ... import unit_context_kernel as uck
from ... import unit_registry as ureg
from ..._attrs import documented, parse_docs
from ..._units import PhysicalQuantity
from ...scenes.spectra import Spectrum, SpectrumFactory
from ...validators import is_positive


@SpectrumFactory.register("solar_irradiance")
@parse_docs
@attr.s(frozen=True)
class SolarIrradianceSpectrum(Spectrum):
    """
    Solar irradiance spectrum scene element [:factorykey:`solar_irradiance`].

    This scene element produces the scene dictionary required to
    instantiate a kernel plugin using the Sun irradiance spectrum. The data set
    used by this element is controlled by the ``dataset`` attribute (see
    :mod:`eradiate.data.solar_irradiance_spectra` for available data sets).

    The spectral range of the data sets shipped can vary and an attempt for use
    outside of the supported spectral range will raise a :class:`ValueError`
    upon calling :meth:`kernel_dict`.

    The generated kernel dictionary varies based on the selected mode of
    operation. The ``scale`` parameter can be used to adjust the value based on
    unit conversion or to account for variations of the Sun-planet distance.

    The produced kernel dictionary automatically adjusts its irradiance units
    depending on the selected kernel default units.
    """

    quantity = attr.ib(default=PhysicalQuantity.IRRADIANCE, init=False, repr=False)

    dataset = documented(
        attr.ib(
            default="thuillier_2003",
            validator=attr.validators.instance_of(str),
        ),
        doc="Dataset identifier. Allowed values: see "
        ":attr:`solar irradiance dataset documentation <eradiate.data.solar_irradiance_spectra>`. "
        'Default: ``"thuillier_2003"``. ',
        type="str",
    )

    scale = documented(
        attr.ib(
            default=1.0,
            converter=float,
            validator=is_positive,
        ),
        doc="Scaling factor. Default: 1.",
        type="float",
    )

    @dataset.validator
    def _dataset_validator(self, attribute, value):
        if value not in data.registered("solar_irradiance_spectrum"):
            raise ValueError(
                f"while setting {attribute.name}: '{value}' not in "
                f"list of supported solar irradiance spectra "
                f"{data.registered('solar_irradiance_spectrum')}"
            )

    data = attr.ib(init=False, repr=False)

    @data.default
    def _data_factory(self):
        # Load dataset
        try:
            return data.open("solar_irradiance_spectrum", self.dataset)
        except KeyError:
            raise ValueError(f"unknown dataset {self.dataset}")

    def eval(self, spectral_ctx=None):
        if self.dataset == "solid_2017":
            raise NotImplementedError(
                "Solar irradiance spectrum datasets with a non-empty time "
                "coordinate are not supported yet."
            )
        # TODO: add support to solar irradiance spectrum datasets with a
        #  non-empty time coordinate

        if eradiate.mode().is_monochromatic():
            wavelength = spectral_ctx.wavelength.m_as(self.data.w.attrs["units"])

            irradiance_magnitude = float(
                self.data.ssi.interp(
                    w=wavelength,
                    method="linear",
                ).values
            )

            # Raise if out of bounds or ill-formed dataset
            if np.isnan(irradiance_magnitude):
                raise ValueError("dataset evaluation returned nan")

            # Apply units
            irradiance = ureg.Quantity(
                irradiance_magnitude, self.data.ssi.attrs["units"]
            )

        else:
            raise ValueError(f"unsupported mode {eradiate.mode().id}")

        return irradiance

    def kernel_dict(self, ctx=None):
        # Apply scaling, build kernel dict
        return {
            "spectrum": {
                "type": "uniform",
                "value": self.eval(ctx.spectral_ctx).m_as(uck.get("irradiance"))
                * self.scale,
            }
        }