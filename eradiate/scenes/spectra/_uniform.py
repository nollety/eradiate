import attr
import pint
import pinttr

from ... import unit_context_config as ucc
from ... import unit_context_kernel as uck
from ..._attrs import documented, parse_docs
from ...scenes.spectra import Spectrum, SpectrumFactory
from ...validators import is_positive


@SpectrumFactory.register("uniform")
@parse_docs
@attr.s
class UniformSpectrum(Spectrum):
    """
    Uniform spectrum (*i.e.* constant against wavelength).
    """

    value = documented(
        attr.ib(default=1.0),
        doc="Uniform spectrum value. If a float is passed and ``quantity`` is not "
        "``None``, it is automatically converted to appropriate configuration "
        "default units. If a :class:`~pint.Quantity` is passed and ``quantity`` "
        "is not ``None``, units will be checked for consistency.",
        type="float or :class:`~pint.Quantity`",
    )

    @value.validator
    def value_validator(self, attribute, value):
        if self.quantity is not None and isinstance(value, pint.Quantity):
            expected_units = ucc.get(self.quantity)

            if not pinttr.util.units_compatible(expected_units, value.units):
                raise pinttr.exceptions.UnitsError(
                    value.units,
                    expected_units,
                    extra_msg=f"while validating {attribute.name}, got units "
                    f"'{value.units}' incompatible with quantity {self.quantity} "
                    f"(expected '{expected_units}')",
                )

        is_positive(self, attribute, value)

    def __attrs_post_init__(self):
        self.update()

    def update(self):
        if self.quantity is not None and self.value is not None:
            self.value = pinttr.converters.ensure_units(
                self.value, ucc.get(self.quantity)
            )

    def eval(self, spectral_ctx=None):
        return self.value

    def kernel_dict(self, ctx=None):
        kernel_units = uck.get(self.quantity)
        spectral_ctx = ctx.spectral_ctx if ctx is not None else None

        return {
            "spectrum": {
                "type": "uniform",
                "value": self.eval(spectral_ctx).m_as(kernel_units),
            }
        }