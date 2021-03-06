import numpy as np
import pinttr
import pytest

from eradiate import unit_context_config as ucc
from eradiate import unit_context_kernel as uck
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.spectra._interpolated import InterpolatedSpectrum


def test_spectra_interpolated_construct(mode_mono):
    # Instantiating without argument fails
    with pytest.raises(TypeError):
        InterpolatedSpectrum()
    # Instantiating with missing argument fails
    with pytest.raises(ValueError):
        InterpolatedSpectrum(wavelengths=[500.0, 600.0])
    with pytest.raises(TypeError):
        InterpolatedSpectrum(values=[0.0, 1.0])
    # Shape mismatch raises
    with pytest.raises(ValueError):
        InterpolatedSpectrum(wavelengths=[500.0, 600.0], values=[0.0, 1.0, 2.0])

    # Instantiating with no quantity applies no units
    spectrum = InterpolatedSpectrum(wavelengths=[500.0, 600.0], values=[0.0, 1.0])
    assert isinstance(spectrum.values, np.ndarray)
    # Instantiating with a quantity applies units
    spectrum = InterpolatedSpectrum(
        quantity="irradiance", wavelengths=[500.0, 600.0], values=[0.0, 1.0]
    )
    assert spectrum.values.is_compatible_with("W/m^2/nm")

    # Inconsistent units raise
    with pytest.raises(pinttr.exceptions.UnitsError):
        InterpolatedSpectrum(
            quantity="irradiance",
            wavelengths=[500.0, 600.0],
            values=[0.0, 1.0] * ureg["W/m^2"],
        )
    with pytest.raises(pinttr.exceptions.UnitsError):
        InterpolatedSpectrum(
            quantity="irradiance",
            wavelengths=[500.0, 600.0] * ureg["s"],
            values=[0.0, 1.0],
        )

    # Instantiate from dictionary
    assert (
        InterpolatedSpectrum.from_dict(
            {
                "quantity": "irradiance",
                "wavelengths": [0.5, 0.6],
                "wavelengths_units": "micron",
                "values": [0.5, 1.0],
                "values_units": "kW/m^2/nm",
            }
        )
        is not None
    )


def test_interpolated_eval(mode_mono):
    spectral_ctx = SpectralContext.new(wavelength=550.0)

    # Spectrum without quantity performs linear interpolation and yields units
    # consistent with values
    spectrum = InterpolatedSpectrum(wavelengths=[500.0, 600.0], values=[0.0, 1.0])
    assert spectrum.eval(spectral_ctx) == 0.5
    spectrum.values *= ureg["W/m^2/nm"]
    assert spectrum.eval(spectral_ctx) == 0.5 * ureg["W/m^2/nm"]

    # Spectrum with quantity performs linear interpolation and yields units
    # consistent with quantity
    spectrum = InterpolatedSpectrum(
        quantity="irradiance", wavelengths=[500.0, 600.0], values=[0.0, 1.0]
    )
    # Interpolation returns quantity
    assert spectrum.eval(spectral_ctx) == 0.5 * ucc.get("irradiance")


def test_spectra_interpolated_kernel_dict(mode_mono):
    from mitsuba.core.xml import load_dict

    ctx = KernelDictContext(spectral_ctx=SpectralContext.new(wavelength=550.0))

    spectrum = InterpolatedSpectrum(
        id="spectrum",
        quantity="irradiance",
        wavelengths=[500.0, 600.0],
        values=[0.0, 1.0],
    )

    # Produced kernel dict is valid
    assert load_dict(spectrum.kernel_dict(ctx)["spectrum"]) is not None

    # Unit scaling is properly applied
    with ucc.override({"radiance": "W/m^2/sr/nm"}):
        s = InterpolatedSpectrum(
            quantity="radiance", wavelengths=[500.0, 600.0], values=[0.0, 1.0]
        )
    with uck.override({"radiance": "kW/m^2/sr/nm"}):
        d = s.kernel_dict(ctx)
        assert np.allclose(d["spectrum"]["value"], 5e-4)
