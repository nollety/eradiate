import numpy as np
import pytest

from eradiate import path_resolver
from eradiate import unit_registry as ureg
from eradiate.contexts import KernelDictContext, SpectralContext
from eradiate.scenes.atmosphere.molecules.afgl1986 import AFGL1986MolecularAtmosphere
from eradiate.scenes.core import KernelDict


@pytest.fixture
def afgl1986_test_absorption_data_sets():
    return {
        "CH4": path_resolver.resolve(
            "tests/spectra/absorption/CH4-spectra-4000_11502.nc"
        ),
        "CO2": path_resolver.resolve(
            "tests/spectra/absorption/CO2-spectra-4000_14076.nc"
        ),
        "CO": path_resolver.resolve(
            "tests/spectra/absorption/CO-spectra-4000_14478.nc"
        ),
        "H2O": path_resolver.resolve(
            "tests/spectra/absorption/H2O-spectra-4000_25711.nc"
        ),
        "N2O": path_resolver.resolve(
            "tests/spectra/absorption/N2O-spectra-4000_10364.nc"
        ),
        "O2": path_resolver.resolve(
            "tests/spectra/absorption/O2-spectra-4000_17273.nc"
        ),
        "O3": path_resolver.resolve("tests/spectra/absorption/O3-spectra-4000_6997.nc"),
    }


def test_afgl1986_molecular_atmosphere_default(
    mode_mono, afgl1986_test_absorption_data_sets
):
    """
    Default constructor produces molecular atmosphere with expected
    collision coefficient values.
    """
    # Default constructor with test absorption data sets (in the infrared,
    # all absorption data sets are opened)
    spectral_ctx = SpectralContext.new(wavelength=1500.0)

    atmosphere = AFGL1986MolecularAtmosphere(
        absorption_data_sets=afgl1986_test_absorption_data_sets
    )
    # Collision coefficients are Quantity objects
    for field in ["sigma_a", "sigma_s", "sigma_t", "albedo"]:
        x = getattr(atmosphere, f"get_{field}")(spectral_ctx)
        assert isinstance(x, ureg.Quantity)
        assert x.shape == (120,)

    # The resulting object produces a valid kernel dictionary
    ctx = KernelDictContext(spectral_ctx=SpectralContext.new())
    kernel_dict = KernelDict.new(atmosphere, ctx=ctx)
    assert kernel_dict.load() is not None


def test_afgl1986_molecular_atmosphere_levels(
    mode_mono, afgl1986_test_absorption_data_sets
):
    # Custom level altitudes (in the visible, only the H2O data set is opened)
    spectral_ctx = SpectralContext.new(wavelength=550.0)

    atmosphere = AFGL1986MolecularAtmosphere(
        levels=ureg.Quantity(np.linspace(0, 100, 101), "km"),
        absorption_data_sets=afgl1986_test_absorption_data_sets,
    )

    # The resulting object produces a valid kernel dictionary
    ctx = KernelDictContext(spectral_ctx=SpectralContext.new())
    kernel_dict = KernelDict.new(atmosphere, ctx=ctx)
    assert kernel_dict.load() is not None
