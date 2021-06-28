import pytest

from eradiate.scenes.atmosphere._heterogeneous_new import HeterogeneousAtmosphereNew


def test_heterogeneous_new_is_abstract():
    with pytest.raises(TypeError):
        HeterogeneousAtmosphereNew()


def test_heterogeneous_new_inheritance(mode_mono):
    """
    Concrete classes implementing abstract class methods can be
    instanciated and HeterogeneousAtmosphere methods return types
    are correct.
    """
    import numpy as np

    from eradiate import unit_registry as ureg
    from eradiate.contexts import SpectralContext, KernelDictContext
    from eradiate.scenes.core import KernelDict
    from eradiate.scenes.phase import RayleighPhaseFunction

    class TestClass(HeterogeneousAtmosphereNew):
        def get_levels(self):
            return ureg.Quantity(np.linspace(0, 100, 11), "km")

        def get_phase(self, spectral_ctx):
            return RayleighPhaseFunction(id="phase_atmosphere")

        def get_sigma_a(self, spectral_ctx):
            return ureg.Quantity(1e-5 * np.ones(11), "km^-1")

        def get_sigma_s(self, spectral_ctx):
            return ureg.Quantity(9.99e-3 * np.ones(11), "km^-1")

    test_atm = TestClass()
    ctx = KernelDictContext(spectral_ctx=SpectralContext.new())

    assert isinstance(test_atm.get_albedo(spectral_ctx=ctx.spectral_ctx), ureg.Quantity)
    assert np.allclose(test_atm.get_albedo(spectral_ctx=ctx.spectral_ctx), 0.999)
    assert isinstance(test_atm.kernel_width(ctx=ctx), ureg.Quantity)
    assert isinstance(test_atm.kernel_phase(ctx=ctx), dict)
    assert isinstance(test_atm.kernel_media(ctx=ctx), dict)
    assert isinstance(test_atm.kernel_shapes(ctx=ctx), dict)
    assert isinstance(
        test_atm.get_sigma_t(spectral_ctx=ctx.spectral_ctx), ureg.Quantity
    )
    assert np.allclose(
        test_atm.get_sigma_t(spectral_ctx=ctx.spectral_ctx),
        ureg.Quantity(1e-2, "km^-1"),
    )

    # The resulting object produces a valid kernel dictionary
    kernel_dict = KernelDict.new(test_atm, ctx=ctx)
    assert kernel_dict.load() is not None
