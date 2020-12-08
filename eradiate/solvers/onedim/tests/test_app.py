import pytest

import eradiate
from eradiate.solvers.onedim.app import OneDimSolverApp


def test_onedim_solver_app_app():
    # Test default configuration handling
    app = OneDimSolverApp()
    assert app.config == {
        "atmosphere": {"type": "rayleigh_homogeneous"},
        "illumination": {"type": "directional"},
        "measure": [{
            "azimuth_res": 10,
            "hemisphere": "back",
            "id": "toa_hsphere",
            "origin": [0, 0, 100.1],
            "spp": 32,
            "type": "radiancemeter_hsphere",
            "zenith_res": 10
        }],
        "mode": {"type": "mono", "wavelength": 550.0},
        "surface": {"type": "lambertian"}
    }

    # Check that the appropriate variant is selected
    assert eradiate.mode.id == "mono"

    # Check that the default scene can be instantiated
    assert app._kernel_dict.load() is not None

    # Pass a well-formed custom configuration object (without an atmosphere)
    config = {
        "mode": {
            "type": "mono",
            "wavelength": 800.
        },
        "illumination": {
            "type": "directional",
            "zenith": 10.,
            "azimuth": 0.,
            "irradiance": {"type": "uniform", "value": 1.}
        },
        "measure": [{
            "type": "toa_hsphere_lo",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        }],
        "surface": {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.35},
        },
        "atmosphere": None,
    }
    app = OneDimSolverApp(config)
    assert app._kernel_dict.load() is not None

    # Pass a well-formed custom configuration object (with an atmosphere)
    config = {
        "mode": {
            "type": "mono",
            "wavelength": 550.
        },
        "illumination": {
            "type": "directional",
            "zenith": 0.,
            "azimuth": 0.,
            "irradiance": {"type": "uniform", "value": 1.}
        },
        "measure": [{
            "type": "toa_hsphere_lo",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
        }],
        "surface": {
            "type": "lambertian",
            "reflectance": {"type": "uniform", "value": 0.5}
        },
        "atmosphere": {
            "type": "rayleigh_homogeneous",
            "height": 1e5,
            "sigma_s": 1e-6
        }
    }
    app = OneDimSolverApp(config)
    assert app._kernel_dict.load() is not None

    # Test measure aliasing
    app1 = OneDimSolverApp({"measure": [{"type": "toa_hsphere", "id": "test_measure"}]})
    app2 = OneDimSolverApp({"measure": [{"type": "toa_hsphere_lo", "id": "test_measure"}]})
    app3 = OneDimSolverApp({"measure": [{"type": "toa_hsphere_brdf", "id": "test_measure"}]})
    app4 = OneDimSolverApp({"measure": [{"type": "toa_hsphere_brf", "id": "test_measure"}]})
    assert app1._kernel_dict == app2._kernel_dict
    assert app1._kernel_dict == app3._kernel_dict
    assert app1._kernel_dict == app4._kernel_dict


@pytest.mark.slow
def test_onedim_solver_app_run():
    """Test the creation of a DataArray from the solver result

    We create a default scene with a set of zenith and azimuth angles,
    render the scene and create the DataArray.

    We assert the correct setting of the DataArray coordinates and dimensions,
    as well as the correct setting of data.
    """
    import numpy as np

    config = {
        "measure": [{
            "type": "toa_hsphere",
            "zenith_res": 45.,
            "azimuth_res": 180.,
            "spp": 1000,
            "hemisphere": "back"
        }]
    }

    app = OneDimSolverApp(config)
    # Assert the correct mode of operation to be set by the application
    assert eradiate.mode.id == "mono"

    app.run()

    results = app.results["toa_hsphere"]

    # Assert the correct dimensions of the application's results
    assert set(results["lo"].dims) == {"sza", "saa", "vza", "vaa", "wavelength"}

    # We expect the whole [0, 360] to be covered
    assert len(results["lo"].coords["vaa"]) == 360 / 180
    # # We expect [0, 90[ to be covered (90° should be missing)
    assert len(results["lo"].coords["vza"]) == 90 / 45
    # We just check that we record something as expected
    assert np.all(results["lo"].data > 0)


def test_rayleigh_solver_app_postprocessing():
    """Test the postprocessing method by computing the processed quantities and comparing
    them to a reference computation."""

    import numpy as np
    config = {
        "measure": [{
            "type": "toa_hsphere",
            "zenith_res": 5.,
            "azimuth_res": 10.,
            "spp": 1000,
            "hemisphere": "back"
        }],
        "illumination": {
            "type": "directional",
            "zenith": 0,
            "azimuth": 0,
            "irradiance": {"type": "uniform", "value": 5.}
        }
    }

    app = OneDimSolverApp(config)
    # Assert the correct mode of operation to be set by the application
    assert eradiate.mode.id == "mono"
    app.run()

    results = app.results["toa_hsphere"]

    # Assert the correct computation of the BRDF and BRF values
    # BRDF
    assert np.allclose(
        results["brdf"],
        results["lo"] / config["illumination"]["irradiance"]["value"]
    )
    # BRF
    assert np.allclose(
        results["brf"],
        results["brdf"] * np.pi
    )