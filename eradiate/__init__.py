"""The Eradiate radiative transfer simulation software package."""

__version__ = "0.0.1"  #: Eradiate version number.

import attr

from .util.units import ureg as _ureg
from .util.collections import configdict as _configdict


@attr.s
class EradiateMode:
    """A very simple container for Eradiate mode configuration."""
    type = attr.ib(default=None)  #: Mode type (str or None).
    config = attr.ib(default=None)  #: Mode configuration (dict).


mode = EradiateMode()
"""Eradiate's mode configuration.

This is the only instance of :class:`EradiateMode`. 
See also :func:`set_mode`.
"""


_mode_default_configs = {
    "mono": {
        "wavelength": 550.,
        "wavelength_unit": _ureg("nm")
    }
}


def set_mode(mode_type, **kwargs):
    """Set Eradiate's mode of operation.

    This function sets and configures Eradiate's mode of operation. In addition,
    it invokes :func:`~mitsuba.set_variant` to select the kernel variant
    corresponding to the selected mode.

    The main argument ``mode_type`` defines which mode is selected. Then,
    keyword arguments are used to pass additional configuration details for the
    selected mode. The mode configuration is critical since many code components
    (_e.g._ spectrum-related components) adapt their behaviour based on the
    selected mode.

    Parameter ``mode_type`` (str):
        Mode to be selected (see list below).

    Valid keyword arguments for ``mono`` (monochromatic mode):
        ``wavelength`` (float):
            Wavelength selected for monochromatic operation [nm].
            Default: 550 nm.
    """
    global mode

    if mode_type not in _mode_default_configs.keys():
        raise ValueError(f"unsupported mode {mode_type}")

    if mode_type == "mono":
        import eradiate.kernel
        eradiate.kernel.set_variant("scalar_mono_double")

        mode.type = "mono"
        mode.config = _configdict(_mode_default_configs[mode_type])
        mode.config.update(kwargs)
