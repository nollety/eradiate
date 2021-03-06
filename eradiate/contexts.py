from abc import ABC, abstractmethod
from typing import Dict, Optional

import attr
import pint
import pinttr

import eradiate

from .attrs import documented, parse_docs
from .exceptions import ModeError
from .units import unit_context_config as ucc
from .units import unit_registry as ureg

# -- Spectral contexts ---------------------------------------------------------


@attr.s
class SpectralContext(ABC):
    """
    Context data structure holding state relevant to the evaluation of spectrally
    dependent objects.

    This object is usually used as part of a :class:`.KernelDictContext` to pass
    around spectral information to kernel dictionary emission methods which
    require spectral configuration information.

    While this class is abstract, it should however be the main entry point
    to create :class:`.SpectralContext` child class objects through the
    :meth:`.SpectralContext.new` class method constructor.
    """

    @property
    @abstractmethod
    def wavelength(self):
        # Wavelength associated with spectral context
        # (may raise NotImplementedError if irrelevant)
        pass

    @staticmethod
    def new(**kwargs) -> "SpectralContext":
        """
        Create a new instance of one of the :class:`SpectralContext` child
        classes. *The instantiated class is defined based on the currently active
        mode.* Keyword arguments are passed to the instantiated class's
        constructor:

        .. rubric:: Monochromatic modes [:class:`MonoSpectralContext`]

        Parameter ``wavelength`` (float):
            Wavelength. Default: 550 nm.

            Unit-enabled field (default: ucc[wavelength]).

        .. seealso::

           * :func:`eradiate.mode`
           * :func:`eradiate.set_mode`
        """
        mode = eradiate.mode()

        if mode.is_monochromatic():
            return MonoSpectralContext(**kwargs)

        raise ModeError(f"unsupported mode '{mode.id}'")

    @staticmethod
    def from_dict(d: Dict) -> "SpectralContext":
        """
        Create from a dictionary. This class method will additionally pre-process
        the passed dictionary to merge any field with an associated ``"_units"``
        field into a :class:`pint.Quantity` container.

        Parameter ``d`` (dict):
            Configuration dictionary used for initialisation.

        Returns → :class:`.SpectralContext`:
            Created object.
        """

        # Pre-process dict: apply units to unit-enabled fields
        d_copy = pinttr.interpret_units(d, ureg=ureg)

        # Perform object creation
        return SpectralContext.new(**d_copy)

    @staticmethod
    def convert(value):
        """
        Object converter method.

        If ``value`` is a dictionary, this method uses :meth:`from_dict` to
        create a :class:`.SpectralContext`.

        Otherwise, it returns ``value``.
        """
        if isinstance(value, dict):
            return SpectralContext.from_dict(value)

        return value


@parse_docs
@attr.s
class MonoSpectralContext(SpectralContext):
    """
    Monochromatic spectral context data structure.
    """

    _wavelength: pint.Quantity = documented(
        pinttr.ib(
            default=ureg.Quantity(550.0, ureg.nm),
            units=ucc.deferred("wavelength"),
        ),
        doc="A single wavelength value.\n\nUnit-enabled field "
        "(default: ucc[wavelength]).",
        type="float",
        default="550.0 nm",
    )

    @property
    def wavelength(self):
        return self._wavelength

    @wavelength.setter
    def wavelength(self, value):
        self._wavelength = value


# -- Kernel dictionary contexts ------------------------------------------------


@parse_docs
@attr.s
class KernelDictContext:
    """
    Kernel dictionary evaluation context data structure. This class is used
    *e.g.* to store information about the spectral configuration to apply
    when generating kernel dictionaries associated with a :class:`.SceneElement`
    instance.
    """

    spectral_ctx: SpectralContext = documented(
        attr.ib(
            factory=SpectralContext.new,
            converter=SpectralContext.convert,
            validator=attr.validators.instance_of(SpectralContext),
        ),
        doc="Spectral context (used to evaluate quantities with any degree "
        "or kind of dependency vs spectrally varying quantities).",
        type=":class:`.SpectralContext`",
        default=":meth:`SpectralContext.new() <.SpectralContext.new>`",
    )

    ref: bool = documented(
        attr.ib(default=True, converter=bool),
        doc="If ``True``, use references when relevant during kernel dictionary "
        "generation.",
        type="bool",
        default="True",
    )

    override_surface_width: Optional[pint.Quantity] = documented(
        pinttr.ib(default=None, units=ucc.deferred("length")),
        doc="If relevant, value which must be used as the surface width "
        "(*e.g.* when surface size must match atmosphere or canopy size).\n"
        "\n"
        "Unit-enabled field (default: cdu[length]).",
        type="float or None",
        default="None",
    )
