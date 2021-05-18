from abc import ABC, abstractmethod, abstractproperty, abstractstaticmethod
import attr
import numpy as np
from scipy.constants import physical_constants

from eradiate import unit_registry as ureg

# Physical constants
_LOSCHMIDT = ureg.Quantity(
    *physical_constants["Loschmidt constant (273.15 K, 101.325 kPa)"][:2]
)


@attr.s
class RefractiveIndexCalculator(ABC):
    """
    Abstract class for a refractive index calculator.

    The reference property hold the bibliographical reference to the formula
    used to compute the refractive index.
    The eval method implements the latter formula. The formula can take two
    parameters; wavelength (w) and number density (n).
    """

    reference = attr.ib(
        default="Unknown", converter=str, validator=attr.validators.instance_of(str)
    )

    @abstractstaticmethod
    def eval(w, n):
        """
        Evaluate refractive index at given wavelength and number density.

        Parameter ``w`` (float):
            Wavelength.

        Parameter ``n`` (float):
            Number density.

        Returns → float:
            Refractive index.
        """
        pass


class N2RefractiveIndex(RefractiveIndexCalculator):
    @property
    def reference(self):
        return "Bates1984RayleighScatteringAir"

    def eval(w, n):
        if 1e6 / np.power(w, 2) == 144:
            raise ZeroDivisionError
        if w >= 468:
            a = 6855.200
            b = 3243157.0
        elif 254 < w < 468:
            a = 5989.242
            b = 3363266.3
        elif 0 < w < 254:
            a = 6998.749
            b = 3233582.0
        else:
            raise ValueError("wavelength must be strictly positive")

        return 1 + (a + b / (144 - (1e6 / np.power(w, 2)))) * 1e-8


@attr.s
class KingFactorCalculator:
    @abstractproperty
    def reference(self):
        pass

    @abstractmethod
    def eval(self, w, n):
        """
        Evaluate King factor index at given wavelength and number density.

        Parameter ``w`` (float):
            Wavelength.

        Parameter ``n`` (float):
            Number density.

        Returns → float:
            King factor.
        """
        pass


class ArKingFactor(KingFactorCalculator):
    @property
    def reference(self):
        return "Alms1975MeasurementDispersionPolarizabilities"

    def eval(self, w, n):
        return 1.0


class CO2KingFactor(KingFactorCalculator):
    @property
    def reference(self):
        return "Alms1975MeasurementDispersionPolarizabilities"

    def eval(self, w, n):
        return 1.15


class H2OKingFactor(KingFactorCalculator):
    @property
    def reference(self):
        return "Tomasi2005ImprovedAlgorithmCalculations"

    def eval(self, w, n):
        return 1.001


class N2KingFactor(KingFactorCalculator):
    @property
    def reference(self):
        return "Bates1984RayleighScatteringAir"

    def eval(self, w, n):
        return n / _LOSCHMIDT * 1.034 + 317 / np.power(w, 2)


class O2KingFactor(KingFactorCalculator):
    @property
    def reference(self):
        return "Bates1984RayleighScatteringAir"

    def eval(self, w, n):
        return 1.096 + 1385 / np.power(w, 2) + 1.448e8 / np.power(w, 4)


@attr.s
class Molecule:
    name = attr.ib(validator=attr.validators.instance_of(str))  # str
    refractive_index_implementation = attr.ib()  # callable
    king_factor_implementation = attr.ib()  # callable

    def refractive_index(self, w, n):
        return self.refractive_index_implementation(w, n)

    def king_factor(self, w, n):
        f = self.king_factor_implementation(w)
        return f * n / _LOSCHMIDT

    def _king_factor():
        pass
