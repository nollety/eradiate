import attr

from .._heterogeneous_new import HeterogeneousAtmosphereNew


@attr.s
class MolecularAtmosphere(HeterogeneousAtmosphereNew):
    """
    An abstract class defining common facilities for all molecular atmospheres.
    """