import pytest

from eradiate.scenes.atmosphere.molecules import MolecularAtmosphere


def test_molecular_atmosphere_is_abstract():
    with pytest.raises(TypeError):
        MolecularAtmosphere()
