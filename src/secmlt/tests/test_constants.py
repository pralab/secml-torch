from secmlt.adv import Backends
from secmlt.adv.evasion import LpPerturbationModels


def test_backends() -> None:
    assert hasattr(Backends, "FOOLBOX")
    assert hasattr(Backends, "NATIVE")
    assert Backends.FOOLBOX == "foolbox"
    assert Backends.NATIVE == "native"


def test_perturbation_models() -> None:
    assert hasattr(LpPerturbationModels, "L0")
    assert hasattr(LpPerturbationModels, "L1")
    assert hasattr(LpPerturbationModels, "L2")
    assert hasattr(LpPerturbationModels, "LINF")
