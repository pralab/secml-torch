from secmlt.adv.backends import Backends


def test_backends():
    assert hasattr(Backends, "FOOLBOX")
    assert hasattr(Backends, "NATIVE")
    assert Backends.FOOLBOX == "foolbox"
    assert Backends.NATIVE == "native"
