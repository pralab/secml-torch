from unittest import mock

import pytest
import secmlt.models.pytorch as pytorch_models
from secmlt import adv, metrics, optimization, trackers
from secmlt.adv import Backends, evasion
from secmlt.adv.evasion import DDN, FMN, PGD, LpPerturbationModels
from secmlt.metrics import Accuracy
from secmlt.models.pytorch import BasePyTorchClassifier, BasePyTorchTrainer
from secmlt.optimization import MaskConstraint
from secmlt.trackers import (
    ImageGradientsTracker,
    ImageSampleTracker,
    Tracker,
)


def test_package_level_imports():
    assert {"Backends"} <= set(adv.__all__)
    assert all(hasattr(adv, name) for name in adv.__all__)
    assert adv.Backends is Backends
    assert {"DDN", "FMN", "LpPerturbationModels", "PGD"} <= set(evasion.__all__)
    assert all(hasattr(evasion, name) for name in evasion.__all__)
    assert evasion.DDN is DDN
    assert evasion.FMN is FMN
    assert evasion.LpPerturbationModels is LpPerturbationModels
    assert evasion.PGD is PGD
    assert metrics.Accuracy is Accuracy
    assert pytorch_models.BasePyTorchClassifier is BasePyTorchClassifier
    assert pytorch_models.BasePyTorchTrainer is BasePyTorchTrainer
    assert optimization.MaskConstraint is MaskConstraint


def test_tracker_package_exports():
    expected_exports = {
        "ImageGradientsTracker",
        "ImageSampleTracker",
        "IMAGE",
        "MULTI_SCALAR",
        "SCALAR",
        "Tracker",
    }

    assert expected_exports <= set(trackers.__all__)
    assert all(hasattr(trackers, name) for name in trackers.__all__)
    assert trackers.ImageGradientsTracker is ImageGradientsTracker
    assert trackers.ImageSampleTracker is ImageSampleTracker
    assert trackers.Tracker is Tracker


@pytest.mark.parametrize(
    "missing_module, impl_getter",
    [
        ("foolbox", FMN._get_foolbox_implementation),
        ("adv_lib", FMN._get_advlib_implementation),
        ("foolbox", PGD._get_foolbox_implementation),
        ("adv_lib", PGD._get_advlib_implementation),
        ("foolbox", DDN._get_foolbox_implementation),
        ("adv_lib", DDN._get_advlib_implementation),
    ],
)
def test_attack_importerror_on_missing_dependency(missing_module, impl_getter):
    expected_msg = f"{missing_module} extra not installed"

    with (
        mock.patch(
            "importlib.util.find_spec",
            side_effect=lambda name, _: (
                None if name == missing_module else mock.DEFAULT
            ),
        ),
        pytest.raises(ImportError, match=expected_msg),
    ):
        impl_getter()
