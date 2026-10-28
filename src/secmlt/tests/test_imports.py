from unittest import mock

import pytest
import secmlt.trackers as trackers
from secmlt.adv.evasion.ddn import DDN
from secmlt.adv.evasion.fmn import FMN
from secmlt.adv.evasion.pgd import PGD
from secmlt.trackers import ImageGradientsTracker, ImageSampleTracker, Tracker


def test_tracker_package_exports():
    expected_exports = {
        "ImageGradientsTracker",
        "ImageSampleTracker",
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
            side_effect=lambda name, _: None
            if name == missing_module
            else mock.DEFAULT,
        ),
        pytest.raises(ImportError, match=expected_msg),
    ):
        impl_getter()
