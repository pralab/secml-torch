from unittest import mock

import pytest
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.fmn import FMN
from secmlt.adv.evasion.pgd import PGD


@pytest.mark.parametrize(
    "backend",
    [Backends.FOOLBOX, Backends.ADVLIB],
)
@pytest.mark.parametrize(
    ["attack", "attack_args"],
    [
        (
            PGD,
            {
                "perturbation_model": "linf",
                "num_steps": 10,
                "step_size": 0.1,
                "epsilon": 0.5,
            },
        ),
        (
            FMN,
            {
                "perturbation_model": "l2",
                "num_steps": 10,
                "step_size": 0.1,
            },
        ),
    ],
)
def test_imports(backend, attack, attack_args):
    missing_module = "foolbox" if backend == Backends.FOOLBOX else "adv_lib"

    with (
        mock.patch(
            "importlib.util.find_spec",
            side_effect=lambda name, _: None
            if name == missing_module
            else mock.DEFAULT,
        ),
        pytest.raises(ImportError),
    ):
        attack(backend=backend, **attack_args)
