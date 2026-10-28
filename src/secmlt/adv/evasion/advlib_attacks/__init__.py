"""Wrappers of Adversarial Library for evasion attacks."""

import importlib.util

__all__ = []

if importlib.util.find_spec("adv_lib", None) is not None:
    from .advlib_base import BaseAdvLibEvasionAttack
    from .advlib_cw import CWAdvLib
    from .advlib_ddn import DDNAdvLib
    from .advlib_deepfool import DeepFoolAdvLib
    from .advlib_fgsm import FGSMAdvLib
    from .advlib_fmn import FMNAdvLib
    from .advlib_pgd import PGDAdvLib

    __all__ = [
        "BaseAdvLibEvasionAttack",
        "CWAdvLib",
        "DDNAdvLib",
        "DeepFoolAdvLib",
        "FGSMAdvLib",
        "FMNAdvLib",
        "PGDAdvLib",
    ]
