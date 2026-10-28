"""Wrappers of Adversarial Library for evasion attacks."""

import importlib.util

__all__ = []

if importlib.util.find_spec("adv_lib", None) is not None:
    from .advlib_cw import *  # noqa: F403
    from .advlib_ddn import *  # noqa: F403
    from .advlib_deepfool import *  # noqa: F403
    from .advlib_fgsm import *  # noqa: F403
    from .advlib_fmn import *  # noqa: F403
    from .advlib_pgd import *  # noqa: F403

    __all__ = [
        "CWAdvLib",
        "DDNAdvLib",
        "DeepFoolAdvLib",
        "FGSMAdvLib",
        "FMNAdvLib",
        "PGDAdvLib",
    ]
