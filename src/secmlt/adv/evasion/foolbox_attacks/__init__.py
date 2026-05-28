"""Wrappers of Foolbox library for evasion attacks."""

import importlib.util

if importlib.util.find_spec("foolbox", None) is not None:
    from .foolbox_cw import *  # noqa: F403
    from .foolbox_ddn import *  # noqa: F403
    from .foolbox_deepfool import *  # noqa: F403
    from .foolbox_fgsm import *  # noqa: F403
    from .foolbox_pgd import *  # noqa: F403
