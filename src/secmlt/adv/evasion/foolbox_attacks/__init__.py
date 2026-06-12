"""Wrappers of Foolbox library for evasion attacks."""

import importlib.util

if importlib.util.find_spec("foolbox", None) is not None:
    from .foolbox_additive_noise import *  # noqa: F403
    from .foolbox_boundary import *  # noqa: F403
    from .foolbox_contrast_reduction import *  # noqa: F403
    from .foolbox_cw import *  # noqa: F403
    from .foolbox_ddn import *  # noqa: F403
    from .foolbox_deepfool import *  # noqa: F403
    from .foolbox_fgsm import *  # noqa: F403
    from .foolbox_gaussian_blur import *  # noqa: F403
    from .foolbox_hopskipjump import *  # noqa: F403
    from .foolbox_pgd import *  # noqa: F403
    from .foolbox_saltandpepper import *  # noqa: F403
    from .foolbox_spatial import *  # noqa: F403
    from .foolbox_vat import *  # noqa: F403
