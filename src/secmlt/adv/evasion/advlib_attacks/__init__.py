"""Wrappers of Adversarial Library for evasion attacks."""

import importlib.metadata
import importlib.util


def _adv_lib_gte(major: int, minor: int, patch: int) -> bool:
    try:
        version_str = importlib.metadata.version("adv-lib")
    except importlib.metadata.PackageNotFoundError:
        return False
    else:
        parts = tuple(int(x) for x in version_str.split(".")[:3])
        return parts >= (major, minor, patch)


if importlib.util.find_spec("adv_lib", None) is not None:
    from .advlib_cw import *  # noqa: F403
    from .advlib_ddn import *  # noqa: F403
    from .advlib_fgsm import *  # noqa: F403
    from .advlib_fmn import *  # noqa: F403
    from .advlib_pgd import *  # noqa: F403

    if _adv_lib_gte(0, 2, 3):
        from .advlib_deepfool import *  # noqa: F403
