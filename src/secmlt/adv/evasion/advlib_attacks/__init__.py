"""Wrappers of Adversarial Library for evasion attacks."""

import importlib

if importlib.util.find_spec("adv_lib", None) is not None:
    from .advlib_pgd import *  # noqa: F403
