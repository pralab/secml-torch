"""Wrappers of Nevergrad for evasion attacks."""

import importlib.util

if importlib.util.find_spec("nevergrad", None) is not None:
    from .ng_modular_attack import *  # noqa: F403
