"""Wrappers of Foolbox library for evasion attacks."""

import importlib

if importlib.util.find_spec("foolbox", None) is not None:
    from .foolbox_pgd import *  # noqa: F403
