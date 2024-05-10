"""Evasion attack functionalities."""

import importlib

if importlib.util.find_spec("foolbox", None) is not None:
    from .foolbox_attacks import *  # noqa: F403

if importlib.util.find_spec("adv_lib", None) is not None:
    from .advlib_attacks import *  # noqa: F403
