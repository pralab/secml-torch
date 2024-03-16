"""Evasion attack functionalities."""

import importlib

if importlib.util.find_spec("foolbox", None) is not None:
    from .foolbox_attacks import *  # noqa: F403
