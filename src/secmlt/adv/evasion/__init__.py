"""Evasion attack functionalities."""

import importlib.util
import sys
import types


def _ensure_visdom_stub() -> None:
    """Provide a minimal visdom stub if visdom is unavailable."""
    if importlib.util.find_spec("visdom", None) is not None:  # pragma: no cover
        return
    if "visdom" in sys.modules:  # pragma: no cover
        return

    visdom_stub = types.ModuleType("visdom")  # pragma: no cover

    class Visdom:  # pragma: no cover
        def __init__(self, *args, **kwargs) -> None:  # pragma: no cover
            pass  # pragma: no cover

    visdom_stub.Visdom = Visdom  # pragma: no cover
    sys.modules["visdom"] = visdom_stub  # pragma: no cover

if importlib.util.find_spec("foolbox", None) is not None:  # pragma: no cover
    from .foolbox_attacks import *  # noqa: F403

if importlib.util.find_spec("adv_lib", None) is not None:  # pragma: no cover
    _ensure_visdom_stub()
    from .advlib_attacks import *  # noqa: F403
