"""Evasion attack functionalities."""

import importlib.util
import sys
import types
from importlib import import_module

__all__ = [
    "CW",
    "DDN",
    "FGSM",
    "FMN",
    "PGD",
    "VAT",
    "AdditiveNoise",
    "BaseEvasionAttack",
    "BaseEvasionAttackCreator",
    "BoundaryAttack",
    "ContrastReduction",
    "DDNNative",
    "DeepFool",
    "FMNNative",
    "GaussianBlur",
    "HopSkipJump",
    "LpPerturbationModels",
    "PGDNative",
    "SaltAndPepperNoise",
    "SpatialAttack",
]

_EXPORTS = {
    name: (f".{module}", name)
    for module, names in {
        "additive_noise": ["AdditiveNoise"],
        "base_evasion_attack": ["BaseEvasionAttack", "BaseEvasionAttackCreator"],
        "boundary_attack": ["BoundaryAttack"],
        "contrast_reduction": ["ContrastReduction"],
        "cw": ["CW"],
        "ddn": ["DDN", "DDNNative"],
        "deepfool": ["DeepFool"],
        "fgsm": ["FGSM"],
        "fmn": ["FMN", "FMNNative"],
        "gaussian_blur": ["GaussianBlur"],
        "hopskipjump": ["HopSkipJump"],
        "perturbation_models": ["LpPerturbationModels"],
        "pgd": ["PGD", "PGDNative"],
        "saltandpepper": ["SaltAndPepperNoise"],
        "spatial_attack": ["SpatialAttack"],
        "vat": ["VAT"],
    }.items()
    for name in names
}


def __getattr__(name: str) -> object:
    """Load a public attack symbol on first access."""
    try:
        module, attribute = _EXPORTS[name]
    except KeyError:
        message = f"module {__name__!r} has no attribute {name!r}"
        raise AttributeError(message) from None
    if module == ".advlib_attacks":
        _ensure_visdom_stub()
    return getattr(import_module(module, __name__), attribute)


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
    _foolbox_exports = [
        "AdditiveNoiseFoolbox",
        "BaseFoolboxEvasionAttack",
        "BoundaryAttackFoolbox",
        "ContrastReductionFoolbox",
        "CWFoolbox",
        "DDNFoolbox",
        "DeepFoolFoolbox",
        "FGSMFoolbox",
        "FMNFoolbox",
        "GaussianBlurFoolbox",
        "HopSkipJumpFoolbox",
        "PGDFoolbox",
        "SaltAndPepperNoiseFoolbox",
        "SpatialAttackFoolbox",
        "VATFoolbox",
    ]

    _EXPORTS.update(
        {name: (".foolbox_attacks", name) for name in _foolbox_exports},
    )
    __all__ += _foolbox_exports  # noqa: PLE0605

if importlib.util.find_spec("adv_lib", None) is not None:  # pragma: no cover
    _advlib_exports = [
        "BaseAdvLibEvasionAttack",
        "CWAdvLib",
        "DDNAdvLib",
        "DeepFoolAdvLib",
        "FGSMAdvLib",
        "FMNAdvLib",
        "PGDAdvLib",
    ]

    _EXPORTS.update({name: (".advlib_attacks", name) for name in _advlib_exports})
    __all__ += _advlib_exports  # noqa: PLE0605
