"""Wrappers of Foolbox library for evasion attacks."""

import importlib.util

__all__ = []

if importlib.util.find_spec("foolbox", None) is not None:
    from .foolbox_additive_noise import AdditiveNoiseFoolbox
    from .foolbox_base import BaseFoolboxEvasionAttack
    from .foolbox_boundary import BoundaryAttackFoolbox
    from .foolbox_contrast_reduction import ContrastReductionFoolbox
    from .foolbox_cw import CWFoolbox
    from .foolbox_ddn import DDNFoolbox
    from .foolbox_deepfool import DeepFoolFoolbox
    from .foolbox_fgsm import FGSMFoolbox
    from .foolbox_fmn import FMNFoolbox
    from .foolbox_gaussian_blur import GaussianBlurFoolbox
    from .foolbox_hopskipjump import HopSkipJumpFoolbox
    from .foolbox_pgd import PGDFoolbox
    from .foolbox_saltandpepper import SaltAndPepperNoiseFoolbox
    from .foolbox_spatial import SpatialAttackFoolbox
    from .foolbox_vat import VATFoolbox

    __all__ = [
        "AdditiveNoiseFoolbox",
        "BaseFoolboxEvasionAttack",
        "BoundaryAttackFoolbox",
        "CWFoolbox",
        "ContrastReductionFoolbox",
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
