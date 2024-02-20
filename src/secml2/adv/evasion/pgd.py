from typing import Optional, List

from secml2.adv.backends import Backends
from secml2.adv.evasion.base_evasion_attack import (
    BaseEvasionAttackCreator,
)
from secml2.adv.evasion.composite_attack import CompositeEvasionAttack, CE_LOSS
from secml2.adv.evasion.perturbation_models import PerturbationModels
from secml2.manipulations.manipulation import AdditiveManipulation
from secml2.optimization.constraints import (
    ClipConstraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
)
from secml2.optimization.gradient_processing import LinearProjectionGradientProcessing
from secml2.optimization.initializer import Initializer
from secml2.optimization.optimizer_factory import OptimizerFactory


class PGD(BaseEvasionAttackCreator):
    def __new__(
        cls,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        step_size: float,
        random_start: bool,
        y_target: Optional[int] = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        **kwargs
    ):
        cls.check_perturbation_model_available(perturbation_model)
        implementation = cls.get_implementation(backend)
        return implementation(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            random_start=random_start,
            y_target=y_target,
            lb=lb,
            ub=ub,
            **kwargs
        )

    @staticmethod
    def _get_foolbox_implementation():
        try:
            from .foolbox_attacks.foolbox_pgd import PGDFoolbox
        except ImportError:
            raise ImportError("Foolbox extra not installed")
        return PGDFoolbox

    @staticmethod
    def get_native_implementation():
        return PGDNative


class PGDNative(CompositeEvasionAttack):
    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        step_size: float,
        random_start: bool,
        y_target: Optional[int] = None,
        lb: float = 0.0,
        ub: float = 1.0,
        **kwargs
    ) -> None:
        perturbation_models = {
            PerturbationModels.L1: L1Constraint,
            PerturbationModels.L2: L2Constraint,
            PerturbationModels.LINF: LInfConstraint,
        }
        initializer = Initializer()
        if random_start:
            raise NotImplementedError("Random start in LP ball not yet implemented.")
        self.epsilon = epsilon
        gradient_processing = LinearProjectionGradientProcessing(perturbation_model)
        perturbation_constraints = [
            perturbation_models[perturbation_model](radius=self.epsilon)
        ]
        domain_constraints = [ClipConstraint(lb=lb, ub=ub)]
        manipulation_function = AdditiveManipulation(
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
        )
        super().__init__(
            y_target=y_target,
            num_steps=num_steps,
            step_size=step_size,
            loss_function=CE_LOSS,
            optimizer_cls=OptimizerFactory.create_sgd(step_size),
            manipulation_function=manipulation_function,
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
            gradient_processing=gradient_processing,
            initializer=initializer,
        )
