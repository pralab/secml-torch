from typing import Optional, List

import torch
from torch.optim import Adam

from secml2.adv.evasion.composite_attack import CompositeEvasionAttack, CE_LOSS, SGD
from secml2.manipulations.manipulation import AdditiveManipulation
from secml2.optimization.initializer import Initializer
from secml2.optimization.gradient_processing import LinearProjectionGradientProcessing
from secml2.adv.evasion.foolbox import BaseFoolboxEvasionAttack

from secml2.adv.evasion.perturbation_models import PerturbationModels
from secml2.adv.backends import Backends
from secml2.adv.evasion.base_evasion_attack import (
    BaseEvasionAttackCreator,
)

from foolbox.attacks.projected_gradient_descent import (
    L1ProjectedGradientDescentAttack,
    L2ProjectedGradientDescentAttack,
    LinfProjectedGradientDescentAttack,
)

from secml2.optimization.constraints import (
    ClipConstraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
    Constraint,
)


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
    def get_foolbox_implementation():
        return PGDFoolbox

    @staticmethod
    def get_native_implementation():
        return PGDNative


class PGDFoolbox(BaseFoolboxEvasionAttack):
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
            PerturbationModels.L1: L1ProjectedGradientDescentAttack,
            PerturbationModels.L2: L2ProjectedGradientDescentAttack,
            PerturbationModels.LINF: LinfProjectedGradientDescentAttack,
        }
        foolbox_attack_cls = perturbation_models.get(perturbation_model, None)
        if foolbox_attack_cls is None:
            raise NotImplementedError(
                "This threat model is not implemented in foolbox."
            )

        foolbox_attack = foolbox_attack_cls(
            abs_stepsize=step_size, steps=num_steps, random_start=random_start
        )

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=epsilon,
            y_target=y_target,
            lb=lb,
            ub=ub,
        )


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
        perturbation_constraints = [perturbation_models[perturbation_model]]
        domain_constraints = [ClipConstraint(lb=lb, ub=ub)]
        manipulation_function = AdditiveManipulation()
        super().__init__(
            y_target=y_target,
            num_steps=num_steps,
            step_size=step_size,
            loss_function=CE_LOSS,
            optimizer_cls=SGD,
            manipulation_function=manipulation_function,
            domain_constraints=domain_constraints,
            perturbation_constraints=perturbation_constraints,
            gradient_processing=gradient_processing,
            initializer=initializer,
        )

    def init_perturbation_constraints(self) -> List[Constraint]:
        return [p(radius=self.epsilon) for p in self.perturbation_constraints]
