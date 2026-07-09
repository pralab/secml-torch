"""Ensemble PGD attack module for adversarial robustness."""
import torch
from secmlt.adv.evasion.ensemble.loss.avg_loss import AvgEnsembleLoss
from secmlt.adv.evasion.modular_attacks.modular_attack_fixed_eps import (
    ModularEvasionAttackFixedEps,
)
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import (
    ClipConstraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
)
from secmlt.optimization.gradient_processing import LinearProjectionGradientProcessing
from secmlt.optimization.initializer import Initializer, RandomLpInitializer
from secmlt.optimization.optimizer_factory import OptimizerFactory
from secmlt.optimization.scheduler_factory import LRSchedulerFactory
from secmlt.trackers.trackers import Tracker


class EnsemblePGD(ModularEvasionAttackFixedEps):
    """Projected Gradient Descent attack working on an ensemble of models."""

    def __init__(
            self,
            perturbation_model: str,
            epsilon: float,
            num_steps: int,
            step_size: float,
            random_start: bool,
            loss_function: torch.nn.Module = None,
            y_target: int | None = None,
            lb: float = 0.0,
            ub: float = 1.0,
            trackers: list[Tracker] | None = None,
            **kwargs,
    ) -> None:
        """
        Create Native PGD attack.

        Parameters
        ----------
        perturbation_model : str
            Perturbation model for the attack. Available: 1, 2, inf.
        epsilon : float
            Radius of the constraint for the Lp ball.
        num_steps : int
            Number of iterations for the attack.
        step_size : float
            Attack step size.
        random_start : bool
            Whether to use a random initialization onto the Lp ball.
        loss_function : torch.nn.Module, optional
            Loss function to minimize. It must handle ensemble outputs of shape
            (n_models, batch_size, *model_output).
            Default: AvgEnsembleLoss with CrossEntropy, which computes the
            cross-entropy loss for each model and averages them.
        y_target : int | None, optional
            Target label for a targeted attack, None
            for untargeted attack, by default None.
        lb : float, optional
            Lower bound of the input space, by default 0.0.
        ub : float, optional
            Upper bound of the input space, by default 1.0.
        trackers : list[Tracker] | None, optional
            Trackers to check various attack metrics (see secmlt.trackers),
            available only for native implementation, by default None.
        """
        perturbation_models = {
            LpPerturbationModels.L1: L1Constraint,
            LpPerturbationModels.L2: L2Constraint,
            LpPerturbationModels.LINF: LInfConstraint,
        }

        if random_start:
            initializer = RandomLpInitializer(
                perturbation_model=perturbation_model,
                radius=epsilon,
            )
        else:
            initializer = Initializer()
        if loss_function is None:
            loss_function = AvgEnsembleLoss()
        self.epsilon = epsilon
        gradient_processing = LinearProjectionGradientProcessing(perturbation_model)
        perturbation_constraints = [
            perturbation_models[perturbation_model](radius=self.epsilon),
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
            loss_function=loss_function,
            optimizer_cls=OptimizerFactory.create_sgd(step_size),
            scheduler_cls=LRSchedulerFactory.create_no_scheduler(),
            manipulation_function=manipulation_function,
            gradient_processing=gradient_processing,
            initializer=initializer,
            trackers=trackers,
        )
