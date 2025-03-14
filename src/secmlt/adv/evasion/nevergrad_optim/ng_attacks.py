"""Genetic algorithm to compute evasion attacks with nevergrad."""
from secmlt.adv.evasion import LpPerturbationModels
from secmlt.adv.evasion.modular_attack import CE_LOSS
from secmlt.adv.evasion.nevergrad_optim import NgModularEvasionAttackFixedEps
from secmlt.adv.evasion.nevergrad_optim.ng_initializer import NevergradInitializer
from secmlt.adv.evasion.nevergrad_optim.ng_optimizer_factory import (
    NevergradOptimizerFactory,
)
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import (
    ClipConstraint,
    L1Constraint,
    L2Constraint,
    LInfConstraint,
)
from secmlt.optimization.initializer import Initializer, RandomLpInitializer
from secmlt.trackers import Tracker


class NevergradGeneticAlgorithm(NgModularEvasionAttackFixedEps):
    """Create a genetic algorithm with nevergrad."""

    def __init__(
            self,
            perturbation_model: LpPerturbationModels,
            epsilon: float,
            num_steps: int,
            budget: int,
            population_size: int,
            random_start: bool,
            y_target: int | None = None,
            lb: float = 0.0,
            ub: float = 1.0,
            trackers: list[Tracker] | None = None,
    ) -> None:
        """
        Create Nevergrad Genetic Algorithm attack.

        Parameters
        ----------
        perturbation_model : str
            Perturbation model for the attack. Available: 1, 2, inf.
        epsilon : float
            Radius of the constraint for the Lp ball.
        num_steps : int
            Number of iterations for the attack.
        budget : int
            Amount of query budget
        random_start : bool
            Whether to use a random initialization onto the Lp ball.
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
        if perturbation_model not in perturbation_models:
            msg = f"Perturbation model {perturbation_model} not yet implemented."
            raise NotImplementedError(msg)
        if random_start:
            initializer = RandomLpInitializer(
                perturbation_model=perturbation_model,
                radius=epsilon,
            )
        else:
            initializer = Initializer()
        initializer = NevergradInitializer(initializer, lb=lb, ub=ub)
        self.epsilon = epsilon
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
            loss_function=CE_LOSS,
            optimizer_cls=NevergradOptimizerFactory.create_ga(population_size=population_size),
            manipulation_function=manipulation_function,
            initializer=initializer,
            trackers=trackers,
            budget=budget
        )

    @classmethod
    def get_perturbation_models(cls) -> set[str]:
        """
        Check if a given perturbation model is implemented.

        Returns
        -------
        set[str]
            Set of perturbation models available for this attack.
        """
        return {
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }
