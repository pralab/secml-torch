"""Implementations of the Projected Gradient Descent evasion attack."""

import importlib

from secmlt.adv.backends import Backends
from secmlt.adv.evasion import BaseFoolboxEvasionAttack
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from secmlt.adv.evasion.foolbox_attacks.foolbox_pgd import PGDFoolbox
from secmlt.adv.evasion.modular_attack import CE_LOSS, ModularEvasionAttackFixedEps
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
from secmlt.trackers.trackers import Tracker


class PGD(BaseEvasionAttackCreator):
    """Creator for the Projected Gradient Descent (PGD) attack."""

    def __new__(
        cls,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        step_size: float,
        random_start: bool = False,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        backend: str = Backends.FOOLBOX,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> BaseEvasionAttack:
        """
        Create the PGD attack.

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
        random_start : bool, optional
            Whether to use a random initialization onto the Lp ball, by
            default False.
        y_target : int | None, optional
            Target label for a targeted attack, None
            for untargeted attack, by default None.
        lb : float, optional
            Lower bound of the input space, by default 0.0.
        ub : float, optional
            Upper bound of the input space, by default 1.0.
        backend : str, optional
            Backend to use to run the attack, by default Backends.FOOLBOX
        trackers : list[Tracker] | None, optional
            Trackers to check various attack metrics (see secmlt.trackers),
            available only for native implementation, by default None.

        Returns
        -------
        BaseEvasionAttack
            PGD attack instance.
        """
        cls.check_backend_available(backend)
        implementation = cls.get_implementation(backend)
        implementation.check_perturbation_model_available(perturbation_model)
        return implementation(
            perturbation_model=perturbation_model,
            epsilon=epsilon,
            num_steps=num_steps,
            step_size=step_size,
            random_start=random_start,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the PGD attack."""
        return [Backends.FOOLBOX, Backends.NATIVE]

    @staticmethod
    def _get_foolbox_implementation() -> type[PGDFoolbox]:
        if importlib.util.find_spec("foolbox", None) is not None:
            return PGDFoolbox
        msg = "Foolbox extra not installed"
        raise ImportError(msg)

    @staticmethod
    def _get_native_implementation() -> type["PGDNative"]:
        return PGDNative


class PGDFoolbox(BaseFoolboxEvasionAttack):
    """Foolbox implementation of the PGD attack."""

    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        step_size: float,
        random_start: bool,
        y_target: int | None = None,
        lb: float = 0.0,
        ub: float = 1.0,
        trackers: list[Tracker] | None = None,
        **kwargs,
    ) -> None:
        """
        Create Foolbox PGD attack.

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

        Raises
        ------
        NotImplementedError
            Raises NotImplementedError if the requested perturbation
            model is not defined for this attack.
        """
        from foolbox.attacks import (
            L1ProjectedGradientDescentAttack,
            L2ProjectedGradientDescentAttack,
            LinfProjectedGradientDescentAttack,
        )

        perturbation_models = {
            LpPerturbationModels.L1: L1ProjectedGradientDescentAttack,
            LpPerturbationModels.L2: L2ProjectedGradientDescentAttack,
            LpPerturbationModels.LINF: LinfProjectedGradientDescentAttack,
        }
        foolbox_attack_cls = perturbation_models.get(perturbation_model)
        if foolbox_attack_cls is None:
            msg = "This perturbation model is not implemented in foolbox."
            raise NotImplementedError(msg)

        foolbox_attack = foolbox_attack_cls(
            abs_stepsize=step_size,
            steps=num_steps,
            random_start=random_start,
        )

        super().__init__(
            foolbox_attack=foolbox_attack,
            epsilon=epsilon,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
        )

    @staticmethod
    def get_perturbation_models() -> set[str]:
        """
        Check the perturbation models implemented for the given attack.

        Returns
        -------
        set[str]
            The set of perturbation models for which the attack is implemented.

        Raises
        ------
        NotImplementedError
            Raises NotImplementedError if not implemented in the inherited class.
        """
        return {
            LpPerturbationModels.L1,
            LpPerturbationModels.L2,
            LpPerturbationModels.LINF,
        }


class PGDNative(ModularEvasionAttackFixedEps):
    """Native implementation of the Projected Gradient Descent attack."""

    def __init__(
        self,
        perturbation_model: str,
        epsilon: float,
        num_steps: int,
        step_size: float,
        random_start: bool,
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
            loss_function=CE_LOSS,
            optimizer_cls=OptimizerFactory.create_sgd(step_size),
            manipulation_function=manipulation_function,
            gradient_processing=gradient_processing,
            initializer=initializer,
            trackers=trackers,
        )
