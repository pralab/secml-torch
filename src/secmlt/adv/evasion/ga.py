"""Implementation of Genetic Algorithm attack."""
import importlib
from typing import Optional

from secmlt.adv.backends import Backends
from secmlt.adv.evasion.base_evasion_attack import (
    BaseEvasionAttack,
    BaseEvasionAttackCreator,
)
from secmlt.trackers import Tracker


class GeneticAlgorithm(BaseEvasionAttackCreator):
    """Implementation of Genetic Algorithm."""

    def __new__(
            cls,
            perturbation_model: str,
            epsilon: float,
            num_steps: int,
            budget: Optional[int] = None,
            population_size: int = 10,
            random_start: bool = False,
            y_target: int | None = None,
            lb: float = 0.0,
            ub: float = 1.0,
            backend: str = Backends.NEVERGRAD,
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
            Maximum number of iterations for the attack.
        budget : int, optional
            Maximum number of queries.
            Default None means that num_steps will be set.
        population_size: int, optional
            Number of variants created at each round of optimization.
            Default 10.
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
            budget=budget,
            population_size=population_size,
            random_start=random_start,
            y_target=y_target,
            lb=lb,
            ub=ub,
            trackers=trackers,
            **kwargs,
        )

    @staticmethod
    def get_backends() -> list[str]:
        """Get available implementations for the GA attack."""
        return [Backends.NEVERGRAD]

    @classmethod
    def get_implementation(cls, backend: str) -> "BaseEvasionAttack":
        """
        Get the implementation of the attack with the given backend.

        Parameters
        ----------
        backend : str
            The backend for the attack. See secmlt.adv.backends for
            available backends.

        Returns
        -------
        BaseEvasionAttack
            Attack implementation.
        """
        implementations = {
            Backends.NEVERGRAD: cls.get_foolbox_implementation,
        }
        cls.check_backend_available(backend)
        return implementations[backend]()

    @classmethod
    def get_nevergrad_implementation(cls) -> "BaseEvasionAttack":
        """
        Get the Foolbox implementation of the attack.

        Returns
        -------
        BaseEvasionAttack
            Foolbox implementation of the attack.

        Raises
        ------
        ImportError
            Raises ImportError if Foolbox extra is not installed.
        """
        if importlib.util.find_spec("nevergrad", None) is not None:
            return cls._get_nevergrad_implementation()
        msg = "Nevergrad extra not installed."
        raise ImportError(msg)

    @staticmethod
    def _get_nevergrad_implementation() -> type["PGDFoolbox"]:  # noqa: F821
        if importlib.util.find_spec("nevergrad", None) is not None:
            from secmlt.adv.evasion.nevergrad_optim.ng_attacks import (
                NevergradGeneticAlgorithm,
            )

            return NevergradGeneticAlgorithm
        msg = "Nevergrad extra not installed"
        raise ImportError(msg)
