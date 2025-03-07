"""Factory for creating nevergrad optimization algorithms."""

from functools import partial
from typing import ClassVar, Union

from nevergrad.optimization.base import ConfiguredOptimizer
from nevergrad.optimization.differentialevolution import DifferentialEvolution

GA = "ga"


class NevergradOptimizerFactory:
    """Factory for creating optimization algorithms using nevergrad."""

    NG_OPTIMIZERS: ClassVar[dict[str, ConfiguredOptimizer]] = {
        GA: DifferentialEvolution,
    }

    @staticmethod
    def create_from_name(
            optimizer_name: str,
            **kwargs,
    ) -> ConfiguredOptimizer:
        """
        Create a Nevergrad optimization algorithm by name.

        Available: "ga" -> Differential Evolution.

        Parameters
        ----------
        optimizer_name : str
            The name of the optjmizer to create.
            Available: "ga"

        Returns
        -------
        Optimizer
            the created optimizer.

        Raises
        ------
        ValueError
            Raises ValueError when the requested optimizer is not in the list
            of implemented optimizers.
        """
        return NevergradOptimizerFactory.create(optimizer_name, **kwargs)

    @staticmethod
    def create(optim_cls: Union[str, ClassVar[ConfiguredOptimizer]],
               **optimizer_args) -> ConfiguredOptimizer:
        """
        Create a Nevergrad optimization algorithm.

        Use its name among the available or the class itself.
        Check the nevergrad documentation for more options
        (https://facebookresearch.github.io/nevergrad/optimizers_ref.html).

        Parameters
        ----------
        optim_cls : Union[str, ClassVar[ConfiguredOptimizer]]
            a string or a nevergrad optimizer class to instantiate

        Returns
        -------
        ConfiguredOptimizer :
            the created optimizer.
        """
        if not isinstance(optim_cls, str):
            return partial(optim_cls, **optimizer_args)()
        if optim_cls == "ga":
            return NevergradOptimizerFactory.create_ga(**optimizer_args)
        msg = f"Optimizer not found. Use one of: \
                    {list(NevergradOptimizerFactory.NG_OPTIMIZERS.keys())}"
        raise ValueError(msg)

    @staticmethod
    def create_ga(population_size: int = 10, crossover: str = "twopoints",
                  **kwargs) -> ConfiguredOptimizer:
        """
        Create a Differential Evolution (genetic algorithm) using Nevergrad as backend.

        Check the nevergrad documentation for more options
        (https://facebookresearch.github.io/nevergrad/optimizers_ref.html#nevergrad.families.DifferentialEvolution).

        Parameters
        ----------
        population_size : int = 10
            the amount of sampling applied to determine the
            direction to take while optimizing
        crossover : str = "twopoints"
            define the merging strategy for the crossover.
            Default is "twopoints".

        Returns
        -------
        DifferentialEvolution :
            the created optimizer.

        """
        return DifferentialEvolution(popsize=population_size,
                                     crossover=crossover,
                                     **kwargs)
