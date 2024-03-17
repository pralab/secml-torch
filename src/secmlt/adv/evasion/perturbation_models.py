"""Implementation of perturbation models for perturbations of adversarial examples."""

from typing import ClassVar


class LpPerturbationModels:
    """Lp perturbation models."""

    L0 = "l0"
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"
    pert_models: ClassVar[dict[str, float]] = {L0: 0, L1: 1, L2: 2, LINF: float("inf")}

    @classmethod
    def is_perturbation_model_available(cls, perturbation_model: str) -> bool:
        """
        Check availability of the perturbation model requested.

        Parameters
        ----------
        perturbation_model : str
            A perturbation model as a string.

        Returns
        -------
        bool
            True if the perturbation model is found in PerturbationModels.pert_models.
        """
        return perturbation_model in (cls.pert_models)

    @classmethod
    def get_p(cls, perturbation_model: str) -> float:
        """
        Get the float representation of p from the given string.

        Parameters
        ----------
        perturbation_model : str
            One of the strings defined in PerturbationModels.pert_models.

        Returns
        -------
        float
            The float representation of p, to use. e.g., in torch.norm(p=...).

        Raises
        ------
        ValueError
            Raises ValueError if the norm given is not in PerturbationModels.pert_models
        """
        if cls.is_perturbation_model_available(perturbation_model):
            return cls.pert_models[perturbation_model]
        msg = "Perturbation model not implemented"
        raise ValueError(msg)
