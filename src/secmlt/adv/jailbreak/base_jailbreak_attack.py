"""Base classes for implementing attacks and wrapping backends."""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from secmlt.models.base_language_model import BaseLanguageModel


Objective = Callable[..., Any] | None


class BaseJailbreakAttack:
    """Base class for jailbreak attacks."""

    def __call__(
        self,
        model: BaseLanguageModel,
        behaviors: list[dict[str, Any]],
        objective: Objective = None,
    ) -> list[tuple[str, dict[str, Any]]]:
        """
        Compute the jailbreak attack against the model, using the input behaviors.

        Parameters
        ----------
        model : BaseLanguageModel
            Victim language model to attack.
        behaviors : list of dict
            List of behavior specifications. Each element represents
            a single behavior to jailbreak.
        objective : callable or None, optional
            Optional objective used by the attack (e.g., loss, judge, score).
            If not required by the attack, it can be None.

        Returns
        -------
        list of tuples
            List of (adv_prompt, logs) pairs, one for each behavior.
        """
        results: list[tuple[str, dict[str, Any]]] = []
        for behavior in behaviors:
            adv_prompt, logs = self._run(model, behavior, objective)
            results.append((adv_prompt, logs))
        return results

    @abstractmethod
    def _run(
        self,
        model: BaseLanguageModel,
        behavior: dict[str, Any],
        objective: Objective,
    ) -> tuple[str, dict[str, Any]]:
        """
        Run the jailbreak attack on a single behavior.

        Parameters
        ----------
        model : BaseLanguageModel
            Victim language model.
        behavior : dict
            Behavior specification to jailbreak.
        objective : callable or None
            Optional objective used internally by the attack.

        Returns
        -------
        adv_prompt : str
            Generated adversarial prompt for the behavior.
        logs : dict
            Dictionary containing attack-specific logs and metadata.
        """
        raise NotImplementedError

    @abstractmethod
    def is_jailbreak(
        self,
        model: BaseLanguageModel,
        behavior: dict[str, Any],
        adv_prompt: str,
        logs: dict[str, Any],
        objective: Objective,
    ) -> bool:
        """
        Check whether the generated adversarial prompt is a successful jailbreak.

        Parameters
        ----------
        model : BaseLanguageModel
            Victim language model.
        behavior : dict
            Original behavior specification.
        adv_prompt : str
            Generated adversarial prompt.
        logs : dict
            Logs produced during the attack.
        objective : callable or None
            Optional objective used by the attack.

        Returns
        -------
        bool
            True if the attack is considered successful, False otherwise.
        """
        raise NotImplementedError
