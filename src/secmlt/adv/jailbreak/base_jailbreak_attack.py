"""Base classes for implementing attacks and wrapping backends."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from collections.abc import Callable

    from secmlt.models.base_language_model import BaseLanguageModel

    IS_JAILBREAK_TYPE = Callable[
        [BaseLanguageModel, dict[str, Any], str, dict[str, Any]],
        bool,
    ]


class BaseJailbreakAttack:
    """Base class for jailbreak attacks."""

    def __init__(
        self,
        is_jailbreak: IS_JAILBREAK_TYPE | None = None,
    ) -> None:
        """
        Create a jailbreak attack.

        Parameters
        ----------
        is_jailbreak : callable or None, optional
            Success criterion used by the attack, by default None.
        """
        self._is_jailbreak = is_jailbreak

    def __call__(
        self,
        model: BaseLanguageModel,
        behaviors: list[dict[str, Any]],
        objectives: list[str] | None = None,
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
        objectives : list of str or None, optional
            Optional objectives used by the attack, one for each behavior
            (e.g., target prefixes such as "Sure, here is"), by default None.

        Returns
        -------
        list of tuples
            List of (adv_prompt, logs) pairs, one for each behavior.
        """
        if objectives is not None and len(objectives) != len(behaviors):
            msg = "Objectives and behaviors must have the same length."
            raise ValueError(msg)

        results = []
        for i, behavior in enumerate(behaviors):
            objective = None if objectives is None else objectives[i]
            results.append(self._run(model, behavior, objective))
        return results

    @abstractmethod
    def _run(
        self,
        model: BaseLanguageModel,
        behavior: dict[str, Any],
        objective: str | None = None,
    ) -> tuple[str, dict[str, Any]]:
        """
        Run the jailbreak attack on a single behavior.

        Parameters
        ----------
        model : BaseLanguageModel
            Victim language model.
        behavior : dict
            Behavior specification to jailbreak.
        objective : str or None, optional
            Optional objective used by the attack for the behavior, by default None.

        Returns
        -------
        adv_prompt : str
            Generated adversarial prompt.
        logs : dict
            Attack logs.
        """
        raise NotImplementedError

    def is_jailbreak(
        self,
        model: BaseLanguageModel,
        behavior: dict[str, Any],
        adv_prompt: str,
        logs: dict[str, Any],
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

        Returns
        -------
        bool
            True if the attack is considered successful, False otherwise.
        """
        if self._is_jailbreak is None:
            msg = "Jailbreak success criterion not available."
            raise NotImplementedError(msg)
        return self._is_jailbreak(model, behavior, adv_prompt, logs)
