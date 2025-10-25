"""Basic wrapper for generic language model."""

from abc import ABC, abstractmethod
from typing import Union

import torch


class BaseLanguageModel(ABC):
    """Abstract base class defining the common interface for language models."""

    @abstractmethod
    def encode(self, text: Union[str, list[str]], **kwargs) -> torch.LongTensor:
        """
        Convert input text into token IDs.

        Parameters
        ----------
        text : str or list of str
            Input text(s) to tokenize.

        Returns
        -------
        torch.LongTensor
            Token IDs tensor.
        """
        ...

    @abstractmethod
    def decode(self, ids: torch.LongTensor, **kwargs) -> Union[str, list[str]]:
        """
        Convert token IDs back into text.

        Parameters
        ----------
        ids : torch.LongTensor
            Token IDs tensor.

        Returns
        -------
        str or list of str
            Decoded text(s).
        """
        ...

    @abstractmethod
    def predict(self, input_ids: torch.LongTensor, **kwargs) -> torch.Tensor:
        """
        Predict logits for the next token given an input sequence.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs.

        Returns
        -------
        torch.Tensor
            Logits for the next token (shape [batch, vocab_size]).
        """
        ...

    @abstractmethod
    def generate(self, prompts: list[str], **kwargs) -> list[str]:
        """
        Generate text continuations from given prompts.

        Parameters
        ----------
        prompts : list of str
            List of prompt strings.

        Returns
        -------
        list of str
            Generated text outputs.
        """
        ...

    @abstractmethod
    def logprobs(
        self, prompts: list[str], targets: list[str], **kwargs
    ) -> list[torch.Tensor]:
        """
        Compute log-probabilities for each token in the target continuations.

        Parameters
        ----------
        prompts : list of str
            Conditioning prompts.
        targets : list of str
            Target continuations.

        Returns
        -------
        list of torch.Tensor
            List of log-probability tensors, one per sample in the batch.
            Each tensor has shape [target_len_i].
        """
        ...

    @abstractmethod
    def hidden_states(
        self, input_ids: torch.LongTensor, **kwargs
    ) -> list[torch.Tensor]:
        """
        Return hidden states of the model for given input.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs.

        Returns
        -------
        list of torch.Tensor
            Hidden representations per layer.
        """
        ...

    def __call__(self, input_ids: torch.LongTensor, **kwargs) -> torch.Tensor:
        """
        Shortcut for self.predict().

        Parameters
        ----------
        input_ids : torch.LongTensor
            Input token IDs.

        Returns
        -------
        torch.Tensor
            Logits for the next token.
        """
        return self.predict(input_ids, **kwargs)
