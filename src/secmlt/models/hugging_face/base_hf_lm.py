"""Wrapper for Hugging Face causal language models."""
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from secmlt.models.base_language_model import BaseLanguageModel


class HFCausalLM(BaseLanguageModel):
    """Wrapper for Hugging Face causal language models."""

    def __init__(
        self,
        model_path: str,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
        tokenizer_kwargs: Optional[dict] = None,
        model_kwargs: Optional[dict] = None,
    ) -> None:
        """
        Create a wrapped Hugging Face causal language model.

        Parameters
        ----------
        model_path : str
            Model name or local path.
        device : torch.device, optional
            Device where the model is loaded. Defaults to GPU if available.
        dtype : torch.dtype, optional
            Model precision. Defaults to model default dtype.
        tokenizer_kwargs : dict, optional
            Extra arguments for AutoTokenizer.from_pretrained(). Defaults to None.
        model_kwargs : dict, optional
            Extra arguments for AutoModelForCausalLM.from_pretrained(). Defaults to None.
        """
        self._model_path = model_path
        self._device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self._dtype = dtype
        self._tokenizer = None
        self._model = None
        self._tokenizer_kwargs = tokenizer_kwargs or {}
        self._model_kwargs = model_kwargs or {}

        self.load()

    def load(self) -> None:
        """
        
        Load model and tokenizer if not already loaded.
        
        """
        if self._model is not None and self._tokenizer is not None:
            return

        self._tokenizer = AutoTokenizer.from_pretrained(self._model_path, **self._tokenizer_kwargs)
        self._model = AutoModelForCausalLM.from_pretrained(self._model_path, **self._model_kwargs).eval()

        if self._tokenizer.pad_token_id is None and self._tokenizer.eos_token_id is not None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        self._dtype = self._dtype or getattr(self._model, "dtype", torch.float32)
        self._model.to(self._device, dtype=self._dtype)

    @property
    def model(self) -> AutoModelForCausalLM:
        """
        Get the wrapped Hugging Face model.

        Returns
        -------
        AutoModelForCausalLM
            Wrapped Hugging Face model.
        """
        return self._model

    @property
    def tokenizer(self) -> AutoTokenizer:
        """
        Get the wrapped Hugging Face tokenizer.

        Returns
        -------
        AutoTokenizer
            Wrapped Hugging Face tokenizer.
        """
        return self._tokenizer

    @torch.no_grad()
    def encode(self, texts: List[str], **kwargs) -> torch.LongTensor:
        """
        Tokenize a batch of text prompts.

        Parameters
        ----------
        texts : list of str
            Batch of input prompts.

        Returns
        -------
        torch.LongTensor
            Tensor of token IDs.
        """
        enc = self._tokenizer(
            texts,
            return_tensors="pt",
            padding=True,
            truncation=kwargs.pop("truncation", True),
            **kwargs,
        )
        return enc["input_ids"].to(self._device)

    @torch.no_grad()
    def decode(self, ids: torch.LongTensor, **kwargs) -> List[str]:
        """
        Decode a batch of token IDs into text.

        Parameters
        ----------
        ids : torch.LongTensor
            Tensor of token IDs.

        Returns
        -------
        list of str
            Decoded text.
        """
        return self._tokenizer.batch_decode(ids, skip_special_tokens=True, **kwargs)

    @torch.no_grad()
    def predict(self, input_ids: torch.LongTensor, **kwargs) -> torch.Tensor:
        """
        Compute next-token logits for each sequence in the batch.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Tensor of token IDs.

        Returns
        -------
        torch.Tensor
            Logits for the next token.
        """
        if input_ids.device != self._device:
            input_ids = input_ids.to(self._device)

        pad_id = self._tokenizer.pad_token_id
        if pad_id is not None:
            attention_mask = (input_ids != pad_id).long()
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        out = self._model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return out.logits[:, -1, :]

    @torch.no_grad()
    def generate(self, prompts: List[List[dict]], **kwargs) -> List[str]:
        """
        Generate text completions from chat-style prompts.

        Parameters
        ----------
        prompts : list of list of dict
            Batch of chat messages, each formatted as a list of
            {"role": str, "content": str}.
        **kwargs
            Additional parameters for model.generate().

        Returns
        -------
        list of str
            Generated text completions.

        Raises
        ------
        ValueError
            If tokenizer does not define `apply_chat_template`.
        """
        if not hasattr(self._tokenizer, "apply_chat_template"):
            raise ValueError(
                "Tokenizer does not define `apply_chat_template`. "
                "Provide a chat template when loading the model."
            )

        rendered_prompts = [
            self._tokenizer.apply_chat_template(p, add_generation_prompt=True, tokenize=False)
            for p in prompts
        ]

        enc = self._tokenizer(rendered_prompts, return_tensors="pt", padding=True, truncation=True)
        input_ids = enc["input_ids"].to(self._device)
        attention_mask = enc.get("attention_mask")

        if attention_mask is None:
            pad_id = self._tokenizer.pad_token_id
            if pad_id is not None:
                attention_mask = (input_ids != pad_id).long()
            else:
                attention_mask = torch.ones_like(input_ids, dtype=torch.long)
        else:
            attention_mask = attention_mask.to(self._device)

        gen_ids = self._model.generate(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        return self._tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    @torch.no_grad()
    def logprobs(self, prompts: list[str], targets: list[str], **kwargs) -> list[torch.Tensor]:
        """
        Compute log-probabilities for each token in the target continuation.

        Parameters
        ----------
        prompts : list of str
            Conditioning prompts.
        targets : list of str
            Target continuations.

        Returns
        -------
        list of torch.Tensor
            List of log-probabilities for each target token.
            Each tensor has shape [target_len_i].
        """
        assert len(prompts) == len(targets), "Prompts and targets must have the same length."

        enc_p = self._tokenizer(prompts, return_tensors="pt", padding=True, truncation=True)
        enc_t = self._tokenizer(
            targets,
            return_tensors="pt",
            padding=True,
            truncation=True,
            add_special_tokens=False,
        )

        input_ids = torch.cat([enc_p["input_ids"], enc_t["input_ids"]], dim=1).to(self._device)
        attention_mask = torch.cat(
            [enc_p["attention_mask"], torch.ones_like(enc_t["input_ids"])], dim=1
        ).to(self._device)

        out = self._model(input_ids=input_ids, attention_mask=attention_mask, **kwargs)
        logp = out.logits.log_softmax(dim=-1)  # [B, T, V]

        tgt_ids = enc_t["input_ids"].to(self._device)
        pad_id = self._tokenizer.pad_token_id
        tgt_lens = (
            (tgt_ids != pad_id).sum(dim=1) if pad_id is not None else tgt_ids.ne(-1).sum(dim=1)
        )
        prm_lens = enc_p["attention_mask"].sum(dim=1).to(self._device)

        batch_logps = []
        for b in range(input_ids.size(0)):
            Lp = int(prm_lens[b])
            Lt = int(tgt_lens[b])
            if Lt == 0:
                batch_logps.append(torch.empty(0, device=self._device))
                continue

            idx_pos = torch.arange(Lt, device=self._device) + (Lp - 1)
            idx_tok = tgt_ids[b, :Lt]
            lp = logp[b, idx_pos, :].gather(1, idx_tok.unsqueeze(1)).squeeze(1)
            batch_logps.append(lp)

        return batch_logps

    @torch.no_grad()
    def hidden_states(self, input_ids: torch.LongTensor, **kwargs) -> List[torch.Tensor]:
        """
        Return hidden states of the model.

        Parameters
        ----------
        input_ids : torch.LongTensor
            Tensor of token IDs.

        Returns
        -------
        list of torch.Tensor
            Hidden representations per layer.
        """
        if input_ids.device != self._device:
            input_ids = input_ids.to(self._device)

        pad_id = self._tokenizer.pad_token_id
        if pad_id is not None:
            attention_mask = (input_ids != pad_id).long()
        else:
            attention_mask = torch.ones_like(input_ids, dtype=torch.long)

        out = self._model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            **kwargs,
        )
        return list(out.hidden_states)