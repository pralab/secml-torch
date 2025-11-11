"""Basic tests for the language model wrapper."""

import torch

EXPECTED_LOGITS_NDIM = 2


def test_mock_hf_lm(mock_hf_lm):
    """Basic test for mock Hugging Face LM wrapper."""
    texts = ["hello world"]

    # encode / decode
    ids = mock_hf_lm.encode(texts)
    assert isinstance(ids, torch.Tensor)
    decoded = mock_hf_lm.decode(ids)
    assert isinstance(decoded, list)

    # predict
    logits = mock_hf_lm.predict(ids)
    assert logits.ndim == EXPECTED_LOGITS_NDIM

    # generate
    prompts = [[{"role": "user", "content": "The capital of France is"}]]
    generations = mock_hf_lm.generate(prompts)
    assert isinstance(generations[0], str)

    # logprobs
    logps = mock_hf_lm.logprobs(["a"], ["b"])
    assert isinstance(logps, list)
    assert all(isinstance(lp, torch.Tensor) for lp in logps)

    # hidden states
    hidden = mock_hf_lm.hidden_states(ids)
    assert isinstance(hidden, list)
    assert all(isinstance(h, torch.Tensor) for h in hidden)
