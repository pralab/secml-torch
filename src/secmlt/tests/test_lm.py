"""Tests for the Hugging Face language model wrapper."""

import pytest
import torch

EXPECTED_LOGITS_NDIM = 2

def test_hf_load_cached(mock_hf_lm):
    """Ensure load() exits early when model/tokenizer are already loaded."""
    mock_hf_lm._model = mock_hf_lm.model
    mock_hf_lm._tokenizer = mock_hf_lm.tokenizer
    mock_hf_lm.load()

def test_hf_tokenizer(mock_hf_lm):
    """Test encode/decode and tokenizer-related error handling."""
    texts = ["hello world"]

    ids = mock_hf_lm.encode(texts)
    assert isinstance(ids, torch.Tensor)
    decoded = mock_hf_lm.decode(ids)
    assert isinstance(decoded, list)

    if hasattr(mock_hf_lm.tokenizer, "apply_chat_template"):
        del mock_hf_lm.tokenizer.apply_chat_template
    with pytest.raises(ValueError, match="apply_chat_template"):
        mock_hf_lm.generate([[{"role": "user", "content": "hi"}]])

def test_hf_model(mock_hf_lm):
    """Test model behavior: predict, __call__, generate, logprobs, hidden states."""
    texts = ["hello world"]
    ids = mock_hf_lm.encode(texts)

    logits = mock_hf_lm.predict(ids)
    assert logits.ndim == EXPECTED_LOGITS_NDIM

    out = mock_hf_lm(ids)
    assert out.shape == logits.shape

    prompts = [[{"role": "user", "content": "The capital of France is"}]]
    generations = mock_hf_lm.generate(prompts)
    assert isinstance(generations[0], str)

    logps = mock_hf_lm.logprobs(["a"], ["b"])
    assert isinstance(logps, list)
    assert all(isinstance(lp, torch.Tensor) for lp in logps)

    res = mock_hf_lm.logprobs(["ctx"], [""])
    assert isinstance(res[0], torch.Tensor)
    assert res[0].numel() == 0

    ids = mock_hf_lm.encode(["hi"])
    hidden = mock_hf_lm.hidden_states(ids)
    assert isinstance(hidden, list)
    assert all(isinstance(h, torch.Tensor) for h in hidden)

    mock_hf_lm.tokenizer.pad_token_id = None
    ids = mock_hf_lm.encode(["hi"])
    out = mock_hf_lm.predict(ids)
    assert isinstance(out, torch.Tensor)

    mock_hf_lm.tokenizer.pad_token_id = None
    ids = mock_hf_lm.encode(["hi"])
    hidden = mock_hf_lm.hidden_states(ids)
    assert isinstance(hidden, list)
    assert all(isinstance(h, torch.Tensor) for h in hidden)
