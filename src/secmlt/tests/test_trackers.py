import pytest
import torch
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from secmlt.tests.mocks import MockModel
from secmlt.trackers.image_trackers import (
    GradientsTracker,
    SampleTracker,
)
from secmlt.trackers.model_tracker import ModelTracker
from secmlt.trackers.trackers import (
    GradientNormTracker,
    LossTracker,
    PerturbationNormTracker,
    PredictionTracker,
    ScoresTracker,
)

NUM_STEPS = 5
BATCHES = 2


@pytest.fixture
def dummy_data():
    batch, n_features, n_classes = 4, 6, 3
    data = torch.randn(batch, n_features)
    loss_values = torch.randn(batch)
    output_values = torch.randn(batch, n_classes)
    return data, loss_values, output_values


@pytest.mark.parametrize(
    "tracker",
    [
        GradientsTracker(),
        SampleTracker(),
        GradientNormTracker(),
        LossTracker(),
        PerturbationNormTracker(),
        PredictionTracker(),
        ScoresTracker(y=0),
        ScoresTracker(y=None),
    ],
)
def test_tracker_tracks_and_last(dummy_data, tracker):
    data, loss_values, output_values = dummy_data
    for i in range(NUM_STEPS):
        tracker.track(i, loss_values, output_values, data, data, data)

    # should store NUM_STEPS entries
    assert len(tracker.tracked) == NUM_STEPS
    assert all(torch.is_tensor(x) for x in tracker.tracked)

    # last tracked tensor should match
    last = tracker.get_last_tracked()
    assert isinstance(last, torch.Tensor)


def test_init_end_reset_and_get(dummy_data):
    data, loss_values, output_values = dummy_data
    tracker = LossTracker()

    # Before tracking anything
    assert tracker.get().numel() == 0
    assert tracker.get_last_tracked() is None

    # Track some steps
    for i in range(NUM_STEPS):
        tracker.track(i, loss_values, output_values, data, data, data)

    # get() should return stacked tensor
    hist = tracker.get()
    assert hist.shape[-1] == NUM_STEPS

    # Finalize batch
    tracker.end_tracking()
    assert len(tracker._batches) == 1
    assert tracker.get().shape[-1] == NUM_STEPS

    # Add another batch
    for i in range(NUM_STEPS):
        tracker.track(i, loss_values, output_values, data, data, data)
    tracker.end_tracking()
    assert len(tracker._batches) == BATCHES
    out = tracker.get()
    # should concatenate batches
    assert out.shape[0] == loss_values.shape[0] * 2

    # Reset clears everything
    tracker.reset()
    assert tracker.get().numel() == 0
    assert tracker.get_last_tracked() is None


def test_scores_tracker_behavior(dummy_data):
    data, loss_values, scores = dummy_data

    # full scores
    t1 = ScoresTracker(y=None)
    t1.track(0, loss_values, scores, data, data, data)
    assert torch.allclose(t1.get_last_tracked(), scores)

    # single class
    t2 = ScoresTracker(y=1)
    t2.track(0, loss_values, scores, data, data, data)
    assert torch.allclose(t2.get_last_tracked(), scores[:, 1])


def test_prediction_tracker(dummy_data):
    data, loss_values, scores = dummy_data
    tracker = PredictionTracker()
    tracker.track(0, loss_values, scores, data, data, data)
    preds = tracker.get_last_tracked()
    assert torch.equal(preds, scores.argmax(dim=1))


def test_norm_trackers(dummy_data):
    data, loss_values, scores = dummy_data

    # delta and grad are shaped like data
    delta = torch.randn_like(data)
    grad = torch.randn_like(data)

    # Perturbation norm (L2)
    t1 = PerturbationNormTracker()
    t1.track(0, loss_values, scores, data, delta, grad)
    expected = delta.flatten(start_dim=1).norm(p=2, dim=-1)
    assert torch.allclose(t1.get_last_tracked(), expected)

    # Gradient norm (L1)
    t2 = GradientNormTracker()
    t2.track(0, loss_values, scores, data, delta, grad)
    expected = grad.flatten(start_dim=1).norm(p=2, dim=-1)  # default L2
    assert torch.allclose(t2.get_last_tracked(), expected)


# ---- ModelTracker tests ----


@pytest.fixture
def mock_model():
    return BasePyTorchClassifier(model=MockModel())


def test_model_tracker_tracks_on_forward(mock_model):
    """ModelTracker should call child trackers on each forward pass."""
    loss_tracker = LossTracker(loss_fn=torch.nn.CrossEntropyLoss(reduction="none"))
    pred_tracker = PredictionTracker()

    tracked_model = ModelTracker(mock_model, trackers=[loss_tracker, pred_tracker])

    x_orig = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    tracked_model.init_tracking(x_orig=x_orig, y=y)
    for _ in range(NUM_STEPS):
        tracked_model.decision_function(x_orig)
    tracked_model.end_tracking()

    assert loss_tracker.get().shape == (4, NUM_STEPS)
    assert pred_tracker.get().shape == (4, NUM_STEPS)


def test_model_tracker_computes_delta(mock_model):
    """ModelTracker should compute x_adv - x_orig as delta."""
    pert_tracker = PerturbationNormTracker()
    tracked_model = ModelTracker(mock_model, trackers=[pert_tracker])

    x_orig = torch.randn(4, 3, 32, 32)
    tracked_model.init_tracking(x_orig=x_orig)
    tracked_model.decision_function(x_orig)
    tracked_model.end_tracking()

    # delta = x_orig - x_orig = 0, so norm should be 0
    assert torch.allclose(pert_tracker.get(), torch.zeros(4, 1))


def test_model_tracker_no_tracking_when_disabled(mock_model):
    """No tracking should happen before init_tracking or after end_tracking."""
    loss_tracker = LossTracker(loss_fn=torch.nn.CrossEntropyLoss(reduction="none"))
    tracked_model = ModelTracker(mock_model, trackers=[loss_tracker])

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))

    # Before init_tracking, forward should not track
    tracked_model.decision_function(x)
    assert loss_tracker.get().numel() == 0

    # After end_tracking, forward should not track
    tracked_model.init_tracking(x_orig=x, y=y)
    tracked_model.decision_function(x)
    tracked_model.end_tracking()
    count_after_first = loss_tracker.get().shape[-1]

    tracked_model.decision_function(x)
    assert loss_tracker.get().shape[-1] == count_after_first


def test_model_tracker_multi_batch(mock_model):
    """ModelTracker should support multiple init/end tracking cycles."""
    loss_tracker = LossTracker(loss_fn=torch.nn.CrossEntropyLoss(reduction="none"))
    tracked_model = ModelTracker(mock_model, trackers=[loss_tracker])

    batch_size = 4
    for _ in range(BATCHES):
        x = torch.randn(batch_size, 3, 32, 32)
        y = torch.randint(0, 10, (batch_size,))
        tracked_model.init_tracking(x_orig=x, y=y)
        for _ in range(NUM_STEPS):
            tracked_model.decision_function(x)
        tracked_model.end_tracking()

    result = loss_tracker.get()
    assert result.shape == (batch_size * BATCHES, NUM_STEPS)


def test_model_tracker_reset(mock_model):
    """Reset should clear all child trackers."""
    loss_tracker = LossTracker(loss_fn=torch.nn.CrossEntropyLoss(reduction="none"))
    tracked_model = ModelTracker(mock_model, trackers=[loss_tracker])

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    tracked_model.init_tracking(x_orig=x, y=y)
    tracked_model.decision_function(x)
    tracked_model.end_tracking()

    tracked_model.reset()
    assert loss_tracker.get().numel() == 0


def test_model_tracker_detach(mock_model):
    """After detach, forward calls should not trigger tracking."""
    loss_tracker = LossTracker(loss_fn=torch.nn.CrossEntropyLoss(reduction="none"))
    tracked_model = ModelTracker(mock_model, trackers=[loss_tracker])

    tracked_model.detach()

    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    tracked_model.init_tracking(x_orig=x, y=y)
    tracked_model.decision_function(x)
    tracked_model.end_tracking()

    assert loss_tracker.get().numel() == 0


def test_model_tracker_single_tracker(mock_model):
    """Passing a single tracker (not a list) should work."""
    tracker = PredictionTracker()
    tracked_model = ModelTracker(mock_model, trackers=tracker)

    x = torch.randn(4, 3, 32, 32)
    tracked_model.init_tracking(x_orig=x)
    tracked_model.decision_function(x)
    tracked_model.end_tracking()

    assert tracked_model.trackers == [tracker]
    assert tracker.get().shape == (4, 1)


def test_model_tracker_tracks_gradients_on_backward(mock_model):
    """Gradient-dependent trackers should be updated during backward."""
    grad_tracker = GradientNormTracker()
    tracked_model = ModelTracker(mock_model, trackers=[grad_tracker])

    x = torch.randn(4, 3, 32, 32, requires_grad=True)
    tracked_model.init_tracking(x_orig=x)

    scores = tracked_model.decision_function(x)
    scores.sum().backward()
    tracked_model.end_tracking()

    tracked = grad_tracker.get()
    assert tracked.shape == (4, 1)
    assert torch.isfinite(tracked).all()


def test_model_tracker_from_raw_nn_module():
    """ModelTracker should accept a raw nn.Module via BasePyTorchClassifier."""
    raw_module = MockModel()
    wrapped = BasePyTorchClassifier(model=raw_module)
    tracker = PredictionTracker()
    tracked_model = ModelTracker(wrapped, trackers=[tracker])

    x = torch.randn(4, 3, 32, 32)
    tracked_model.init_tracking(x_orig=x)
    tracked_model.decision_function(x)
    tracked_model.end_tracking()

    assert tracker.get().shape == (4, 1)


def test_attack_auto_wraps_nn_module():
    """BaseEvasionAttack._ensure_wrapped should wrap a raw nn.Module."""
    from secmlt.adv.evasion.base_evasion_attack import BaseEvasionAttack

    raw = MockModel()
    wrapped = BaseEvasionAttack._ensure_wrapped(raw)
    assert isinstance(wrapped, BasePyTorchClassifier)

    # Already wrapped should pass through
    wrapped2 = BaseEvasionAttack._ensure_wrapped(wrapped)
    assert wrapped2 is wrapped

    # Unsupported type should raise
    with pytest.raises(TypeError):
        BaseEvasionAttack._ensure_wrapped("not a model")
