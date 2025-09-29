import pytest
import torch
from secmlt.trackers.image_trackers import (
    GradientsTracker,
    SampleTracker,
)
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
