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
def test_tracker(data, loss_values, output_values, tracker) -> None:
    for i in range(NUM_STEPS):
        tracker.track(i, loss_values, output_values, data, data, data)
    assert len(tracker.tracked) == NUM_STEPS
    assert all(torch.is_tensor(x) for x in tracker.tracked)
    assert torch.is_tensor(tracker.get_last_tracked())
