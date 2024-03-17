import torch

NUM_STEPS = 5


class MockModel:
    def predict(self, x):
        return torch.randint(0, 10, (len(x),))


def test_loss_tracker(data, loss_values, output_values, loss_tracker) -> None:
    for i in range(NUM_STEPS):
        loss_tracker.track(i, loss_values, output_values, data, data, data)
    assert len(loss_tracker.tracked) == NUM_STEPS
    assert all(torch.is_tensor(x) for x in loss_tracker.tracked)


def test_scores_tracker(data, loss_values, output_values, scores_tracker) -> None:
    for i in range(NUM_STEPS):
        scores_tracker.track(i, loss_values, output_values, data, data, data)
    assert len(scores_tracker.tracked) == NUM_STEPS
    assert all(torch.is_tensor(x) for x in scores_tracker.tracked)


def test_prediction_tracker(data, loss_values, output_values, prediction_tracker):
    for i in range(NUM_STEPS):
        prediction_tracker.track(i, loss_values, output_values, data, data, data)
    assert len(prediction_tracker.tracked) == NUM_STEPS
    assert all(torch.is_tensor(x) for x in prediction_tracker.tracked)


def test_perturbation_norm_tracker(
    data,
    loss_values,
    output_values,
    perturbation_norm_tracker,
) -> None:
    for i in range(NUM_STEPS):
        perturbation_norm_tracker.track(i, loss_values, output_values, data, data, data)
    assert len(perturbation_norm_tracker.tracked) == NUM_STEPS
    assert all(torch.is_tensor(x) for x in perturbation_norm_tracker.tracked)


def test_gradient_norm_tracker(
    data,
    loss_values,
    output_values,
    gradient_norm_tracker,
) -> None:
    for i in range(NUM_STEPS):
        gradient_norm_tracker.track(i, loss_values, output_values, data, data, data)
    assert len(gradient_norm_tracker.tracked) == NUM_STEPS
    assert all(torch.is_tensor(x) for x in gradient_norm_tracker.tracked)
