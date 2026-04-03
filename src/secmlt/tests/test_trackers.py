import pytest
import torch
from secmlt.adv.evasion.advlib_attacks.advlib_base import BaseAdvLibEvasionAttack
from secmlt.models.base_model import BaseModel
from secmlt.models.pytorch.base_pytorch_nn import BasePyTorchClassifier
from secmlt.tests.mocks import MockModel
from secmlt.trackers.image_trackers import (
    ImageGradientsTracker,
    ImageSampleTracker,
)
from secmlt.trackers.model_tracker import ModelTracker
from secmlt.trackers.trackers import (
    IMAGE,
    MULTI_SCALAR,
    SCALAR,
    GradientNormTracker,
    GradientsTracker,
    LossTracker,
    PerturbationNormTracker,
    PredictionTracker,
    SampleTracker,
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
        ImageGradientsTracker(),
        ImageSampleTracker(),
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


def test_get_last_tracked_uses_finalized_batches(dummy_data):
    data, loss_values, output_values = dummy_data
    tracker = LossTracker()

    tracker.track(0, loss_values, output_values, data, data, data)
    tracker.track(1, loss_values + 1, output_values, data, data, data)
    tracker.end_tracking()

    assert torch.allclose(tracker.get_last_tracked(), loss_values + 1)


def test_loss_and_gradient_norm_trackers_ignore_none_inputs(dummy_data):
    data, _, scores = dummy_data
    loss_tracker = LossTracker()
    grad_norm_tracker = GradientNormTracker()

    loss_tracker.track(0, None, scores, data, data, data)
    grad_norm_tracker.track(0, None, scores, data, data, None)

    assert loss_tracker.get().numel() == 0
    assert grad_norm_tracker.get().numel() == 0


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


def test_model_tracker_grad_tracker_no_requires_grad_input(mock_model):
    """Gradient trackers should not update when input has no grad."""
    grad_tracker = GradientNormTracker()
    tracked_model = ModelTracker(mock_model, trackers=[grad_tracker])

    x = torch.randn(4, 3, 32, 32, requires_grad=False)
    tracked_model.init_tracking(x_orig=x)
    tracked_model.decision_function(x)
    tracked_model.end_tracking()

    assert grad_tracker.get().numel() == 0


def test_model_tracker_del_detaches_hook(mock_model):
    """__del__ should be safe and release the forward hook handle."""
    tracked_model = ModelTracker(mock_model, trackers=[PredictionTracker()])
    assert tracked_model._hook_handle is not None

    tracked_model.__del__()

    assert tracked_model._hook_handle is None


def test_model_tracker_from_raw_nn_module():
    """ModelTracker should accept a raw nn.Module directly."""
    raw_module = MockModel()
    tracker = PredictionTracker()
    tracked_model = ModelTracker(raw_module, trackers=[tracker])

    x = torch.randn(4, 3, 32, 32)
    tracked_model.init_tracking(x_orig=x)
    tracked_model.decision_function(x)
    tracked_model.end_tracking()

    assert tracker.get().shape == (4, 1)


def test_model_tracker_raises_on_unsupported_basemodel_subtype():
    class DummyBaseModel(BaseModel):
        def predict(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return torch.zeros(x.shape[0], dtype=torch.long)

        def _decision_function(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
            return torch.zeros(x.shape[0], 2)

        def train(self, dataloader):
            return self

    with pytest.raises(TypeError, match="BasePyTorchClassifier"):
        ModelTracker(DummyBaseModel(), trackers=[PredictionTracker()])


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


def test_advlib_base_always_cleans_up_model_tracker_on_exception(mock_model):
    class DummyTracker:
        def __init__(self):
            self.init_called = False
            self.end_called = False

        def init_tracking(self):
            self.init_called = True

        def end_tracking(self):
            self.end_called = True

        def reset(self):
            return None

        def track(self, *args, **kwargs):
            return None

    tracker = DummyTracker()
    attack = BaseAdvLibEvasionAttack(
        advlib_attack=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
        trackers=[tracker],
    )
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    hook_count_before = len(mock_model.model._forward_hooks)

    with pytest.raises(RuntimeError, match="boom"):
        attack._run(model=mock_model, samples=x, labels=y)

    hook_count_after = len(mock_model.model._forward_hooks)
    assert tracker.init_called is True
    assert tracker.end_called is True
    assert hook_count_after == hook_count_before


def test_foolbox_base_always_cleans_up_model_tracker_on_exception(
    mock_model, monkeypatch):
    pytest.importorskip("foolbox")
    from secmlt.adv.evasion.foolbox_attacks.foolbox_base import BaseFoolboxEvasionAttack

    class DummyTracker:
        def __init__(self):
            self.init_called = False
            self.end_called = False

        def init_tracking(self):
            self.init_called = True

        def end_tracking(self):
            self.end_called = True

        def reset(self):
            return None

        def track(self, *args, **kwargs):
            return None

    class DummyPyTorchModel:
        def __init__(self, model, bounds, device):
            self.model = model
            self.bounds = bounds
            self.device = device

    class DummyMisclassification:
        def __init__(self, labels):
            self.labels = labels

    monkeypatch.setattr(
        "secmlt.adv.evasion.foolbox_attacks.foolbox_base.PyTorchModel",
        DummyPyTorchModel,
    )
    monkeypatch.setattr(
        "secmlt.adv.evasion.foolbox_attacks.foolbox_base.Misclassification",
        DummyMisclassification,
    )

    tracker = DummyTracker()
    attack = BaseFoolboxEvasionAttack(
        foolbox_attack=lambda **_: (_ for _ in ()).throw(RuntimeError("boom")),
        trackers=[tracker],
    )
    x = torch.randn(4, 3, 32, 32)
    y = torch.randint(0, 10, (4,))
    hook_count_before = len(mock_model.model._forward_hooks)

    with pytest.raises(RuntimeError, match="boom"):
        attack._run(model=mock_model, samples=x, labels=y)

    hook_count_after = len(mock_model.model._forward_hooks)
    assert tracker.init_called is True
    assert tracker.end_called is True
    assert hook_count_after == hook_count_before


def test_generic_and_image_sample_tracker_types(dummy_data):
    data, loss_values, scores = dummy_data
    generic_tracker = SampleTracker()
    image_tracker = ImageSampleTracker()

    # For the generic SampleTracker with SCALAR type, track per-sample scalars
    generic_tracker.track(0, loss_values, scores, loss_values, loss_values, loss_values)
    # The ImageSampleTracker should continue to track image-like multi-dimensional data
    image_tracker.track(0, loss_values, scores, data, data, data)

    assert generic_tracker.tracked_type == MULTI_SCALAR
    assert image_tracker.tracked_type == IMAGE
    assert torch.allclose(generic_tracker.get_last_tracked(), loss_values)
    assert torch.allclose(image_tracker.get_last_tracked(), data)


def test_generic_and_image_gradients_tracker_types(dummy_data):
    data, loss_values, scores = dummy_data
    generic_tracker = GradientsTracker()
    image_tracker = ImageGradientsTracker()

    generic_tracker.track(0, loss_values, scores, data, data, data)
    image_tracker.track(0, loss_values, scores, data, data, data)

    assert generic_tracker.tracked_type == MULTI_SCALAR
    assert image_tracker.tracked_type == IMAGE
    assert torch.allclose(generic_tracker.get_last_tracked(), data)
    assert torch.allclose(image_tracker.get_last_tracked(), data)


def test_sample_tracker_scalar_type_rejects_non_scalar_samples(dummy_data):
    data, loss_values, scores = dummy_data
    tracker = SampleTracker(tracker_type=SCALAR)

    with pytest.raises(ValueError, match="SampleTracker with tracker_type='scalar'"):
        tracker.track(0, loss_values, scores, data, data, data)


def test_gradients_tracker_scalar_type_rejects_non_scalar_samples(dummy_data):
    data, loss_values, scores = dummy_data
    tracker = GradientsTracker(tracker_type=SCALAR)

    with pytest.raises(
        ValueError,
        match="GradientsTracker with tracker_type='scalar'",
    ):
        tracker.track(0, loss_values, scores, data, data, data)


def test_perturbation_norm_tracker_skips_when_delta_is_none(dummy_data):
    """PerturbationNormTracker must not crash when delta is None (x_orig not set)."""
    data, loss_values, scores = dummy_data
    tracker = PerturbationNormTracker()

    # Should not raise, and should not append anything
    tracker.track(0, loss_values, scores, data, None, None)

    assert len(tracker.tracked) == 0


def test_model_tracker_perturbation_norm_without_x_orig(mock_model):
    """ModelTracker with PerturbationNormTracker must not crash when x_orig omitted."""
    tracker = PerturbationNormTracker()
    tracked_model = ModelTracker(mock_model, trackers=[tracker])

    x = torch.randn(4, 6)
    # init_tracking without x_orig → delta will be None inside the forward hook
    tracked_model.init_tracking()
    tracked_model.decision_function(x)
    tracked_model.end_tracking()

    # Nothing should have been appended (delta=None silently skips)
    assert len(tracker.tracked) == 0


def test_tensorboard_tracker_requires_grad_reflects_sub_trackers(tmp_path):
    """TensorboardTracker.requires_grad is true when a grad sub-tracker exists."""
    pytest.importorskip("tensorboard")
    from secmlt.trackers.tensorboard_tracker import TensorboardTracker

    tb_only_loss = TensorboardTracker(logdir=str(tmp_path), trackers=[LossTracker()])
    assert tb_only_loss.requires_grad is False

    tb_with_grad = TensorboardTracker(
        logdir=str(tmp_path),
        trackers=[LossTracker(), GradientNormTracker()],
    )
    assert tb_with_grad.requires_grad is True


def test_tensorboard_tracker_loss_fn_propagated_from_sub_tracker(tmp_path):
    """TensorboardTracker.loss_fn must expose the sub-LossTracker's loss_fn."""
    pytest.importorskip("tensorboard")
    from secmlt.trackers.tensorboard_tracker import TensorboardTracker

    loss_tracker = LossTracker()
    tb = TensorboardTracker(
        logdir=str(tmp_path),
        trackers=[loss_tracker, ScoresTracker()],
    )

    assert tb.loss_fn is loss_tracker.loss_fn

    # TensorboardTracker with no LossTracker should return None
    tb_no_loss = TensorboardTracker(logdir=str(tmp_path), trackers=[ScoresTracker()])
    assert tb_no_loss.loss_fn is None
