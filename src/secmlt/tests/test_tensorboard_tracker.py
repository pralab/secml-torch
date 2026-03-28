from __future__ import annotations

import pytest
import torch
from secmlt.trackers.trackers import IMAGE, MULTI_SCALAR, SCALAR


tb_module = pytest.importorskip("secmlt.trackers.tensorboard_tracker")
TensorboardTracker = tb_module.TensorboardTracker


class _FakeWriter:
    def __init__(self, log_dir: str):
        self.log_dir = log_dir
        self.scalar_calls = []
        self.scalars_calls = []
        self.image_calls = []

    def add_scalar(self, tag, scalar_value, global_step):
        self.scalar_calls.append((tag, scalar_value, global_step))

    def add_scalars(self, main_tag, tag_scalar_dict, global_step):
        self.scalars_calls.append((main_tag, tag_scalar_dict, global_step))

    def add_image(self, tag, img_tensor, global_step):
        self.image_calls.append((tag, img_tensor, global_step))


class _DummyTracker:
    def __init__(self, name: str, tracked_type: str, value: torch.Tensor):
        self.name = name
        self.tracked_type = tracked_type
        self._value = value
        self.tracked = []
        self.init_calls = 0
        self.end_calls = 0
        self.track_calls = 0

    def init_tracking(self):
        self.init_calls += 1

    def end_tracking(self):
        self.end_calls += 1

    def track(self, iteration, loss, scores, x_adv, delta, grad):
        self.track_calls += 1

    def get_last_tracked(self):
        return self._value


@pytest.fixture
def fake_writer(monkeypatch):
    captured = {}

    def _factory(log_dir: str):
        writer = _FakeWriter(log_dir=log_dir)
        captured["writer"] = writer
        return writer

    monkeypatch.setattr(tb_module, "SummaryWriter", _factory)
    return captured


def test_tensorboard_tracker_routes_scalar_multi_and_image(fake_writer):
    scalar_tracker = _DummyTracker(
        name="ScalarT",
        tracked_type=SCALAR,
        value=torch.tensor([1.0, 2.0]),
    )
    multi_tracker = _DummyTracker(
        name="MultiT",
        tracked_type=MULTI_SCALAR,
        value=torch.tensor([[1.0, 2.0], [3.0, 4.0]]),
    )
    image_tracker = _DummyTracker(
        name="ImageT",
        tracked_type=IMAGE,
        value=torch.zeros(2, 1, 2, 2),
    )

    tracker = TensorboardTracker(
        logdir="/tmp/tb-tests",
        trackers=[scalar_tracker, multi_tracker, image_tracker],
    )

    tracker.track(
        iteration=7,
        loss=torch.zeros(2),
        scores=torch.zeros(2, 2),
        x_adv=torch.zeros(2, 2),
        delta=torch.zeros(2, 2),
        grad=torch.zeros(2, 2),
    )

    writer = fake_writer["writer"]
    assert len(writer.scalar_calls) == 2
    assert len(writer.scalars_calls) == 2
    assert len(writer.image_calls) == 2

    assert writer.scalar_calls[0][0] == "Sample #0/ScalarT"
    assert writer.scalar_calls[1][0] == "Sample #1/ScalarT"
    assert writer.scalar_calls[0][2] == 7

    assert writer.scalars_calls[0][0] == "Sample #0/MultiT"
    assert writer.scalars_calls[1][0] == "Sample #1/MultiT"
    assert writer.scalars_calls[0][2] == 7

    assert writer.image_calls[0][0] == "Sample #0/ImageT"
    assert writer.image_calls[1][0] == "Sample #1/ImageT"
    assert writer.image_calls[0][2] == 7


def test_tensorboard_tracker_init_and_end_tracking(fake_writer):
    first = _DummyTracker(
        name="First",
        tracked_type=SCALAR,
        value=torch.tensor([1.0, 2.0]),
    )
    second = _DummyTracker(
        name="Second",
        tracked_type=SCALAR,
        value=torch.tensor([3.0, 4.0]),
    )

    # end_tracking computes offset from first tracker tracked history
    first.tracked = [torch.zeros(3)]

    tracker = TensorboardTracker(logdir="/tmp/tb-tests", trackers=[first, second])
    assert tracker._global_sample_offset == 0

    tracker.init_tracking()
    tracker.end_tracking()

    assert first.init_calls == 1
    assert second.init_calls == 1
    assert first.end_calls == 1
    assert second.end_calls == 1
    assert tracker._global_sample_offset == 3


def test_tensorboard_tracker_get_last_tracked_returns_not_implemented(fake_writer):
    tracker = TensorboardTracker(logdir="/tmp/tb-tests", trackers=[])
    result = tracker.get_last_tracked()
    assert isinstance(result, NotImplementedError)
