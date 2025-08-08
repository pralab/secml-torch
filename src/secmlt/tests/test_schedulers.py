import pytest
import torch
from secmlt.optimization.scheduler_factory import (
    COSINE_ANNEALING,
    NO_SCHEDULER,
    LRSchedulerFactory,
    NoScheduler,
)
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LRScheduler


def test_create_no_scheduler():
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.1)

    scheduler_factory = LRSchedulerFactory.create_no_scheduler()
    scheduler = scheduler_factory(optimizer)

    assert isinstance(scheduler, NoScheduler)
    # step should not raise or change lr
    lr_before = optimizer.param_groups[0]["lr"]
    scheduler.step()
    lr_after = optimizer.param_groups[0]["lr"]
    assert lr_before == lr_after


def test_create_cosine_annealing():
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.1)

    scheduler_factory = LRSchedulerFactory.create_cosine_annealing()
    with pytest.raises(TypeError):
        scheduler_factory(optimizer)

    scheduler = scheduler_factory(optimizer, T_max=10)
    assert isinstance(scheduler, CosineAnnealingLR)


@pytest.mark.parametrize(
    ["scheduler_name", "scheduler_args"],
    [(COSINE_ANNEALING, {"T_max": 10}), (NO_SCHEDULER, {})],
)
def test_create_from_name(scheduler_name, scheduler_args):
    factory = LRSchedulerFactory.create_from_name(scheduler_name)
    assert callable(factory)

    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.01)

    scheduler = factory(optimizer, **scheduler_args)
    assert isinstance(scheduler, LRScheduler)


def test_create_from_name_invalid():
    with pytest.raises(ValueError, match="Scheduler not found"):
        LRSchedulerFactory.create_from_name("invalid_scheduler")


def test_no_scheduler_step_has_no_effect():
    model = torch.nn.Linear(1, 1)
    optimizer = SGD(model.parameters(), lr=0.01)

    scheduler = NoScheduler(optimizer)
    lr_before = optimizer.param_groups[0]["lr"]

    scheduler.step()
    lr_after = optimizer.param_groups[0]["lr"]

    assert lr_before == lr_after
