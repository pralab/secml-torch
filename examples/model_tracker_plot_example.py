"""Simple example: run PGD with trackers and plot tracked values."""

import importlib.util

import matplotlib.pyplot as plt
import torch
from loaders.get_loaders import get_mnist_loader
from secmlt.adv.backends import Backends
from secmlt.adv.evasion.fmn import FMN
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.optimization.losses import LogitDifferenceLoss
from secmlt.trackers.trackers import (
    LossTracker,
    PerturbationNormTracker,
    PredictionTracker,
)


def _mean_over_samples(tracked: torch.Tensor) -> torch.Tensor:
    """Return a 1D curve (mean over samples) for scalar trackers."""
    if tracked.ndim == 1:
        return tracked.float().cpu()
    return tracked.float().mean(dim=0).cpu()


def main() -> None:
    dataset_path = "example_data/datasets/"
    test_loader = get_mnist_loader(dataset_path)

    model = torch.hub.load(
        "maurapintor/mnist_examples",
        "mnist_model",
        weights="teacher",
    )
    model.eval()

    available_backends = [Backends.NATIVE]
    if importlib.util.find_spec("foolbox") is not None:
        available_backends.append(Backends.FOOLBOX)
    if importlib.util.find_spec("adv_lib") is not None:
        available_backends.append(Backends.ADVLIB)

    num_steps = 200
    step_size = 0.03

    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "axes.facecolor": "#f7f8fa",
            "figure.facecolor": "#ffffff",
            "axes.edgecolor": "#d0d7de",
            "axes.labelcolor": "#24292f",
            "xtick.color": "#57606a",
            "ytick.color": "#57606a",
            "font.size": 10,
            "axes.titlesize": 11,
            "axes.labelsize": 10,
        }
    )

    colors = ["#0f766e", "#2563eb", "#f59e0b"]

    n_rows = len(available_backends)
    n_cols = 3
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(3 * n_cols, 2.5 * n_rows),
        constrained_layout=True,
        squeeze=False,
    )

    for row, backend in enumerate(available_backends):
        trackers = [
            LossTracker(loss_fn=LogitDifferenceLoss()),
            PredictionTracker(),
            PerturbationNormTracker(LpPerturbationModels.LINF),
        ]

        attack = FMN(
            perturbation_model=LpPerturbationModels.LINF,
            num_steps=num_steps,
            step_size=step_size,
            y_target=None,
            backend=backend,
            trackers=trackers,
        )

        _ = attack(model, test_loader)

        for col, tracker in enumerate(trackers):
            ax = axes[row][col]
            tracked = tracker.get()
            curve = _mean_over_samples(tracked)
            x = torch.arange(curve.numel())
            color = colors[col % len(colors)]
            ax.plot(
                x,
                curve,
                color=color,
                linewidth=1.5,
            )
            if row == 0:
                ax.set_title(tracker.name, fontweight="bold")
            ax.set_xlabel("Queries")
            ax.grid(alpha=0.35, linestyle="--", linewidth=0.8)
            ax.set_xlim(0, num_steps - 1)

        axes[row][0].set_ylabel(backend, fontweight="bold", fontsize=10)
    fig.suptitle(
        f"FMN Tracker Curves by Backend ({num_steps} queries)",
        fontsize=14,
        fontweight="bold",
    )

    plt.show()


if __name__ == "__main__":
    main()
