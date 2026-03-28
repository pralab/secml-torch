"""Image-specific trackers."""

from secmlt.trackers.trackers import (
    IMAGE,
    GradientsTracker,
    SampleTracker,
)


class ImageSampleTracker(SampleTracker):
    """Tracker for adversarial examples."""

    def __init__(self) -> None:
        """Create adversarial image tracker."""
        super().__init__(tracker_type=IMAGE)


class ImageGradientsTracker(GradientsTracker):
    """Tracker for gradient images."""

    def __init__(self) -> None:
        """Create gradients tracker."""
        super().__init__(tracker_type=IMAGE)

