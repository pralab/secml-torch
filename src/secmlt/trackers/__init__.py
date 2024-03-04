try:
    import tensorboard
except ImportError:
    pass  # tensorboard is an extra
else:
    from .tensorboard_tracker import TensorboardTracker

from .trackers import *
from .image_trackers import *
