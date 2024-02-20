try:
    import foolbox
except ImportError:
    pass  # foolbox is an extra component and requires the foolbox library
else:
    from .foolbox_attacks import *
