import functools

from torch.optim import Adam, SGD

ADAM = "adam"
StochasticGD = "sgd"


class OptimizerFactory:
    OPTIMIZERS = {ADAM: Adam, StochasticGD: SGD}

    @staticmethod
    def create_from_name(optimizer_name, lr:float, **kwargs):
        if optimizer_name == ADAM:
            return OptimizerFactory.create_adam(lr)
        if optimizer_name == SGD:
            return OptimizerFactory.create_sgd(lr)

    @staticmethod
    def create_adam(lr: float):
        return functools.partial(Adam, lr=lr)

    @staticmethod
    def create_sgd(lr: float):
        return functools.partial(SGD, lr=lr)
