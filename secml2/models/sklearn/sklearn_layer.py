import torch
from torch import nn


class SklearnAutogradFunction(torch.autograd.Function):
    """
    This class wraps a generic Sklearn classifier inside a PyTorch
    autograd function. When the function's backward is called,
    the Sklearn module calls the internal backward of the wrapped BaseModel,
    and links it to the external graph.
    Reference here:
    https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    """

    @staticmethod
    def forward(ctx, input, clf):
        ctx.clf = clf
        ctx.save_for_backward(input)
        out = clf.decision_function(input)
        return out

    @staticmethod
    def backward(ctx, grad_output):
        clf = ctx.clf
        input = ctx.saved_tensors
        # https://github.com/pytorch/pytorch/issues/1776#issuecomment-372150869
        with torch.enable_grad():
            grad_input = clf.gradient(x=input, y=grad_output)
        grad_input = as_tensor(grad_input, True)
        input_shape = input.shape
        grad_input = grad_input.reshape(input_shape)
        return grad_input, None, None, None


def as_tensor(x, requires_grad=False, tensor_type=None):
    x = torch.from_numpy(x)
    x = x.type(x.dtype if tensor_type is None else tensor_type)
    x.requires_grad = requires_grad
    return x


def as_array(x, dtype=None):
    return x.cpu().detach().numpy().astype(dtype)


class SklearnLayer(nn.Module):
    """
    Defines a PyTorch module that wraps a BaseModel classifier.
    Allows autodiff of the Sklearn modules.
    Credits: https://pytorch.org/tutorials/beginner/examples_autograd/two_layer_net_custom_function.html
    Parameters
    ----------
    model : CCLassifier
       Classifier to wrap in the layer. When the layer's backward
       is called, it will internally run the clf's backward and store
       accumulated gradients in the input tensor.
       Function and Gradient call counts will be tracked,
       however they must be reset externally before the call.
    """

    def __init__(self, model):
        super(SklearnLayer, self).__init__()
        self._clf = model
        self.sklearn_autograd = SklearnAutogradFunction.apply
        self.eval()
        self.func_counter = torch.tensor(0)
        self.grad_counter = torch.tensor(0)

    def forward(self, x):
        x = x.reshape(x.shape[0], -1)
        x = self.sklearn_autograd(x, self._clf)
        return as_tensor(x)
