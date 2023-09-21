from typing import Any

import torch
from secml2.models.data_processing.data_processing import DataProcessing
from secml2.models.sklearn.base_sklearn_model import BaseSklearnModel
from sklearn.svm import SVC

from secml2.models.sklearn.sklearn_layer import SklearnLayer


class SVM(BaseSklearnModel):
    def __init__(
        self,
        C: float = 1,
        kernel: str = "rbf",
        degree: int = 3,
        gamma: str = "scale",
        coef0: float = 0,
        shrinking: bool = True,
        probability: bool = False,
        tol: float = 0.001,
        cache_size: int = 200,
        class_weight: Any | None = None,
        verbose: bool = False,
        max_iter: int = -1,
        decision_function_shape: str = "ovr",
        break_ties: bool = False,
        random_state: Any | None = None,
        preprocessing: DataProcessing = None,
    ):
        model = SVC(
            C=C,
            kernel=kernel,
            degree=degree,
            gamma=gamma,
            coef0=coef0,
            shrinking=shrinking,
            probability=probability,
            tol=tol,
            cache_size=cache_size,
            class_weight=class_weight,
            verbose=verbose,
            max_iter=max_iter,
            decision_function_shape=decision_function_shape,
            break_ties=break_ties,
            random_state=random_state,
        )
        super().__init__(model, preprocessing)
        self._pytorch_model = SklearnLayer(self)
 
    @property
    def kernel(self):
        return self._model._clf.kernel
    
    @property
    def alpha(self):
        return self._model._clf.dual_coef_ if self.kernel is not None else None

    @property
    def n_classes(self):
        return len(self._model._clf.classes_)

    def _gradient(self, x: torch.Tensor, y: int) -> torch.Tensor:
        y = y.detach().cpu().numpy()
        v = self.coef_ if self.kernel is None else self.alpha
        if self.n_classes > 2:
            return y.dot(v)
        else:
            return y[0] * -v + y[1] * v
