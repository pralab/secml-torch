class PerturbationModels:
    L0 = "l0"
    L1 = "l1"
    L2 = "l2"
    LINF = "linf"
    pert_models = {
            L0: 0, L1: 1, L2: 2, LINF: float("inf")
        }

    @classmethod
    def is_perturbation_model_available(cls, perturbation_model) -> bool:
        return perturbation_model in (cls.pert_models)

    @classmethod
    def get_p(cls, perturbation_model) -> float:
        if cls.is_perturbation_model_available(perturbation_model):
            return cls.pert_models[perturbation_model]
        raise ValueError("Perturbation model not implemented")