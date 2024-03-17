from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.adv.evasion.pgd import PGD


def test_pgd_native(model, data_loader) -> None:
    attack = PGD(
        perturbation_model=LpPerturbationModels.LINF,
        epsilon=0.5,
        num_steps=10,
        step_size=0.1,
        random_start=True,
        y_target=None,
        lb=0.0,
        ub=1.0,
        backend="native",
    )
    attack(model, data_loader)
