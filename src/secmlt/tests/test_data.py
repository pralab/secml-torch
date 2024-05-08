import torch
from secmlt.adv.evasion.perturbation_models import LpPerturbationModels
from secmlt.data.distributions import GeneralizedNormal, Rademacher
from secmlt.data.lp_uniform_sampling import LpUniformSampling


def test_rademacher_dist_shape():
    dist = Rademacher()
    shape = torch.Size([3, 4])
    sample = dist.sample(shape)
    assert sample.shape == shape


def test_rademacher_dist_values():
    dist = Rademacher()
    shape = torch.Size([3, 4])
    sample = dist.sample(shape)
    assert (sample == -1).any()
    assert (sample == 1).any()
    assert not (((sample != -1) & (sample != 1)).all())


def test_gnormal_dist_shape():
    dist = GeneralizedNormal()
    shape = torch.Size([3, 4])
    sample = dist.sample(shape)
    assert sample.shape == shape


def test_gnormal_dist_dtype():
    dist = GeneralizedNormal()
    shape = torch.Size([3, 4])
    sample = dist.sample(shape)
    assert sample.dtype == torch.float32


def test_gnormal_dist_p():
    dist = GeneralizedNormal()
    shape = torch.Size([3, 4])
    sample1 = dist.sample(shape, p=1)
    sample2 = dist.sample(shape, p=2)
    assert not torch.equal(sample1, sample2)


def test_lp_uniform_sampling():
    shape = (1, 32)
    for _ in range(1, 4):
        for perturbation_model in LpPerturbationModels.pert_models:
            _p = LpPerturbationModels.get_p(perturbation_model)
            sampler = LpUniformSampling(p=perturbation_model)
            rvs = sampler.sample(*shape)

            assert rvs.shape == shape
            if perturbation_model not in [
                LpPerturbationModels.L0,
                LpPerturbationModels.LINF,
            ]:
                assert ((torch.abs(rvs) ** _p).sum(-1) ** (1 / _p) <= 1).all()
            elif perturbation_model == LpPerturbationModels.L0:
                pass
            elif perturbation_model == LpPerturbationModels.LINF:
                assert (torch.abs(rvs) <= 1).all()
