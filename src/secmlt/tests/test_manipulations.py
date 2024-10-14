import pytest
import torch
from secmlt.manipulations.manipulation import AdditiveManipulation
from secmlt.optimization.constraints import Constraint


class MockConstraint(Constraint):
    def __init__(self, mock_return):
        self.mock_return = mock_return

    def _apply_constraint(self, x: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.mock_return


@pytest.fixture
def input_tensor():
    return torch.tensor([[1.0, 2.0], [3.0, 4.0]])


@pytest.fixture
def delta_tensor():
    return torch.tensor([[0.1, 0.2], [0.3, 0.4]])


@pytest.fixture
def domain_constraint():
    return MockConstraint(torch.tensor([[0.5, 0.6], [0.7, 0.8]]))


@pytest.fixture
def perturbation_constraint():
    return MockConstraint(torch.tensor([[0.1, 0.2], [0.3, 0.4]]))


@pytest.fixture
def additive_manipulation(domain_constraint, perturbation_constraint):
    return AdditiveManipulation(
        domain_constraints=[domain_constraint],
        perturbation_constraints=[perturbation_constraint],
    )


def test_apply_domain_constraints(additive_manipulation, input_tensor):
    # apply the domain constraint and check if it modifies the input correctly
    result = additive_manipulation._apply_domain_constraints(input_tensor)
    expected = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


def test_apply_perturbation_constraints(additive_manipulation, delta_tensor):
    # apply the perturbation constraint and check if it modifies the delta correctly
    result = additive_manipulation._apply_perturbation_constraints(delta_tensor)
    expected = torch.tensor([[0.1, 0.2], [0.3, 0.4]])
    assert torch.equal(result, expected), f"Expected {expected}, got {result}"


def test_additive_manipulation(additive_manipulation, input_tensor, delta_tensor):
    # test the additive manipulation
    x_adv, delta = additive_manipulation(input_tensor, delta_tensor)
    expected_x_adv = torch.tensor([[0.5, 0.6], [0.7, 0.8]])
    expected_delta = torch.tensor([[0.1, 0.2], [0.3, 0.4]])

    assert torch.equal(x_adv, expected_x_adv), f"Expected {expected_x_adv}, got {x_adv}"
    assert torch.equal(delta, expected_delta), f"Expected {expected_delta}, got {delta}"


def test_getters_and_setters(
    additive_manipulation, domain_constraint, perturbation_constraint
):
    # test getter for domain_constraints
    domain_constraints = additive_manipulation.domain_constraints
    assert domain_constraints == [
        domain_constraint
    ], f"Expected domain constraints {domain_constraint}, got {domain_constraints}"

    # test setter for domain_constraints
    new_domain_constraint = MockConstraint(torch.tensor([[0.0, 0.1], [0.2, 0.3]]))
    additive_manipulation.domain_constraints = [new_domain_constraint]
    assert additive_manipulation.domain_constraints == [
        new_domain_constraint
    ], "Domain constraints setter did not update correctly"

    # test getter for perturbation_constraints
    perturbation_constraints = additive_manipulation.perturbation_constraints
    assert perturbation_constraints == [perturbation_constraint], (
        f"Expected perturbation constraints {perturbation_constraint}, "
        f"got {perturbation_constraints}"
    )

    # test setter for perturbation_constraints
    new_perturbation_constraint = MockConstraint(
        torch.tensor([[0.05, 0.15], [0.25, 0.35]])
    )
    additive_manipulation.perturbation_constraints = [new_perturbation_constraint]
    assert additive_manipulation.perturbation_constraints == [
        new_perturbation_constraint
    ], "Perturbation constraints setter did not update correctly"
