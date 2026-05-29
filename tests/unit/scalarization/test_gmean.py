import torch
from pytest import mark
from torch import Tensor
from utils.tensors import rand_, tensor_

from torchjd.scalarization import GMean

from ._asserts import (
    assert_grad_flow,
    assert_permutation_invariant,
    assert_returns_scalar,
)
from ._inputs import shapes

positive_inputs: list[Tensor] = [rand_(shape) + 1 for shape in shapes]


def test_value() -> None:
    losses = tensor_([1.0, 2.0, 4.0])
    torch.testing.assert_close(GMean()(losses), tensor_(2.0))


@mark.parametrize("losses", positive_inputs)
def test_expected_structure(losses: Tensor) -> None:
    assert_returns_scalar(GMean(), losses)


@mark.parametrize("losses", positive_inputs)
def test_grad_flow(losses: Tensor) -> None:
    assert_grad_flow(GMean(), losses)


@mark.parametrize("losses", positive_inputs)
def test_permutation_invariant(losses: Tensor) -> None:
    assert_permutation_invariant(GMean(), losses)


def test_returns_zero_on_non_positive_input() -> None:
    # exp(log(x)) is undefined for x <= 0; the forward short-circuits to 0 in that case.
    assert GMean()(tensor_([1.0, 0.0])) == 0.0
    assert GMean()(tensor_([1.0, -1.0])) == 0.0


def test_grad_flow_on_zero_branch() -> None:
    # The short-circuit branch must keep the autograd graph: backward() returns zero
    # gradients rather than raising "does not have a grad_fn".
    assert_grad_flow(GMean(), tensor_([1.0, 0.0, 2.0]))


def test_representations() -> None:
    s = GMean()
    assert repr(s) == "GMean()"
    assert str(s) == "GMean"
