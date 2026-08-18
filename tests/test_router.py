import pytest
import torch

from mopd.router import ORACLE_PRIOR, route, routing_entropy

ROLES = ["math", "instruct", "general"]


def test_uniform_is_uniform():
    w = route("uniform", ROLES, mean_logprobs=torch.zeros(4, 3))
    assert torch.allclose(w, torch.full((4, 3), 1 / 3))
    assert routing_entropy(w).mean() == pytest.approx(1.0)


def test_single_is_one_hot():
    w = route("single", ROLES, mean_logprobs=torch.zeros(2, 3), single_index=1)
    assert torch.equal(w, torch.tensor([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]]))
    assert routing_entropy(w).mean() == pytest.approx(0.0, abs=1e-6)


def test_oracle_follows_the_domain_prior():
    w = route("oracle", ROLES, domains=["math", "ifeval", "tool"])
    for i, d in enumerate(["math", "ifeval", "tool"]):
        want = torch.tensor([ORACLE_PRIOR[d][r] for r in ROLES])
        assert torch.allclose(w[i], want)


def test_oracle_falls_back_to_uniform_on_unknown_domain():
    w = route("oracle", ROLES, domains=["not-a-domain"])
    assert torch.allclose(w[0], torch.full((3,), 1 / 3))


def test_confidence_prefers_the_least_surprised_teacher():
    mlp = torch.tensor([[-0.5, -2.0, -3.0]])
    w = route("confidence", ROLES, mean_logprobs=mlp, temperature=0.5)
    assert w.sum() == pytest.approx(1.0)
    assert w[0, 0] > w[0, 1] > w[0, 2]


def test_confidence_temperature_controls_sharpness():
    mlp = torch.tensor([[-0.5, -1.0, -1.5]])
    sharp = route("confidence", ROLES, mean_logprobs=mlp, temperature=0.1)
    soft = route("confidence", ROLES, mean_logprobs=mlp, temperature=5.0)
    assert routing_entropy(sharp) < routing_entropy(soft)


def test_confidence_token_is_per_position():
    tlp = torch.randn(2, 5, 3)
    w = route("confidence_token", ROLES, token_logprobs=tlp)
    assert w.shape == (2, 5, 3)
    assert torch.allclose(w.sum(-1), torch.ones(2, 5))


def test_unknown_router_rejected():
    with pytest.raises(ValueError, match="unknown router"):
        route("magic", ROLES, mean_logprobs=torch.zeros(1, 3))


def test_missing_inputs_rejected():
    with pytest.raises(ValueError, match="domain labels"):
        route("oracle", ROLES)
    with pytest.raises(ValueError, match="mean_logprobs"):
        route("confidence", ROLES)
