"""Verifier tests that need no model, no network and no optional dependency."""

import pytest

from mopd.verifiers.bfcl_verifier import _local_match, parse_calls
from mopd.verifiers.math_verifier import gsm8k_gold


def test_gsm8k_gold_strips_reasoning():
    raw = "Janet has 3 apples.\nShe eats 1.\n#### 2"
    assert gsm8k_gold(raw) == "2"


def test_gsm8k_gold_strips_thousands_separator():
    assert gsm8k_gold("blah\n#### 18,000") == "18000"


def test_parse_calls_json_form():
    text = '[{"name": "get_weather", "arguments": {"city": "Paris", "unit": "c"}}]'
    assert parse_calls(text) == [{"get_weather": {"city": "Paris", "unit": "c"}}]


def test_parse_calls_fenced_json():
    text = 'Sure!\n```json\n[{"name": "f", "arguments": {"x": 1}}]\n```\n'
    assert parse_calls(text) == [{"f": {"x": 1}}]


def test_parse_calls_python_literal_form():
    assert parse_calls("[get_weather(city='Paris', days=3)]") == [
        {"get_weather": {"city": "Paris", "days": 3}}
    ]


def test_parse_calls_rejects_prose():
    assert parse_calls("I would call the weather API for Paris.") == []


def test_parse_calls_never_evaluates():
    """Argument values are parsed as literals; nothing is executed."""
    assert parse_calls("[f(x=__import__('os').system('echo hi'))]") == []


def test_local_match_accepts_exact_call():
    calls = [{"get_weather": {"city": "Paris"}}]
    ref = [{"get_weather": {"city": ["Paris", "paris"]}}]
    assert _local_match(calls, ref)


def test_local_match_is_case_insensitive_on_strings():
    assert _local_match([{"f": {"a": "YES"}}], [{"f": {"a": ["yes"]}}])


def test_local_match_rejects_wrong_value():
    assert not _local_match([{"f": {"a": "no"}}], [{"f": {"a": ["yes"]}}])


def test_local_match_rejects_hallucinated_argument():
    assert not _local_match([{"f": {"a": 1, "b": 2}}], [{"f": {"a": [1]}}])


def test_local_match_allows_omitted_optional_argument():
    assert _local_match([{"f": {"a": 1}}], [{"f": {"a": [1], "b": ["", "x"]}}])


def test_local_match_rejects_omitted_required_argument():
    assert not _local_match([{"f": {"a": 1}}], [{"f": {"a": [1], "b": ["x"]}}])


def test_local_match_rejects_wrong_arity():
    assert not _local_match([{"f": {"a": 1}}], [{"f": {"a": [1]}}, {"g": {}}])


def test_local_match_tolerates_float_noise():
    assert _local_match([{"f": {"x": 1.0000000001}}], [{"f": {"x": [1]}}])


@pytest.mark.parametrize("domain", ["math", "ifeval", "tool"])
def test_registry_exposes_every_domain(domain):
    from mopd.verifiers import _REGISTRY

    assert domain in _REGISTRY
