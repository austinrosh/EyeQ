"""Pipeline construction, the LTI/tail split, and update routing."""

from eyeq.core import CANONICAL_ORDER, Kind, Pipeline
from eyeq.io import build_pipeline, default_link_config


def _pipe():
    return build_pipeline(default_link_config(modulation="PAM4", reach_class="VSR"))


def test_canonical_order():
    assert _pipe().names() == CANONICAL_ORDER


def test_lti_tail_split():
    pipe = _pipe()
    assert [b.name for b in pipe.lti_prefix()] == [
        "source", "txffe", "txjitter", "channel", "noise", "ctle", "rxffe",
    ]
    assert [b.name for b in pipe.nonlinear_tail()] == ["dfe", "cdr_slicer"]


def test_collect_impulses_empty_in_phase0():
    # All blocks are passthrough stubs -> no concatenable impulses yet.
    assert _pipe().collect_impulses() == []


def test_init_states_covers_all_blocks():
    pipe = _pipe()
    states = pipe.init_states()
    assert set(states) == set(pipe.names())


def test_apply_params_routes_by_kind():
    pipe = _pipe()
    assert pipe.apply_params({"ctle": {"fz": 0.6}}) == {Kind.LTI}
    assert pipe.apply_params({"dfe": {"h1": 5.0}}) == {Kind.NONLINEAR}
    assert pipe.apply_params({"dfe": {"n_taps": 4}}) == {Kind.STRUCTURAL}
    # Dual-nature param (jitter) routes to BOTH the worker and the stat engine.
    assert pipe.apply_params({"txjitter": {"rj_mui": 10.0}}) == {
        Kind.NONLINEAR,
        Kind.LTI,
    }


def test_apply_params_actually_sets_values():
    pipe = _pipe()
    pipe.apply_params({"ctle": {"fz": 0.6}})
    assert pipe.by_name("ctle").get("fz") == 0.6


def test_empty_pipeline_instantiates():
    pipe = Pipeline()
    assert pipe.blocks == []
    assert pipe.lti_prefix() == [] and pipe.nonlinear_tail() == []
