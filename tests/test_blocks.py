"""Every built-in block satisfies the Block protocol and its param schema."""

import pytest

import eyeq.blocks  # noqa: F401  (registers blocks)
from eyeq.core import Block, Param, SimContext, registry
from eyeq.core.context import REACH_PRESETS, Modulation

CTX = SimContext.from_data_rate(
    112.0, Modulation.PAM4, reach=REACH_PRESETS[("112G", "VSR")]
)

ALL_TYPES = [
    "Source", "TXFFE", "TXJitter", "Channel", "Noise",
    "CTLE", "RXFFE", "DFE", "CDRSlicer",
]


@pytest.mark.parametrize("type_name", ALL_TYPES)
def test_block_satisfies_protocol(type_name):
    block = registry.create(type_name)
    assert isinstance(block, Block)
    assert isinstance(block.name, str) and block.name
    assert block.params and all(isinstance(p, Param) for p in block.params)


@pytest.mark.parametrize("type_name", ALL_TYPES)
def test_stub_engine_hooks_are_passthrough(type_name):
    block = registry.create(type_name)
    state = block.init_state(CTX)
    sentinel = object()
    y, st = block.process(sentinel, state, CTX)
    assert y is sentinel and st is state
    assert block.impulse_response(CTX) is None  # Phase 0: all stubs


def test_numeric_param_clamps():
    ctle = registry.create("CTLE")
    ctle.set_params(dc_gain=999.0)  # max is 0.0 dB
    assert ctle.get("dc_gain") == 0.0
    ctle.set_params(dc_gain=-999.0)  # min is -20 dB
    assert ctle.get("dc_gain") == -20.0


def test_unknown_param_raises():
    with pytest.raises(KeyError):
        registry.create("CTLE").set_params(nope=1.0)


def test_choice_param_validates():
    ch = registry.create("Channel")
    ch.set_params(model="tl")
    assert ch.get("model") == "tl"
    with pytest.raises(ValueError):
        ch.set_params(model="bogus")


def test_lti_and_tail_flags():
    # The DFE marks the start of the nonlinear tail; the LTI blocks do not.
    assert registry.create("DFE").is_tail is True
    assert registry.create("TXFFE").is_tail is False
    assert registry.create("CTLE").is_lti is True
    assert registry.create("Source").is_lti is False  # stochastic


def test_constructor_overrides():
    ctle = registry.create("CTLE", fz=0.7, fp=1.2)
    assert ctle.get("fz") == 0.7 and ctle.get("fp") == 1.2
