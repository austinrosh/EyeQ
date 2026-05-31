"""The block registry knows every built-in block and rejects unknown types."""

import pytest

import eyeq.blocks  # noqa: F401  (registers blocks)
from eyeq.core import BlockBase, registry

EXPECTED = {
    "Source", "TXFFE", "TXJitter", "Channel", "Noise",
    "CTLE", "RXFFE", "DFE", "CDRSlicer",
}


def test_all_blocks_registered():
    assert set(registry.known_types()) == EXPECTED


def test_create_returns_block():
    block = registry.create("CTLE")
    assert isinstance(block, BlockBase)
    assert block.type_name == "CTLE"


def test_unknown_type_raises():
    with pytest.raises(KeyError):
        registry.create("DoesNotExist")
