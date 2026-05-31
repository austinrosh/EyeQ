"""A name -> block-class registry for config-driven pipeline construction.

Blocks register themselves with ``@register("CTLE")``. The config loader looks
up a block by its ``type`` string and instantiates it. Importing
``eyeq.blocks`` triggers registration of every built-in block.
"""

from __future__ import annotations

from typing import Type

_REGISTRY: dict[str, Type] = {}


def register(type_name: str):
    """Class decorator registering a block under ``type_name``."""

    def _decorator(cls: Type) -> Type:
        if type_name in _REGISTRY and _REGISTRY[type_name] is not cls:
            raise ValueError(f"block type {type_name!r} already registered")
        _REGISTRY[type_name] = cls
        cls.type_name = type_name
        return cls

    return _decorator


def create(type_name: str, **params):
    """Instantiate a registered block with optional parameter overrides."""
    try:
        cls = _REGISTRY[type_name]
    except KeyError:
        raise KeyError(
            f"unknown block type {type_name!r}; known types: {known_types()}"
        ) from None
    return cls(**params)


def get(type_name: str) -> Type:
    return _REGISTRY[type_name]


def is_registered(type_name: str) -> bool:
    return type_name in _REGISTRY


def known_types() -> list[str]:
    return sorted(_REGISTRY)
