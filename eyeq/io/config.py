"""Config save/load and registry-driven pipeline construction.

A :class:`LinkConfig` is the on-disk truth for a full link setup. The top-level
fields describe the *scenario* (rate, modulation, reach, generation, seed) and
determine the :class:`~eyeq.core.context.SimContext`; ``blocks`` lists each block
in canonical order with its parameter overrides.

Reproducibility contract: a ``LinkConfig`` plus its ``rng_seed`` fully determines
every stochastic draw. The ``version`` field is present from day one so the
schema can be migrated.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Optional

import yaml

# Importing the blocks package registers every built-in block type.
from .. import blocks as _blocks  # noqa: F401
from ..core import registry
from ..core.context import REACH_PRESETS, Modulation, SimContext
from ..core.pipeline import CANONICAL_ORDER, Pipeline

CONFIG_VERSION = 1

# Maps canonical block name -> registry type name, in canonical order.
_NAME_TO_TYPE: dict[str, str] = {
    "source": "Source",
    "txffe": "TXFFE",
    "txjitter": "TXJitter",
    "channel": "Channel",
    "noise": "Noise",
    "ctle": "CTLE",
    "rxffe": "RXFFE",
    "dfe": "DFE",
    "cdr_slicer": "CDRSlicer",
}


@dataclass
class BlockConfig:
    """A block's registry ``type`` plus parameter overrides."""

    type: str
    params: dict[str, Any] = field(default_factory=dict)


@dataclass
class LinkConfig:
    """A complete, reproducible link setup."""

    version: int = CONFIG_VERSION
    data_rate_gbps: float = 112.0
    sps: int = 32
    modulation: str = "PAM4"
    generation: str = "112G"
    reach_class: str = "VSR"
    package: bool = False
    rng_seed: int = 0
    channel_s4p: Optional[str] = None
    blocks: list[BlockConfig] = field(default_factory=list)
    analysis: dict[str, Any] = field(default_factory=dict)
    fec: dict[str, Any] = field(default_factory=dict)
    detector: dict[str, Any] = field(default_factory=dict)
    ui: dict[str, Any] = field(default_factory=dict)


def default_analysis() -> dict[str, Any]:
    return {
        "avg_factor": 3,  # eye-density exponential-decay strength
        "v_bins": 256,
        "phase_points": 32,
    }


def default_fec() -> dict[str, Any]:
    """FEC analysis-layer settings (post-FEC BER estimate; see analysis/fec.py)."""
    return {
        "enabled": False,
        "scheme": "kp4",            # none | kp4 | kr4 | custom
        "target_post_ber": 1e-15,
        "error_model": "random",    # random (justified default) | bursty (coarse)
        "burst_len_bits": 1,
        "interleave_depth": 1,
        "custom_n": 544,
        "custom_k": 514,
        "custom_m": 10,
    }


def default_detector() -> dict[str, Any]:
    """Receiver detection mode. 'slicer'/'dfe' use the eye-tail BER; 'mlsd' uses the
    sequence-error (minimum-distance) estimate. ``mlsd_taps`` is the trellis memory."""
    return {"mode": "dfe", "mlsd_taps": 4}


def default_ui() -> dict[str, Any]:
    """Dashboard view settings (theme, eye colormap, density scale, amplitude mode).

    ``amp_mode`` 'fixed' frames the eye/histogram against ±swing/2 (so the eye breathes
    with loss); 'auto' fits to the data. ``track_swing`` true scales the fixed amplitude
    axis with the launch swing (eye/SBR stay the same size); false anchors it to the
    swing's full range so the eye/SBR visibly grow/shrink as swing changes.
    ``sbr_labels`` toggles the h-1/h0/h+1 cursor labels. ``density_scale`` 'log' makes
    low-density tails visible on the warm colormaps.
    """
    return {
        "theme": "dark",            # dark | light
        "eye_colormap": "turbo",    # turbo | jet | inferno | magma | plasma | hot | viridis | gray
        "density_scale": "log",     # log | linear
        "amp_mode": "fixed",        # fixed (±swing/2) | auto
        "track_swing": True,        # scale the fixed axis with swing | anchor to swing range
        "sbr_labels": True,         # show the SBR h-1/h0/h+1 cursor labels
    }


def default_link_config(
    *,
    data_rate_gbps: float = 112.0,
    modulation: str = "PAM4",
    generation: str = "112G",
    reach_class: str = "VSR",
    sps: int = 32,
    rng_seed: int = 0,
) -> LinkConfig:
    """A canonical 9-block link with default parameters for the given scenario."""
    cfg_blocks = [BlockConfig(type=_NAME_TO_TYPE[name]) for name in CANONICAL_ORDER]
    return LinkConfig(
        data_rate_gbps=data_rate_gbps,
        sps=sps,
        modulation=modulation,
        generation=generation,
        reach_class=reach_class,
        rng_seed=rng_seed,
        blocks=cfg_blocks,
        analysis=default_analysis(),
        fec=default_fec(),
        detector=default_detector(),
        ui=default_ui(),
    )


# --------------------------------------------------------------------------- #
# (de)serialization
# --------------------------------------------------------------------------- #
def to_dict(cfg: LinkConfig) -> dict[str, Any]:
    return asdict(cfg)


def from_dict(d: dict[str, Any]) -> LinkConfig:
    d = dict(d)
    raw_blocks = d.pop("blocks", []) or []
    blocks = [
        b if isinstance(b, BlockConfig)
        else BlockConfig(type=b["type"], params=dict(b.get("params", {})))
        for b in raw_blocks
    ]
    known = {f for f in LinkConfig.__dataclass_fields__}
    kwargs = {k: v for k, v in d.items() if k in known}
    return LinkConfig(blocks=blocks, **kwargs)


def save(cfg: LinkConfig, path: str | Path) -> None:
    """Write a config to YAML (.yaml/.yml) or JSON (.json) by extension."""
    path = Path(path)
    payload = to_dict(cfg)
    if path.suffix.lower() == ".json":
        text = json.dumps(payload, indent=2)
    else:
        text = yaml.safe_dump(payload, sort_keys=False)
    path.write_text(text)


def load(path: str | Path) -> LinkConfig:
    """Read a config from YAML or JSON by extension."""
    path = Path(path)
    text = path.read_text()
    if path.suffix.lower() == ".json":
        data = json.loads(text)
    else:
        data = yaml.safe_load(text)
    cfg = from_dict(data)
    if cfg.version != CONFIG_VERSION:
        # No migrations yet; surface the mismatch loudly rather than silently.
        raise ValueError(
            f"config version {cfg.version} != supported {CONFIG_VERSION}"
        )
    return cfg


# --------------------------------------------------------------------------- #
# construction
# --------------------------------------------------------------------------- #
def build_context(cfg: LinkConfig) -> SimContext:
    """Derive the immutable :class:`SimContext` from a config's scenario fields."""
    try:
        mod = Modulation[cfg.modulation]
    except KeyError:
        raise ValueError(
            f"unknown modulation {cfg.modulation!r}; expected one of "
            f"{[m.name for m in Modulation]}"
        ) from None
    key = (cfg.generation, cfg.reach_class)
    if key not in REACH_PRESETS:
        raise ValueError(
            f"unknown reach preset {key!r}; known: {sorted(REACH_PRESETS)}"
        )
    return SimContext.from_data_rate(
        cfg.data_rate_gbps,
        mod,
        reach=REACH_PRESETS[key],
        sps=cfg.sps,
        rng_seed=cfg.rng_seed,
    )


def build_pipeline(cfg: LinkConfig) -> Pipeline:
    """Construct the canonical pipeline from a config (registry-driven).

    The top-level scenario fields are authoritative: the channel block's
    ``reach``/``package`` params are synced from ``cfg`` so there is one source
    of truth for the reach class.
    """
    ctx = build_context(cfg)
    block_cfgs = cfg.blocks or default_link_config(
        data_rate_gbps=cfg.data_rate_gbps,
        modulation=cfg.modulation,
        generation=cfg.generation,
        reach_class=cfg.reach_class,
        sps=cfg.sps,
        rng_seed=cfg.rng_seed,
    ).blocks

    built = []
    for bc in block_cfgs:
        block = registry.create(bc.type, **bc.params)
        if block.name == "channel":
            block.set_params(reach=cfg.reach_class)
            if cfg.package:  # load the reach-class typical bump-to-bump package loss
                block.set_params(pkg_loss_db=REACH_PRESETS[(cfg.generation, cfg.reach_class)].pkg_db_nyq)
            if cfg.channel_s4p:
                block.set_touchstone(cfg.channel_s4p)
        built.append(block)
    return Pipeline(blocks=built, ctx=ctx)
