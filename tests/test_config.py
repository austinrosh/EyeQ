"""Config round-trip (YAML + JSON) and registry-driven construction."""

import pytest

from eyeq.core import CANONICAL_ORDER
from eyeq.core.context import Modulation
from eyeq.io import build_context, build_pipeline, default_link_config, load, save
from eyeq.io.config import from_dict, to_dict


def test_dict_round_trip():
    cfg = default_link_config()
    assert from_dict(to_dict(cfg)) == cfg


@pytest.mark.parametrize("ext", [".yaml", ".json"])
def test_file_round_trip(tmp_path, ext):
    cfg = default_link_config(modulation="NRZ", reach_class="LR")
    path = tmp_path / f"link{ext}"
    save(cfg, path)
    assert load(path) == cfg


def test_build_context_pam4_vs_nrz():
    pam4 = build_context(default_link_config(modulation="PAM4", reach_class="VSR"))
    nrz = build_context(default_link_config(modulation="NRZ", reach_class="VSR"))
    assert pam4.mod is Modulation.PAM4 and pam4.f_nyq == pytest.approx(28e9)
    assert nrz.mod is Modulation.NRZ and nrz.f_nyq == pytest.approx(56e9)
    assert pam4.reach.name == "VSR"


def test_build_pipeline_is_canonical():
    pipe = build_pipeline(default_link_config())
    assert pipe.names() == CANONICAL_ORDER


def test_package_flag_syncs_to_channel():
    cfg = default_link_config()
    cfg.package = True
    pipe = build_pipeline(cfg)
    assert pipe.by_name("channel").get("package") == "on"
    assert pipe.by_name("channel").get("reach") == "VSR"


def test_unknown_modulation_raises():
    cfg = default_link_config()
    cfg.modulation = "PAM8"
    with pytest.raises(ValueError):
        build_context(cfg)


def test_unknown_reach_raises():
    cfg = default_link_config()
    cfg.reach_class = "ULTRA"
    with pytest.raises(ValueError):
        build_context(cfg)


def test_version_mismatch_rejected(tmp_path):
    cfg = default_link_config()
    path = tmp_path / "bad.yaml"
    save(cfg, path)
    text = path.read_text().replace("version: 1", "version: 99")
    path.write_text(text)
    with pytest.raises(ValueError):
        load(path)
