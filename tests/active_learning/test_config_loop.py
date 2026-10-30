import pytest
from omegaconf import OmegaConf

from src.active_learning.config import (
    AL_BLOCKS,
    LEGACY_BLOCKS,
    al_enabled,
    get_al_cfg,
    request_stop,
    resolve_blocks,
    resolve_start,
    should_stop,
    validate_al_cfg,
)


def make_cfg(al=None, blocks=None, start_at="split", **extra):
    cfg = {
        "pipeline": {"blocks": list(blocks or LEGACY_BLOCKS), "start_at": start_at},
        "loop": {"num_iterations": 5, "current_iter": 0},
        "EQdiffParam": {"num_samples": 20, "properties_list": ["HLgap", "eMBD"]},
        "exp_config_path": "/tmp/Experiments/exp/configs/experiment_config.yaml",
    }
    if al is not None:
        cfg["al"] = al
    cfg.update(extra)
    c = OmegaConf.create(cfg)
    OmegaConf.set_struct(c, True)
    return c


def test_old_configs_without_al_block_are_disabled():
    cfg = make_cfg()
    assert not al_enabled(cfg)
    assert get_al_cfg(cfg)["surrogate"]["props"] == ["HLgap", "eMBD"]


def test_resolve_blocks_modes():
    assert resolve_blocks(make_cfg()) == LEGACY_BLOCKS
    assert resolve_blocks(make_cfg(blocks=AL_BLOCKS)) == LEGACY_BLOCKS
    assert resolve_blocks(make_cfg(al={"enabled": True})) == AL_BLOCKS
    custom = ["split", "train", "generate", "prop_filter", "hdf5"]
    assert resolve_blocks(make_cfg(al={"enabled": True}, blocks=custom)) == ["split", "train", "generate", "hdf5"]


def test_resolve_start_does_not_skip_plan_when_enabled():
    on, off = make_cfg(al={"enabled": True}), make_cfg()
    assert resolve_start(on, AL_BLOCKS) == "plan"
    assert resolve_start(off, LEGACY_BLOCKS) == "split"
    assert resolve_start(make_cfg(al={"enabled": True}, start_at="train"), AL_BLOCKS) == "train"


def test_al_surrogate_props_interpolation_resolves():
    cfg = make_cfg(al={"enabled": True, "surrogate": {"props": "${EQdiffParam.properties_list}"}})
    assert get_al_cfg(cfg)["surrogate"]["props"] == ["HLgap", "eMBD"]


@pytest.mark.parametrize("bad, msg", [
    ({"targets_per_iter": 0}, "targets_per_iter"),
    ({"query_strategy": "magic"}, "query_strategy"),
    ({"budget": 1000}, "budget"),
    ({"oversample": 0}, "oversample"),
    ({"acquisition": {"mode": "x"}}, "acquisition.mode"),
])
def test_validation_errors(bad, msg):
    al = get_al_cfg(make_cfg(al={"enabled": True, **bad}))
    with pytest.raises(ValueError, match=msg):
        validate_al_cfg(al, 20, ["HLgap", "eMBD"])


def test_validation_rejects_unreadable_property_and_ignores_disabled():
    al = get_al_cfg(make_cfg(al={"enabled": True, "budget": 50}))
    with pytest.raises(ValueError, match="cannot be read"):
        validate_al_cfg(al, 20, ["HLgap", "mystery"])
    validate_al_cfg(get_al_cfg(make_cfg(al={"enabled": False, "targets_per_iter": 0})), 20, [])


def test_loop_stop_flag_on_struct_config():
    cfg = make_cfg()
    assert not should_stop(cfg)
    request_stop(cfg)
    assert should_stop(cfg) and cfg.loop.stop is True
