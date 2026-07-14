"""Torch-free tests for adapter-site selection (the dims axis)."""
import pytest

from sequential_adapt.config import SITE_GROUPS, resolve_site_suffixes


def test_default_is_both_all_layers():
    assert resolve_site_suffixes() == ("attn.c_attn", "mlp.c_fc")
    assert resolve_site_suffixes("both") == SITE_GROUPS["both"]


def test_site_groups():
    assert resolve_site_suffixes("attn") == ("attn.c_attn",)
    assert resolve_site_suffixes("mlp") == ("mlp.c_fc",)


def test_layer_range():
    assert resolve_site_suffixes("both", "0-2") == (
        "h.0.attn.c_attn", "h.0.mlp.c_fc",
        "h.1.attn.c_attn", "h.1.mlp.c_fc",
        "h.2.attn.c_attn", "h.2.mlp.c_fc",
    )


def test_layer_list_sorted_deduped():
    assert resolve_site_suffixes("attn", "5,3,3") == (
        "h.3.attn.c_attn", "h.5.attn.c_attn")
    assert resolve_site_suffixes("mlp", "0,1-2") == (
        "h.0.mlp.c_fc", "h.1.mlp.c_fc", "h.2.mlp.c_fc")


def test_layer_suffix_disambiguates_double_digits():
    # endswith matching in attach_adapter_sites: "h.0..." must not
    # capture layer 10 in deeper models.
    assert not "transformer.h.10.attn.c_attn".endswith("h.0.attn.c_attn")
    assert "transformer.h.0.attn.c_attn".endswith("h.0.attn.c_attn")


def test_invalid_specs_raise():
    with pytest.raises(ValueError):
        resolve_site_suffixes("qkv")
    with pytest.raises(ValueError):
        resolve_site_suffixes("both", "2-0")
    with pytest.raises(ValueError):
        resolve_site_suffixes("both", ",")
    with pytest.raises(ValueError):
        resolve_site_suffixes("both", "-1")
