from __future__ import annotations

# ==========================================================================
# Central channel policy for Terra-AID.

# Saved dataset YAML contains one channel list only: ``bands.included``.
# This module derives model-input channels and mask/QC channels in code so
# configs do not duplicate profile/input-channel data.
# ==========================================================================

MASK_ONLY_CHANNELS = {"SCL", "QA", "QA60", "QC", "VALID_MASK", "CLOUD_MASK"}
DEFAULT_BANDS = ["B2", "B3", "B4", "B8", "B11", "B12", "SCL", "NDVI", "BSI"]
DEFAULT_RGB = ["B4", "B3", "B2"]


def norm_channel_name(name) -> str:
    return str(name or "").strip().upper()


def is_mask_only_channel(name) -> bool:
    return norm_channel_name(name) in MASK_ONLY_CHANNELS


def unique_channels(channels) -> list[str]:
    out, seen = [], set()
    for ch in channels or []:
        label = str(ch).strip()
        if not label:
            continue
        key = norm_channel_name(label)
        if key not in seen:
            out.append(label)
            seen.add(key)
    return out


# --------------------------------------------------------------------
# Return channels physically requested/available for a dataset.
# --------------------------------------------------------------------
def get_download_bands(dataset_cfg_or_dict) -> list[str]:
    
    if dataset_cfg_or_dict is None:
        return []

    if isinstance(dataset_cfg_or_dict, dict):
        bands = (dataset_cfg_or_dict.get("bands") or {}).get("included", [])
    else:
        bands_obj = getattr(dataset_cfg_or_dict, "bands", None)
        bands = getattr(bands_obj, "included", []) if bands_obj is not None else []

    return unique_channels(bands)


# --------------------------------------------------------------------
# Split one physical channel list into model inputs and mask/QC channels.
# --------------------------------------------------------------------
def split_input_and_mask_channels(channels, allow_categorical: bool = False):
    
    input_channels, mask_channels = [], []
    seen_input, seen_mask = set(), set()

    for ch in unique_channels(channels):
        key = norm_channel_name(ch)
        if is_mask_only_channel(key) and not allow_categorical:
            if key not in seen_mask:
                mask_channels.append(key)
                seen_mask.add(key)
        else:
            if key not in seen_input:
                input_channels.append(ch)
                seen_input.add(key)

    return input_channels, mask_channels


def _allow_categorical(cfg) -> bool:
    if cfg is None:
        return False
    if isinstance(cfg, dict):
        return bool((cfg.get("processing") or {}).get("allow_categorical_inputs", False))
    return bool(getattr(getattr(cfg, "processing", None), "allow_categorical_inputs", False))


def get_model_input_channels(dataset_cfg_or_dict) -> list[str]:
    channels = get_download_bands(dataset_cfg_or_dict)
    inputs, _ = split_input_and_mask_channels(channels, allow_categorical=_allow_categorical(dataset_cfg_or_dict))
    return inputs


def get_mask_channels(dataset_cfg_or_dict) -> list[str]:
    channels = get_download_bands(dataset_cfg_or_dict)
    _, masks = split_input_and_mask_channels(channels, allow_categorical=_allow_categorical(dataset_cfg_or_dict))
    return masks


def get_num_model_channels(dataset_cfg_or_dict) -> int:
    return len(get_model_input_channels(dataset_cfg_or_dict))


def get_rgb_channels(dataset_cfg_or_dict) -> list[str]:
    if dataset_cfg_or_dict is None:
        return list(DEFAULT_RGB)
    if isinstance(dataset_cfg_or_dict, dict):
        order = (dataset_cfg_or_dict.get("bands") or {}).get("rgb_order", [])
    else:
        bands = getattr(dataset_cfg_or_dict, "bands", None)
        order = getattr(bands, "rgb_order", []) if bands is not None else []
    return list(order or DEFAULT_RGB)


# --------------------------------------------------------------------
# Compatibility object for old code. Do not save this to YAML.
# --------------------------------------------------------------------

def make_runtime_profile(dataset_cfg_or_dict):
    
    from types import SimpleNamespace
    inputs = get_model_input_channels(dataset_cfg_or_dict)
    masks = get_mask_channels(dataset_cfg_or_dict)
    return SimpleNamespace(
        name=f"derived_{len(inputs)}ch",
        num_input_channels=len(inputs),
        input_channels=inputs,
        mask_channels=masks,
    )
