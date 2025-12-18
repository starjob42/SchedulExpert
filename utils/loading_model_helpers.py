from typing import Dict, List, Tuple, Optional
import torch
import glob
import json
import os
from utils.sampling import JobShopStates
from dataclasses import dataclass

# -------------------------
# Device
# -------------------------
DEV = 'cuda' if torch.cuda.is_available() else 'cpu'


# -------------------------
# Checkpoint discovery / JSON loading
# -------------------------
@dataclass()
class CheckpointSpec:
    dir: str
    pt_path: str
    json_path: str
    json_type: str  # 'improved' (hparams.json with {'hparams':...}) or 'original' ( *_arguments.json ) or 'other'
    hparams: Dict


def _pick_pt(ckpt_dir: str, index: Optional[int]) -> str:
    pts = sorted(glob.glob(os.path.join(ckpt_dir, "*.pt")))
    if not pts:
        raise FileNotFoundError(f"No .pt files found in {ckpt_dir}")
    if index is None:
        # pick the newest by modification time
        pts.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return pts[0]
    if index < 0 or index >= len(pts):
        raise IndexError(f"--pt_index {index} out of range (found {len(pts)} .pt files).")
    return pts[index]


def _find_json(ckpt_dir: str) -> Tuple[str, str]:
    """
    Returns (path, json_type)
      json_type in {'improved', 'original', 'other'}
    """
    # Preferred for improved
    improved = os.path.join(ckpt_dir, "hparams.json")
    if os.path.isfile(improved):
        return improved, "improved"

    # Original training: *_arguments.json
    args_json = sorted(glob.glob(os.path.join(ckpt_dir, "*_arguments.json")))
    if args_json:
        return args_json[0], "original"

    # Last resort: any .json
    any_json = sorted(glob.glob(os.path.join(ckpt_dir, "*.json")))
    if any_json:
        return any_json[0], "other"

    raise FileNotFoundError(f"No JSON with hyperparameters found in {ckpt_dir}")


def _load_hparams(json_path: str, json_type: str) -> Dict:
    with open(json_path, "r") as f:
        data = json.load(f)

    if json_type == "improved":
        # expected payload: {'hparams': {...}, 'meta': {...}}
        if isinstance(data, dict) and "hparams" in data:
            return dict(data["hparams"])
        # if it’s already flat, just return it
        return dict(data)

    if json_type in ("original", "other"):
        # expected flat dict of CLI args
        if isinstance(data, dict):
            return dict(data)
        raise ValueError(f"Unexpected JSON structure in {json_path}")

    return dict(data)


def discover_checkpoint(ckpt_dir: str, pt_index: Optional[int]) -> CheckpointSpec:
    pt_path   = _pick_pt(ckpt_dir, pt_index)
    json_path, jtype = _find_json(ckpt_dir)
    hparams   = _load_hparams(json_path, jtype)
    return CheckpointSpec(ckpt_dir, pt_path, json_path, jtype, hparams)


# -------------------------
# Architecture detection + builder
# -------------------------
def _is_improved(hp: Dict) -> bool:
    """
    Heuristics:
      Improved training exposes keys like enc_layers/enc_heads/enc_jk/enc_embed, etc.
      Original exposes enc_out but no enc_layers.
    """
    if any(k in hp for k in ("enc_layers", "enc_heads", "enc_jk", "enc_embed", "use_degree_pe", "use_film")):
        return True
    if "enc_out" in hp and "enc_layers" not in hp:
        return False
    # Default to improved if ambiguous (safer: supports many flags; decoder heads exist there)
    return True


def build_models(hp: Dict, input_size: int):
    """
    Construct encoder/decoder using hyperparams from JSON.
    Returns (encoder, decoder, arch_name)
    """
    if _is_improved(hp):
        from architectures.SchedulExpert_improved import GATEncoder, MHADecoder

        enc = GATEncoder(
            input_size=input_size,
            hidden_size=int(hp.get("enc_hidden", 64)),
            n_heads=int(hp.get("enc_heads", 4)),
            n_layers=int(hp.get("enc_layers", 3)),
            enc_jk=str(hp.get("enc_jk", "none")),
            use_degree_pe=bool(hp.get("use_degree_pe", False)),
            enc_act=str(hp.get("enc_act", "gelu")),
            enc_embed=int(hp.get("enc_embed", 128)),
            use_moe=bool(hp.get("use_moe", False)),
            n_experts=int(hp.get("n_experts", 4)),
            expert_hidden=(None if int(hp.get("expert_hidden", 0)) == 0 else int(hp.get("expert_hidden", 128))),
            expert_dropout=float(hp.get("expert_dropout", 0.0)),
            moe_capacity=float(hp.get("moe_capacity", 1.25)),
            attn_dropout=float(hp.get("attn_dropout", 0.1)),
            mlp_dropout=float(hp.get("mlp_dropout", 0.1)),
        ).to(DEV)

        dec = MHADecoder(
            encoder_size=enc.out_size,
            context_size=JobShopStates.size,
            hidden_size=int(hp.get("mem_hidden", 64)),
            mem_size=int(hp.get("mem_out", 128)),
            clf_size=int(hp.get("clf_hidden", 128)),
            n_heads=int(hp.get("dec_heads", 4)),
            attn_dropout=float(hp.get("attn_dropout", 0.1)),
            mlp_dropout=float(hp.get("mlp_dropout", 0.1)),
            use_film=bool(hp.get("use_film", False)),
            use_dec_global_attn=bool(hp.get("use_dec_global_attn", False)),
        ).to(DEV)

        return enc, dec, "improved"

    else:
        from architectures.SchedulExpert import GATEncoder, MHADecoder

        enc = GATEncoder(
            input_size=input_size,
            hidden_size=int(hp.get("enc_hidden", 64)),
            embed_size=int(hp.get("enc_out", 128)),
            n_experts=int(hp.get("n_experts", 4)),
        ).to(DEV)

        dec = MHADecoder(
            encoder_size=enc.out_size,
            context_size=JobShopStates.size,
            hidden_size=int(hp.get("mem_hidden", 64)),
            mem_size=int(hp.get("mem_out", 64)),
            clf_size=int(hp.get("clf_hidden", 64)),
            n_heads=int(hp.get("dec_heads", 3)),  # original default 3
        ).to(DEV)

        return enc, dec, "original"


def load_weights(enc: torch.nn.Module, dec: torch.nn.Module, pt_path: str) -> None:
    ckpt = torch.load(pt_path, map_location=DEV)
    if isinstance(ckpt, dict):
        if "encoder" in ckpt and "decoder" in ckpt:
            enc.load_state_dict(ckpt["encoder"])
            dec.load_state_dict(ckpt["decoder"])
        else:
            # some older runs saved {'state_dict':...} etc.; try common fallbacks
            if "model" in ckpt and isinstance(ckpt["model"], (list, tuple)) and len(ckpt["model"]) == 2:
                enc.load_state_dict(ckpt["model"][0])
                dec.load_state_dict(ckpt["model"][1])
            else:
                raise KeyError(f"Dict checkpoint missing 'encoder'/'decoder' keys: {pt_path}")
    elif isinstance(ckpt, (list, tuple)) and len(ckpt) == 2:
        enc.load_state_dict(ckpt[0])
        dec.load_state_dict(ckpt[1])
    else:
        raise ValueError(f"Unrecognized checkpoint format at {pt_path}")

