# Path: train_schedulexpert_improved.py
import torch
import random
import json
import os
import sys
import wandb
import torch.nn.functional as F

from argparse import ArgumentParser
from tqdm import tqdm

from architectures.SchedulExpert_improved import GATEncoder, MHADecoder
from utils.inout import load_dataset
from utils.sampling import JobShopStates
from utils.utils import *  # Logger, AverageMeter, ObjMeter, TimeObjMeter, etc.

from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau


device = 'cuda' if torch.cuda.is_available() else 'cpu'
PROBE_EVERY = 100
LOGGING_STEPS = 5


def get_grad_norm(model):
    total = 0.0
    for p in model.parameters():
        if p.grad is not None:
            n = p.grad.data.norm(2).item()
            total += n * n
    return total ** 0.5


# --- add this near your imports ---
import socket
import datetime
import platform

def _to_builtin(o):
    if isinstance(o, (str, int, float, bool)) or o is None:
        return o
    if isinstance(o, (list, tuple)):
        return [_to_builtin(x) for x in o]
    if isinstance(o, dict):
        return {str(k): _to_builtin(v) for k, v in o.items()}
    # Fallback for enums/numpy types/Namespaces/etc.
    try:
        return _to_builtin(vars(o))
    except Exception:
        return str(o)

def dump_hparams(save_dir, args, encoder=None, decoder=None, extra: dict | None = None):
    """
    Writes a complete hyperparameter/config snapshot into the checkpoint folder.
    - args: argparse.Namespace (or anything dict-like)
    - extra: any extra info (wandb run id, seed, commit hash, etc.)
    - also records environment & model param counts
    """
    os.makedirs(save_dir, exist_ok=True)

    # Prefer wandb.config if present (sweeps may override args)
    try:
        cfg = dict(wandb.config) if wandb.run is not None else vars(args)
    except Exception:
        cfg = vars(args)

    meta = {
        "timestamp": datetime.datetime.now().isoformat(),
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": getattr(torch.version, "cuda", None),
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "wandb_run_id": (wandb.run.id if wandb.run is not None else None),
        "wandb_run_name": (wandb.run.name if wandb.run is not None else None),
        "wandb_project": (wandb.run.project if wandb.run is not None else None),
    }

    if encoder is not None:
        meta["encoder_num_params"] = sum(p.numel() for p in encoder.parameters())
        meta["encoder_trainable_params"] = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        meta["encoder_out_size"] = getattr(encoder, "out_size", None)
    if decoder is not None:
        meta["decoder_num_params"] = sum(p.numel() for p in decoder.parameters())
        meta["decoder_trainable_params"] = sum(p.numel() for p in decoder.parameters() if p.requires_grad)

    if extra:
        meta.update(extra)

    payload = {
        "hparams": _to_builtin(cfg),
        "meta": _to_builtin(meta),
    }

    # main JSON
    hp_path = os.path.join(save_dir, "hparams.json")
    with open(hp_path, "w") as f:
        json.dump(payload, f, indent=2)

    # optional human-friendly txt
    with open(os.path.join(save_dir, "hparams.txt"), "w") as f:
        f.write("# === Hyperparameters ===\n")
        for k, v in sorted(payload["hparams"].items()):
            f.write(f"{k}: {v}\n")
        f.write("\n# === Meta ===\n")
        for k, v in sorted(payload["meta"].items()):
            f.write(f"{k}: {v}\n")

    print(f"[hparams] dumped to {hp_path}")



# ---------------------------
# Sampling helpers (train/eval)
# ---------------------------
def _sample_rollout(ins: dict,
                    encoder: torch.nn.Module,
                    decoder: torch.nn.Module,
                    bs: int,
                    device: str,
                    temp: float = 1.0):
    """
    One rollout batch with multinomial sampling (used by train/eval).
    Returns:
      trajs: (bs, T) long
      ptrs:  (bs, T, num_jobs) logits
      mss:   (bs,) makespans
    """
    encoder.train(False)  # eval speed for encoding (we don't backprop through sampling)
    decoder.train(False)

    num_j, num_m = ins['j'], ins['m']
    num_ops = num_j * num_m - 1

    trajs = -torch.ones((bs, num_ops), dtype=torch.long, device=device)
    ptrs = -torch.ones((bs, num_ops, num_j), dtype=torch.float32, device=device)

    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, bs)

    # Encode once
    embed = encoder(ins['x'].to(device), edge_index=ins['edge_index'].to(device))

    for t in range(num_ops):
        ops = jsp.ops
        logits = decoder(embed[ops], state)
        logits = logits + mask.log()               # mask invalid jobs
        if temp != 1.0:
            logits = logits / temp
        probs = F.softmax(logits, dim=-1)
        jobs = probs.multinomial(1, replacement=False).squeeze(1)
        trajs[:, t] = jobs
        ptrs[:, t] = logits
        state, mask = jsp.update(jobs)

    # schedule last op
    jsp(mask.float().argmax(-1), state)
    return trajs, ptrs, jsp.makespan


def _sample_training(ins: dict,
                     encoder: torch.nn.Module,
                     decoder: torch.nn.Module,
                     bs: int,
                     device: str,
                     temp: float = 1.0):
    """
    Training-time sampling WITH gradients through the decoder logits only,
    not through the encoder (we re-encode in train mode for dropout etc.).
    """
    encoder.train()
    decoder.train()

    num_j, num_m = ins['j'], ins['m']
    num_ops = num_j * num_m - 1

    trajs = -torch.ones((bs, num_ops), dtype=torch.long, device=device)
    ptrs = -torch.ones((bs, num_ops, num_j), dtype=torch.float32, device=device)

    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, bs)

    embed = encoder(ins['x'].to(device), edge_index=ins['edge_index'].to(device))

    for t in range(num_ops):
        ops = jsp.ops
        logits = decoder(embed[ops], state)
        logits = logits + mask.log()
        if temp != 1.0:
            logits = logits / temp
        probs = F.softmax(logits, dim=-1)
        jobs = probs.multinomial(1, replacement=False).squeeze(1)
        trajs[:, t] = jobs
        ptrs[:, t] = logits
        state, mask = jsp.update(jobs)

    jsp(mask.float().argmax(-1), state)
    return trajs, ptrs, jsp.makespan


@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               val_set: list,
               num_sols: int = 16,
               seed: int = 42,
               temp: float = 1.0):
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()
    gaps = ObjMeter()

    for ins in tqdm(val_set):
        trajs, ptrs, mss = _sample_rollout(ins, encoder, decoder, bs=num_sols, device=device, temp=temp)
        min_gap = (mss.min().item() / ins['makespan'] - 1) * 100
        gaps.update(ins, min_gap)

    avg_gap = gaps.avg
    print(f"\t\tVal set: AVG Gap={avg_gap:.3f}")
    print(gaps)
    return avg_gap


# ---------------------------
# Training
# ---------------------------
def train(encoder: torch.nn.Module,
          decoder: torch.nn.Module,
          train_set: list,
          val_set: list,
          args):
    # W&B
    def b(x: bool) -> str:
        return "1" if x else "0"
    run_name = (
        f"SX_impr"
        f"-BS{args.bs}"
        f"-Beta{args.beta}"
        f"-L{args.enc_layers}"
        f"-H{args.enc_heads}"
        f"-JK{args.enc_jk}"                    # <-- include enc_jk
        f"-DPE{b(args.use_degree_pe)}"         # <-- include use_degree_pe
        f"-FILM{b(args.use_film)}"             # <-- include use_film
        f"-GAttn{b(args.use_dec_global_attn)}"
        f"-MOE{b(args.use_moe)}x{args.n_experts}"
    )
    save_dir = f"checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    wandb.init(
        project="schedulexpert-improved",
        config=vars(args),
        name=run_name
    )

    save_dir = f"checkpoints/{run_name}"
    os.makedirs(save_dir, exist_ok=True)

    wandb.init(
        project="schedulexpert-improved",
        config=vars(args),
        name=run_name
    )

    dump_hparams(save_dir, args, encoder, decoder, extra={"phase": "start"})

    wandb.watch(encoder, log='all', log_freq=200)
    wandb.watch(decoder, log='all', log_freq=200)

    # Optimizer & scheduler
    params = list(encoder.parameters()) + list(decoder.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    if args.scheduler_type == "cosine":
        sched = CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.scheduler_eta_min)
    elif args.scheduler_type == "reduce":
        sched = ReduceLROnPlateau(opt, mode='min', factor=0.5, patience=5, min_lr=args.scheduler_eta_min)
    else:
        sched = None

    # Loss objects
    ce_loss = torch.nn.CrossEntropyLoss(reduction='mean', label_smoothing=args.label_smoothing)

    scaler = torch.cuda.amp.GradScaler(enabled=(args.amp and device == 'cuda'))

    best_val = None
    frac = 1.0 / args.bs
    size = len(train_set)
    indices = list(range(size))

    logger = Logger(run_name)
    print("Training ...")

    for epoch in range(args.epochs):
        losses = AverageMeter()
        gaps = ObjMeter()
        random.shuffle(indices)
        cnt = 0

        for idx, i in tqdm(enumerate(indices), total=size):
            ins = train_set[i]

            # Sample multiple solutions
            with torch.cuda.amp.autocast(enabled=(args.amp and device == 'cuda')):
                trajs, logits, mss = _sample_training(
                    ins, encoder, decoder, bs=args.beta, device=device, temp=args.sample_temp
                )

                # Pick best among sampled solutions
                ms, argmin = mss.min(-1)
                logits_best = logits[argmin]      # (T, J)
                trajs_best = trajs[argmin]        # (T,)

                # Cross-entropy over the full sequence (pointer-style)
                loss = ce_loss(logits_best, trajs_best)

                # Entropy encouragement (maximize entropy -> subtract)
                if args.entropy_w > 0.0:
                    probs = F.softmax(logits_best, dim=-1)
                    ent = -(probs * (probs.clamp_min(1e-9)).log()).sum(-1).mean()
                    loss = loss - args.entropy_w * ent

                # MoE aux loss (if any)
                if getattr(encoder, "aux_loss", 0.0) != 0.0 and args.moe_aux_w > 0.0:
                    loss = loss + args.moe_aux_w * encoder.aux_loss

            losses.update(loss.item())
            gaps.update(ins, (ms.item() / ins['makespan'] - 1) * 100)

            # Virtual batch accumulation
            scaler.scale(loss * frac).backward()

            cnt += 1
            if cnt == args.bs or idx + 1 == size:
                # unscale & clip ONCE per optimizer step (after accumulation)
                if args.grad_clip > 0:
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(params, args.grad_clip)

                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                cnt = 0

            # Periodic logs
            if (idx + 1) % LOGGING_STEPS == 0:
                wandb.log({
                    "epoch": epoch,
                    "train_loss_avg": losses.avg,
                    "train_gap_avg": gaps.avg,
                    "enc_grad_norm": get_grad_norm(encoder),
                    "dec_grad_norm": get_grad_norm(decoder)
                })

            # Probe
            if idx > 0 and idx % PROBE_EVERY == 0:
                print(f'\tSTEP {idx:02}: avg loss={losses.avg:.4f}')
                print(f"\t\tTrain: AVG Gap={gaps.avg:2.3f}")
                print(gaps)

                val_gap = validation(encoder, decoder, val_set, num_sols=128, temp=args.sample_temp)
                wandb.log({"validation_gap": val_gap, "epoch": epoch})

                if best_val is None or val_gap < best_val:
                    best_val = val_gap
                    torch.save((encoder.state_dict(), decoder.state_dict()),
                               os.path.join(save_dir, f"{run_name}.pt"))

                if sched is not None:
                    if args.scheduler_type == "reduce":
                        sched.step(val_gap)
                    else:
                        sched.step()

        # End epoch logs
        val_gap = validation(encoder, decoder, val_set, num_sols=128, temp=args.sample_temp)
        wandb.log({
            "epoch": epoch,
            "train_loss_epoch": losses.avg,
            "train_gap_epoch": gaps.avg,
            "validation_gap": val_gap
        })

        if best_val is None or val_gap < best_val:
            best_val = val_gap
            torch.save((encoder.state_dict(), decoder.state_dict()),
                       os.path.join(save_dir, f"{run_name}.pt"))

        if sched is not None:
            if args.scheduler_type == "reduce":
                sched.step(val_gap)
            else:
                sched.step()

        # log current LR
        if sched is not None and args.scheduler_type != "reduce":
            lr_now = sched.get_last_lr()[0]
        else:
            lr_now = opt.param_groups[0]['lr']
        wandb.log({"learning_rate": lr_now, "epoch": epoch})

        logger.train(epoch, losses.avg, gaps.avg)
        logger.validation(val_gap)
        logger.flush()

    wandb.finish()


# ---------------------------
# Argparse
# ---------------------------
def build_parser():
    p = ArgumentParser(description="SchedulExpert Improved")
    # Data
    p.add_argument("-data_path", type=str,
                   default="/home/lamsade/habgaryan/phd_project/agg_slj/SelfLabelingJobShop/dataset_starjob_for_slj")
    p.add_argument("-val_path", type=str, default="./benchmarks/TA")

    # Core training
    p.add_argument("-epochs", type=int, default=20)
    p.add_argument("-lr", type=float, default=2e-4)
    p.add_argument("-wd", type=float, default=0.0)
    p.add_argument("-bs", type=int, default=16, help="virtual batch size (num updates per optimizer step)")
    p.add_argument("-beta", type=int, default=8, help="solutions sampled per instance")
    p.add_argument("-scheduler_type", type=str, default="None", choices=["None", "cosine", "reduce"])
    p.add_argument("-scheduler_eta_min", type=float, default=1e-6)

    # Encoder (improvements)
    p.add_argument("-enc_hidden", type=int, default=64)
    p.add_argument("-enc_heads", type=int, default=4)
    p.add_argument("-enc_layers", type=int, default=3)
    p.add_argument("-enc_jk", type=str, default="none", choices=["none", "cat", "max", "lstm"])
    p.add_argument("-use_degree_pe", action="store_true", default=False)
    p.add_argument("-enc_act", type=str, default="gelu", choices=["relu", "gelu", "leaky"])
    p.add_argument("-enc_embed", type=int, default=128)

    # MoE
    p.add_argument("-use_moe", action="store_true", default=False)
    p.add_argument("-n_experts", type=int, default=4)
    p.add_argument("-expert_hidden", type=int, default=0, help="0 -> same as enc_embed")
    p.add_argument("-expert_dropout", type=float, default=0.0)
    p.add_argument("-moe_capacity", type=float, default=1.25)
    p.add_argument("-moe_aux_w", type=float, default=0.0)

    # Decoder (improvements)
    p.add_argument("-mem_hidden", type=int, default=64)
    p.add_argument("-mem_out", type=int, default=128)
    p.add_argument("-clf_hidden", type=int, default=128)
    p.add_argument("-dec_heads", type=int, default=4)
    p.add_argument("-attn_dropout", type=float, default=0.1)
    p.add_argument("-mlp_dropout", type=float, default=0.1)
    p.add_argument("-use_film", action="store_true", default=False)
    p.add_argument("-use_dec_global_attn", action="store_true", default=False)

    # Training niceties
    p.add_argument("-amp", action="store_true", default=True)
    p.add_argument("-grad_clip", type=float, default=1.0)
    p.add_argument("-label_smoothing", type=float, default=0.0)
    p.add_argument("-entropy_w", type=float, default=0.0)
    p.add_argument("-sample_temp", type=float, default=1.0)

    return p


if __name__ == "__main__":
    args = build_parser().parse_args()
    print(args)

    # Load data
    print(f"Using device: {device}")
    train_set = load_dataset(args.data_path)
    val_set = load_dataset(args.val_path, device=device)

    # Build model
    # Encoder out_size must be input_size + hidden_size
    input_size = train_set[0]['x'].shape[1]

    enc = GATEncoder(
        input_size=input_size,
        hidden_size=args.enc_hidden,
        n_heads=args.enc_heads,
        n_layers=args.enc_layers,
        enc_jk=args.enc_jk,
        use_degree_pe=args.use_degree_pe,
        enc_act=args.enc_act,
        enc_embed=args.enc_embed,
        use_moe=args.use_moe,
        n_experts=args.n_experts,
        expert_hidden=(None if args.expert_hidden == 0 else args.expert_hidden),
        expert_dropout=args.expert_dropout,
        moe_capacity=args.moe_capacity,
        attn_dropout=args.attn_dropout,
        mlp_dropout=args.mlp_dropout
    ).to(device)

    dec = MHADecoder(
        encoder_size=enc.out_size,
        context_size=JobShopStates.size,
        hidden_size=args.mem_hidden,
        mem_size=args.mem_out,
        clf_size=args.clf_hidden,
        n_heads=args.dec_heads,
        attn_dropout=args.attn_dropout,
        mlp_dropout=args.mlp_dropout,
        use_film=args.use_film,
        use_dec_global_attn=args.use_dec_global_attn
    ).to(device)
    print(enc)
    print(dec)

    # Train
    train(enc, dec, train_set, val_set, args)
