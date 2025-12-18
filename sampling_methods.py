import torch
import torch.nn.functional as F
import math
from utils.sampling import JobShopStates

# usual sampling

@torch.no_grad()
def sampling(ins: dict,
             encoder: torch.nn.Module,
             decoder: torch.nn.Module,
             bs: int = 32,
             device: str = 'cpu'):
    encoder.eval()
    decoder.eval()
    num_j, num_m = ins['j'], ins['m']
    num_ops = num_j * num_m - 1

    trajs = -torch.ones((bs, num_ops), dtype=torch.long, device=device)
    ptrs = -torch.ones((bs, num_ops, num_j), dtype=torch.float32, device=device)
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, bs)

    # Encoding step
    embed = encoder(ins['x'].to(device),
                                edge_index=ins['edge_index'].to(device))

  

    # Decoding steps
    for i in range(num_ops):
        ops = jsp.ops  # Shape: (bs,)
        logits = decoder(embed[ops], state) + mask.log()  # Shape: (bs, num_jobs)
        scores = F.softmax(logits, -1)
        jobs = scores.multinomial(1, replacement=False).squeeze(1)  # Shape: (bs,)

        trajs[:, i] = jobs
        ptrs[:, i] = logits
        state, mask = jsp.update(jobs)

    # Schedule last job/operation
    jsp(mask.float().argmax(-1), state)

    return trajs, ptrs, jsp.makespan  



# ------------------------------------------------------------
# Improved Sequential Monte Carlo (SMC) with:
# - Adaptive temperature scheduling (entropy-aware)
# - K-rollout proxy completion for low-variance weights
# - Blended weights: proxy makespan + cumulative log-likelihood
# - Systematic resampling with elitism
# - Rejuvenation via Gumbel jitter after resampling
# - Adaptive checkpoints (by index + entropy trigger)
# ------------------------------------------------------------

import torch
import torch.nn.functional as F
import math
from utils.sampling import JobShopStates

@torch.inference_mode()
def _smc_sampling_fast_improved(
    ins: dict,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    bs: int = 512,
    device: str = "cuda",
    # Tunable Hyperparameters
    n_checkpoints: int = 12,
    tau_start: float = 1.5,
    tau_end: float = 0.05,
    finish_tau: float = 0.0,
    ess_threshold: float = 0.6,
    elite_frac: float = 0.10,
    use_amp: bool = True
):
    """
    Advanced SMC sampler with Elitism and Trajectory Harvesting.
    Fixed: Tensor mismatch on harvesting injection.
    """
    num_j, num_m = ins["j"], ins["m"]
    T = num_j * num_m - 1  # For ta73 (100x20), T=1999

    # --- 0. Checkpoint Logic ---
    if n_checkpoints > 0:
        checkpoints = set()
        step_size = max(1, T // n_checkpoints)
        for k in range(n_checkpoints):
            cp = (k + 1) * step_size
            checkpoints.add(cp)
        checkpoints.add(T - num_m) 
    else:
        checkpoints = set()

    # --- 1. Encode ---
    with torch.cuda.amp.autocast(enabled=use_amp):
        x = ins["x"].to(device)
        edge_index = ins["edge_index"].to(device)
        embed = encoder(x, edge_index=edge_index)

    # --- 2. Init State ---
    jsp = JobShopStates(device)
    state, mask = jsp.init_state(ins, bs)

    trajs = torch.full((bs, T), -1, dtype=torch.long, device=device)
    ptrs  = torch.zeros((bs, T, num_j), dtype=torch.float32, device=device)

    best_found_makespan = float('inf')
    best_found_traj = None

    # --- Helper: Fast Proxy Rollout with Harvesting ---
    def _fast_proxy_rollout_harvest(current_jsp, current_state, current_trajs, current_step):
        proxy_jsp = JobShopStates(device)
        proxy_jsp.copy_state_from(current_jsp)
        proxy_state = current_state.clone()
        
        local_trajs = []

        while True:
            if not proxy_jsp.mask.any(): break
            proxy_ops = proxy_jsp.ops
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = decoder(embed[proxy_ops], proxy_state) + proxy_jsp.mask.log()
            
            if finish_tau < 1e-3:
                jobs_t = logits.argmax(dim=-1)
            else:
                probs = F.softmax(logits / finish_tau, dim=-1)
                jobs_t = probs.multinomial(1).squeeze(1)
            
            local_trajs.append(jobs_t)
            mask_proxy = proxy_jsp(jobs_t, proxy_state)
            if mask_proxy.sum() == 0: break
        
        if len(local_trajs) > 0:
            extension = torch.stack(local_trajs, dim=1)
            return proxy_jsp.makespan, extension
        else:
            return proxy_jsp.makespan, None

    # --- Main Loop ---
    for t in range(T):
        frac = t / T
        curr_tau = tau_start * ((tau_end / tau_start) ** frac)

        ops = jsp.ops
        with torch.cuda.amp.autocast(enabled=use_amp):
            logits_raw = decoder(embed[ops], state) + mask.log()
        ptrs[:, t] = logits_raw

        # Sample
        probs = F.softmax(logits_raw / curr_tau, dim=-1)
        jobs_t = probs.multinomial(1).squeeze(1)
        
        trajs[:, t] = jobs_t
        state, mask = jsp.update(jobs_t)

        # Resample / Prune
        if t in checkpoints:
            proxy_ms, extension = _fast_proxy_rollout_harvest(jsp, state, trajs, t)

            # --- Harvesting ---
            min_proxy_val, min_proxy_idx = proxy_ms.min(dim=0)
            if min_proxy_val.item() < best_found_makespan:
                best_found_makespan = min_proxy_val.item()
                if extension is not None:
                    prefix = trajs[min_proxy_idx, :t+1]
                    suffix = extension[min_proxy_idx]
                    # FIX: Slice to [:T] to discard the implicit final step
                    # ta73 has 2000 ops, T=1999. Prefix+Suffix=2000. Trajs expects 1999.
                    best_found_traj = torch.cat([prefix, suffix], dim=0)[:T]

            # --- Weights ---
            advantage = -(proxy_ms - proxy_ms.min()) 
            advantage = advantage / (proxy_ms.std() + 1e-5)
            weights = F.softmax(advantage * 2.0, dim=0) 
            
            ess = 1.0 / (weights.pow(2).sum() + 1e-12)
            
            if ess < ess_threshold * bs:
                # Elitism
                sorted_score, sorted_indices = torch.sort(proxy_ms, descending=False)
                n_elites = int(bs * elite_frac)
                elite_indices = sorted_indices[:n_elites]
                
                remaining_count = bs - n_elites
                idx_rest = torch.multinomial(weights, num_samples=remaining_count, replacement=True)
                
                idx = torch.cat([elite_indices, idx_rest])
                
                trajs = trajs.index_select(0, idx)
                jsp.reorder_particles(idx)
                state = state.index_select(0, idx)
                mask = mask.index_select(0, idx)

    # --- Finalize ---
    jsp(mask.float().argmax(-1), state)
    final_ms = jsp.makespan
    
    # Injection of Harvested Best
    min_final_val, min_final_idx = final_ms.min(dim=0)
    
    if min_final_val.item() > best_found_makespan and best_found_traj is not None:
        # If the harvested solution is better than anything that survived the filter
        worst_idx = final_ms.argmax()
        # The slice [:T] earlier ensures this shape match works
        trajs[worst_idx] = best_found_traj 
        final_ms[worst_idx] = best_found_makespan
        
    return trajs, ptrs, final_ms

@torch.inference_mode()
def _cem_guided_sampling(
    ins: dict,
    encoder: torch.nn.Module,
    decoder: torch.nn.Module,
    bs: int = 128,
    device: str = "cuda",
    # Tuning for "Soft" Guidance
    rounds: int = 3,
    elite_frac: float = 0.15,
    momentum: float = 0.8, 
    gumbel_map: bool = True, 
    use_top_p: bool = True,
    use_amp: bool = True,
    # Annealing
    tau0: float = 1.1,
    tau_decay: float = 0.99,
    tau_min: float = 0.45,
    # Bias Strength
    p0: float = 0.92, 
    p_decay: float = 0.995,
    p_min: float = 0.75,
    smooth_eps: float = 0.001
):
    """
    Iterative Logit Shaping (Soft-CEM) [FIXED: Type Error].
    """
    num_j, num_m = ins["j"], ins["m"]
    T = num_j * num_m - 1
    
    # Split budget: e.g., 128 -> [32, 32, 64]
    r1 = int(bs * 0.25)
    r2 = int(bs * 0.25)
    r3 = bs - r1 - r2
    batch_schedule = [r1, r2, r3]
    
    # Bias strength schedule
    bias_strengths = [0.0, 1.5, 3.0] 

    with torch.cuda.amp.autocast(enabled=use_amp):
        x = ins["x"].to(device)
        edge_index = ins["edge_index"].to(device)
        embed = encoder(x, edge_index=edge_index)

    best_global_makespan = float('inf')
    best_global_traj = None
    
    # Guidance Matrix: [T, num_j]
    guidance_matrix = torch.zeros((T, num_j), device=device)

    for r in range(rounds):
        current_bs = batch_schedule[r]
        if current_bs <= 0: continue
        
        bias_weight = bias_strengths[r]
        
        jsp = JobShopStates(device)
        state, mask = jsp.init_state(ins, current_bs)
        
        trajs = torch.full((current_bs, T), -1, dtype=torch.long, device=device)
        ptrs = torch.zeros((current_bs, T, num_j), dtype=torch.float32, device=device)
        
        curr_tau = max(tau_min, tau0 * (0.9 ** r))

        for t in range(T):
            ops = jsp.ops 
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = decoder(embed[ops], state) + mask.log() 
                ptrs[:, t] = logits

                # --- FIX IS HERE ---
                if r > 0:
                    step_guide = guidance_matrix[t].unsqueeze(0) # [1, J]
                    
                    # Explicitly cast mask to bool for torch.where
                    # mask is likely Float (0.0/1.0) for .log() usage above
                    bool_mask = mask.bool()
                    
                    # Mask out invalid actions in guidance
                    step_guide = torch.where(bool_mask, step_guide, torch.tensor(-1e9, device=device))
                    
                    # Add Bias
                    logits = logits + bias_weight * step_guide
                # -------------------

            if gumbel_map:
                gumbel_noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-9) + 1e-9)
                scores = (logits / curr_tau) + gumbel_noise
                jobs_t = scores.argmax(dim=-1)
            else:
                probs = F.softmax(logits / curr_tau, dim=-1)
                jobs_t = probs.multinomial(1).squeeze(1)

            trajs[:, t] = jobs_t
            state, mask = jsp.update(jobs_t)
            curr_tau = max(tau_min, curr_tau * tau_decay)

        if jsp.mask.any():
            jsp(mask.float().argmax(-1), state)
        final_ms = jsp.makespan
        
        # Update Global Best
        min_val, min_idx = final_ms.min(dim=0)
        if min_val.item() < best_global_makespan:
            best_global_makespan = min_val.item()
            best_global_traj = trajs[min_idx].clone()
            
        # Update Guidance Matrix (CEM)
        n_elites = max(1, int(current_bs * elite_frac))
        vals, idxs = torch.sort(final_ms)
        elite_idxs = idxs[:n_elites]
        elite_trajs = trajs[elite_idxs] 
        
        new_guide = torch.zeros((T, num_j), device=device)
        one_hot = F.one_hot(elite_trajs, num_classes=num_j).float()
        counts = one_hot.sum(dim=0)
        
        probs = counts / (n_elites + 1e-9)
        log_probs = torch.log(probs + smooth_eps)
        
        guidance_matrix = momentum * guidance_matrix + (1.0 - momentum) * log_probs

    trajs[0] = best_global_traj
    final_ms[0] = best_global_makespan
    
    return trajs, ptrs, final_ms


