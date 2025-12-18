import argparse
import torch
import pandas as pd
import os
from utils.sampling import JobShopStates
from utils.inout import load_data
from utils.utils import ObjMeter
from utils.utils import TimeObjMeter
import torch.nn.functional as F
import json
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import seaborn as sns
import numpy as np
import glob
from tqdm import tqdm
import time
from test_schedulexpert_improved_hyper_search import _cem_guided_sampling

# Training device
dev = 'cuda' if torch.cuda.is_available() else 'cpu'



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



@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               ins,
               beta: int = 32,
               seed: int = None,
               collect_latent: bool = False):
    """
    Args:
        encoder: Encoder.
        decoder: Decoder.
        ins: JSP instance.
        beta: Number of solutions to generate for each instance.
        seed: Random seed.
        collect_latent: If True, return latent variables.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()

    st = time.time()
    if beta > 1:
        trajs, ptrs, mss = sampling(ins, encoder, decoder, bs=beta, device=dev)
        # trajs, ptrs, mss = _cem_guided_sampling(ins, encoder, decoder, bs=beta, device=dev,elite_frac=0.15,gumbel_map=True,momentum=0.8,p0=0.92,p_decay=0.995,p_min=0.75,rounds=3,smooth_eps=0.001,tau0=1.1,tau_decay=0.99,tau_min=0.45,use_top_p=True)
        # Identify the index of the best solution
        best_idx = torch.argmin(mss)
    else:
        trajs, ptrs, mss = sampling(ins, encoder, decoder, bs=1, device=dev)

    exe_time = time.time() - st

    _gaps = (mss / ins['makespan'] - 1) * 100
    min_gap = _gaps.min().item()
    print(f'\t- {ins["name"]} = {min_gap:.3f}%')
    results = {'NAME': ins['name'],
               'UB': ins['makespan'],
               'MS': mss.min().item(),
               'MS-AVG': mss.mean().item(),
               'MS-STD': mss.std().item(),
               'GAP': min_gap,
               'GAP-AVG': _gaps.mean().item(),
               'GAP-STD': _gaps.std().item(),
               'TIME': exe_time}

    if collect_latent:
        # Collect number of jobs and machines
        j = ins['j']
        m = ins['m']
        return results, j, m
    else:
        return results

def load_args(arguments_path):
    with open(arguments_path, 'r') as f:
        args_dict = json.load(f)
    return argparse.Namespace(**args_dict)

def find_files(folder_path):
    """
    Finds the first .pt and .json files in the given folder.

    Args:
        folder_path (str): Path to the folder.

    Returns:
        tuple: (model_path, arguments_path)
    """
    pt_files = glob.glob(os.path.join(folder_path, '*.pt'))
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    if len(pt_files) == 0:
        raise FileNotFoundError(f"No .pt file found in the folder: {folder_path}")
    elif len(pt_files) > 1:
        raise ValueError(f"Multiple .pt files found in the folder: {folder_path}. Please ensure only one exists.")

    if len(json_files) == 0:
        raise FileNotFoundError(f"No .json file found in the folder: {folder_path}")
    elif len(json_files) > 1:
        raise ValueError(f"Multiple .json files found in the folder: {folder_path}. Please ensure only one exists.")

    return pt_files[0], json_files[0]

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Test Pointer Net with Latent Visualization and Instance Size Filtering'
    )
    
    # Add the new folder_path argument
    parser.add_argument(
        "-folder_path", type=str, required=True,
        help="Path to the folder containing the model checkpoint (.pt) and arguments JSON file."
    )
    
    # Keep the rest of the arguments as they are
    parser.add_argument("-benchmark", type=str, required=False,
                        default='TA', help="Name of the benchmark for testing.")
    parser.add_argument("-beta", type=int, default=512, required=False,
                        help="Number of sampled solutions for each instance.")
    parser.add_argument("-seed", type=int, default=42,
                        required=False, help="Random seed.")
    parser.add_argument("-annotate", default=False, action='store_true',
                        help="Whether to annotate gaps of the instances in the latent space plot.")
    parser.add_argument("-output_plot", type=str, default='latent_space_images/latent_space.png',
                        help="Path to save the latent space plot.")
    parser.add_argument("-max_jobs", type=int, default=None,
                        help="Maximum number of jobs (j) to include in testing.")
    parser.add_argument("-max_machines", type=int, default=None,
                        help="Maximum number of machines (m) to include in testing.")
    parser.add_argument("-num_instances", type=int, default=None,
                        help="Number of instances to test.")
    parser.add_argument("-random", action='store_true',
                        help="Randomly select instances to test.")
    parser.add_argument("-infer_sch_expert", action='store_true',help="Use the scheduling expert model architecture.")
    args = parser.parse_args()
    print(f"Arguments: {args}")

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")



    if args.infer_sch_expert:
        from architectures.SchedulExpert import GATEncoder, MHADecoder
        # from architectures.SchedulExpert_improved import GATEncoder, MHADecoder

    else:
        print("New improved verison is not ready yet")
        



    # Find the .pt and .json files in the specified folder
    try:
        model_path, arguments_path = find_files(args.folder_path)
        print(f"Detected model checkpoint: {model_path}")
        print(f"Detected arguments file: {arguments_path}")
    except (FileNotFoundError, ValueError) as e:
        print(f"Error: {e}")
        sys.exit(1)

    # Load hyperparameters
    print(f"Loading hyperparameters from {arguments_path}")
    args_loaded = load_args(arguments_path)

    encoder = GATEncoder(
        input_size=15,
        hidden_size=args_loaded.enc_hidden,
        embed_size=args_loaded.enc_out,
        n_experts=args_loaded.n_experts,
    ).to(device)

    decoder = MHADecoder(
        encoder_size=encoder.out_size,
        context_size=JobShopStates.size,  # Ensure this is correctly defined/imported
        hidden_size=args_loaded.mem_hidden,
        mem_size=args_loaded.mem_out,
        clf_size=args_loaded.clf_hidden,
    ).to(device)

    # Load the model checkpoint
    print(f"Loading model weights from {model_path}")
    try:
        checkpoint = torch.load(model_path, map_location=device)
    except Exception as e:
        print(f"Failed to load the checkpoint: {e}")
        sys.exit(1)

    if isinstance(checkpoint, dict):
        try:
            encoder.load_state_dict(checkpoint['encoder'])
            decoder.load_state_dict(checkpoint['decoder'])
        except KeyError as e:
            print(f"KeyError: {e}. Ensure the checkpoint contains 'encoder' and 'decoder' keys.")
            sys.exit(1)
    elif isinstance(checkpoint, tuple) and len(checkpoint) == 2:
        encoder.load_state_dict(checkpoint[0])
        decoder.load_state_dict(checkpoint[1])
    else:
        raise ValueError("Invalid checkpoint format. Expected a tuple of (encoder_state_dict, decoder_state_dict) or a dict with 'encoder' and 'decoder' keys.")
    
    print("Model weights loaded successfully.")
    print(encoder)
    print(decoder)

    # Load the test dataset
    if args.benchmark == 'train':
        path = f'dataset5k/'
    else:
        path = f'./benchmarks/{args.benchmark}'

    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    out_file = f'output/{args.benchmark}-B{args.beta}_{args.seed}.csv'

    gaps = ObjMeter()
    time_meter = TimeObjMeter()

    # Lists to collect the best z vectors, instance sizes, and gap percentages
    z_collection = []
    j_collection = []
    m_collection = []
    gap_collection = []  # New list to collect gap percentages

    # Gather all instance files
    try:
        all_files = [file for file in os.listdir(path) if not (file.startswith('.') or file.startswith('cached'))]
    except FileNotFoundError:
        print(f"Benchmark path not found: {path}")
        sys.exit(1)

    if not all_files:
        print(f"No valid instance files found in the benchmark path: {path}")
        sys.exit(1)

    # Apply random shuffling if -random is set
    if args.random:
        np.random.seed(args.seed)  # For reproducibility
        np.random.shuffle(all_files)

    # Initialize a counter for processed instances
    processed_instances = 0

    for file in tqdm(all_files):
        if args.num_instances is not None and processed_instances >= args.num_instances:
            print(f"Reached the limit of {args.num_instances} instances. Stopping the testing process.")
            break



        # Solve the instance
        instance_path = os.path.join(path, file)
        instance = load_data(instance_path, device=dev)
        
        # Apply filtering based on size thresholds
        try:
            j = instance['j']
            m = instance['m']
        except KeyError as e:
            print(f"Skipping {file}: Missing key {e}.")
            continue

        # Check against maximum thresholds
        if args.max_jobs is not None and j > args.max_jobs:
            print(f"Skipping {file}: number of jobs (j={j}) exceeds max_jobs={args.max_jobs}.")
            continue
        if args.max_machines is not None and m > args.max_machines:
            print(f"Skipping {file}: number of machines (m={m}) exceeds max_machines={args.max_machines}.")
            continue

        
        try:
            start_time = time.time()
            res = validation(encoder, decoder, instance,
                                beta=args.beta, seed=args.seed)
            # Determine elapsed time and update time meter per shape
            elapsed_time = time.time() - start_time
            time_meter.update(instance, elapsed_time)
        except Exception as e:
            print(f"Validation failed for {file}: {e}")
            continue

        # Save results
        try:
            pd.DataFrame([res]).to_csv(out_file, index=False, mode='a+', sep=',')
        except Exception as e:
            print(f"Failed to write results for {file} to {out_file}: {e}")
            continue
        gaps.update(instance, res['GAP'])

        # Increment the processed instances counter
        processed_instances += 1

    
    print(f"\t\t{args.benchmark} set: AVG Gap={gaps.avg:2.3f}")
    print(gaps)
    print(f"Average processing time per instance: {time_meter.avg:4.3f} seconds")
    print("Processing time metrics per shape:")
    print(time_meter)

# python3 test_schedulexpert.py -folder_path checkpoints/SchedulExpert -num_instances 80 -benchmark TA -infer_sch_expert