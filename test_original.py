import argparse
import torch
import pandas as pd
import os
from utils.sampling import sampling, greedy
from utils.inout import load_data
import time 
from utils.utils import ObjMeter
from utils.utils import TimeObjMeter
from tqdm import tqdm
from sampling_methods import _smc_sampling_fast_improved,_smc_sampling_fast

# Training device
dev = 'cuda' if torch.cuda.is_available() else 'cpu'


@torch.no_grad()
def validation(encoder: torch.nn.Module,
               decoder: torch.nn.Module,
               ins,
               beta: int = 32,
               seed: int = None):
    """

    Args:
        encoder: Encoder.
        decoder: Decoder.
        ins: JSP instance.
        beta: Number of solution to generate for each instance.
        seed: Random seed.
    """
    if seed is not None:
        torch.manual_seed(seed)
    encoder.eval()
    decoder.eval()

    st = time.time()
    if beta > 1:
        # s, mss = sampling(ins, encoder, decoder, bs=beta, device=dev)
        trajs, ptrs, mss = _smc_sampling_fast_improved(
                ins, encoder, decoder, 
                bs=beta, device=dev,
                n_checkpoints=6,
                tau_start=1.99,
                tau_end=0.20,
                finish_tau=0.0267,
                ess_threshold=0.51
            )
    else:
        s, mss = greedy(ins, encoder, decoder, device=dev)
    exe_time = time.time() - st

    #
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
    return results


#
parser = argparse.ArgumentParser(description='Test Pointer Net')
parser.add_argument("-model_path", type=str, required=False,
                    default="./checkpoints/slj_original/PtrNet-B256.pt",
                    help="Path to the model.")
parser.add_argument("-benchmark", type=str, required=False,
                    default='TA', help="Name of the benchmark for testing.")
parser.add_argument("-beta", type=int, default=512, required=False,
                    help="Number of sampled solutions for each instance.")
parser.add_argument("-seed", type=int, default=42,
                    required=False, help="Random seed.")
args = parser.parse_args()
print(args)

if __name__ == '__main__':
    from architectures.PointerNet import GATEncoder
    import sys
    import architectures.PointerNet as PointerNet
    sys.modules['PointerNet'] = PointerNet  

    print(f"Using {dev}...")

    # Load the model
    print(f"Loading {args.model_path}")
    enc_w, dec_ = torch.load(args.model_path, map_location=dev)
    enc_ = GATEncoder(15).to(dev)   # Load weights to avoid bug with new PyG
    enc_.load_state_dict(enc_w)
    m_name = args.model_path.rsplit('/', 1)[1].split('.', 1)[0]

    

    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in enc_.parameters() if p.requires_grad)
    print(f"Encoder Number of trainable parameters: {trainable_params}")

    # Calculate trainable parameters
    trainable_params = sum(p.numel() for p in dec_.parameters() if p.requires_grad)
    print(f"Decoder Number of trainable parameters: {trainable_params}")


    #
    path = f'./benchmarks/{args.benchmark}'
    if not os.path.exists('./output/'):
        os.makedirs('./output/')
    out_file = f'output/{m_name}_{args.benchmark}-B{args.beta}_{args.seed}.csv'

    #
    gaps = ObjMeter()
    time_meter = TimeObjMeter()
    for file in tqdm(os.listdir(path)):
        if file.startswith('.') or file.startswith('cached'):
            continue
        # Solve the instance
        instance = load_data(os.path.join(path, file), device=dev)
        start_time = time.time()
        res = validation(enc_, dec_, instance,
                         beta=args.beta, seed=args.seed)
        elapsed_time = time.time() - start_time
        time_meter.update(instance, elapsed_time)
        # Save results
        pd.DataFrame([res]).to_csv(out_file, index=False, mode='a+', sep=',')
        gaps.update(instance, res['GAP'])

    #
    print(f"\t\t{args.benchmark} set: AVG Gap={gaps.avg:2.3f}")
    print(gaps)
    print(f"Average processing time per instance: {time_meter.avg:4.3f} seconds")
    print("Processing time metrics per shape:")
    print(time_meter)

# python3 test_original.py -folder_path checkpoints/SchedulExpert -num_instances 80 -benchmark TA