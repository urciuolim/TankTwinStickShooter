import argparse
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--idx", type=int, default=0, help="For parallel processing, portion of population this job will train (from 1 to {args.part} )")
args = parser.parse_args()

device_str = "cuda:" + str(args.idx%4)
torch.cuda.set_device(device_str)


print("Idx:", args.idx, "\tdevice:", torch.cuda.current_device())