# use_device.py
import torch

def setup_gpu(gpu_ids):

    if torch.cuda.is_available() and gpu_ids != "":
        gpu_list = [int(x) for x in gpu_ids.split(",")]
        ngpus = len(gpu_list)
        device = torch.device("cuda")
        print("Device =", device)
        print(f"use_GPU: {ngpus}, GPU_id: {gpu_list}")
    else:
        device = torch.device("cpu")
        ngpus = 0
        print("Device =", device)
        print("CPU for train")
    return device, ngpus