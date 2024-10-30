import torch
import torch.nn 
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))
MODEL = importlib.import_module('pointnet2_sem_seg')
classifier = MODEL.get_model(13).cuda()
from deepspeed.profiling.flops_profiler import get_model_profile
detailed = False
args1=[torch.rand(1,9,4096).cuda()]
flops, macs, params = get_model_profile(
    model=classifier,
    args=args1,
    print_profile=detailed,  # prints the model graph with the measured profile attached to each module
    detailed=detailed,  # print the detailed profile
    warm_up=10,  # the number of warm-ups before measuring the time of each module
    as_string=False,  # print raw numbers (e.g. 1000) or as human-readable strings (e.g. 1k)
    output_file=None,  # path to the output file. If None, the profiler prints to stdout.
    ignore_modules=None)  # the list of modules to ignore in the profiling
print(f'\tParams.(M)\tGFLOPs')
print(f'\t{params / 1e6: .3f}\t{flops / (float(1) * 1e9): .2f}')
# n_runs = 200
# import time
# with torch.no_grad():
#     for _ in range(10):  # warm up.
#         classifier(*args1)
#     start_time = time.time()
#     for _ in range(n_runs):
#         classifier(*args1)
#         torch.cuda.synchronize()
#     time_taken = time.time() - start_time
# n_batches = n_runs * 64
# print(f'Throughput (ins./s): {float(n_batches) / float(time_taken)}')