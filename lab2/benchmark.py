import sklearn
import s3prl.hub as hub
import torch
import numpy as np
import argparse
from timeit import default_timer as timer

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark script.')
    parser.add_argument("model_name", help="Name of model to benchmark")
    parser.add_argument("--timing-iters", default=10, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--input-size", default=10000, type=int)
    args = parser.parse_args()

    model_name = args.model_name

    # Load model
    model = getattr(hub, model_name)()

    model.eval()

    # Get parameter count
    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print(f"Paremeter count for model {model_name}: {num_params}")

    # first pass to "warm up"
    with torch.no_grad():
        wavs = [torch.randn(10000, dtype=torch.float) for _ in range(2)]
        model(wavs)


    # benchmark timing
    with torch.no_grad():
        times = []
        for _ in range(args.timing_iters):
            wavs = [torch.randn(args.input_size, dtype=torch.float) for _ in range(args.batch_size)]
            start = timer()
            model(wavs)
            end = timer()

            cpu_inference_time = end-start
            times.append(cpu_inference_time)

        times = np.array(times)

        print(f"Data for model {model_name} with input size {args.input_size} and batch size {args.batch_size}.")
        print(f"Average CPU Inference Time: {np.mean(times)} seconds.")
        print(f"Standard Deviation of CPU Inference Times: {np.std(times)} seconds.")

    
