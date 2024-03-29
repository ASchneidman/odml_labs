import sklearn
import s3prl.hub as hub
import torch
import numpy as np
import argparse
from timeit import default_timer as timer
import matplotlib.pyplot as plt

import torch.quantization


def modified_load(old_load):
    def new_load(*args, **kwargs):
        kwargs['map_location'] = torch.device('cpu')
        return old_load(*args, **kwargs)

    torch.load = new_load

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Benchmark script.')
    #parser.add_argument("model_name", help="Name of model to benchmark")
    parser.add_argument("--timing-iters", default=10, type=int)
    parser.add_argument("--batch-size", default=2, type=int)
    parser.add_argument("--input-size", default=10000, type=int)
    args = parser.parse_args()

    #model_name = args.model_name
    
    models = ['wav2vec', 'wav2vec2_xlsr', 'vq_wav2vec', 'modified_cpc', 'wav2vec2', 'hubert']

    old_load = torch.load
    modified_load(old_load)

    for model_name in models:
        # Load model
        model = getattr(hub, model_name)()

        model.eval()

        #quantized_model = torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
        
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


        """
        
        with torch.no_grad():
            times = []
            X = np.arange(1000, 11000, 1000)
            for i in range(len(X)):
                wavs = [torch.randn(X[i], dtype=torch.float) for _ in range(args.batch_size)]
                start = timer()
                model(wavs)
                end = timer()

                cpu_inference_time = end-start
                times.append(cpu_inference_time)

            times = np.array(times)
            np.savez('./{}.npz'.format(model_name), X, times)

            print(f"Data for model {model_name} with input size {args.input_size} and batch size {args.batch_size}.")
            print(f"Average CPU Inference Time: {np.mean(times)} seconds.")
            print(f"Standard Deviation of CPU Inference Times: {np.std(times)} seconds.")

            plot_fname = "{}_plot.png".format(model_name)
            x = X # e.g. batch sizes
            y = times # mean timings
        
            plt.plot(x, y, 'o')
            plt.xlabel('Input Size')
            plt.ylabel('Inference Times')
            plt.savefig(plot_fname)
        """