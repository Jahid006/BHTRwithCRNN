
import numpy as np
import torch


# https://deci.ai/blog/measure-inference-time-deep-neural-networks/


def inference_time_calculator(
        model,
        inputs,
        dummy_inputs,
        repetitions=20,
        device='cuda'
):
    model.to(device)
    dummy_inputs.to(device)
    inputs.to(device)
    model.eval()

    starter, ender = (
        torch.cuda.Event(enable_timing=True), 
        torch.cuda.Event(enable_timing=True)
    )

    for _ in range(repetitions//5):
        model(dummy_inputs)
    batch_size = inputs.shape[0]
    timings = np.zeros((repetitions, 1))

    with torch.no_grad():
        for rep in range(repetitions):
            starter.record()
            _ = model(inputs)
            ender.record()
            torch.cuda.synchronize()
            curr_time = starter.elapsed_time(ender)
            timings[rep] = curr_time

    mean_syn = np.sum(timings) / repetitions
    std_syn = np.std(timings)
    total_time = np.sum(timings)
    Throughput = (repetitions*batch_size)/total_time

    return {
        'mean': mean_syn,
        'std': std_syn,
        'throughput': Throughput
    }
