import json
import math
import sys
import torch
import argparse
import timeit
import statistics
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from contextlib import nullcontext


model_configs = {
    "small":  {"d_model": 768,  "d_ff": 3072,  "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096,  "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120,  "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10B":    {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}
def sync_gpu():
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    else:
        return


def setup(
    vocab_size: int,
    context_length: int,
    d_model: int,
    num_layers: int,
    num_heads: int,
    d_ff: int,
    batch_size: int,
    device: str
) -> tuple:

    with torch.device(device):
        model = BasicsTransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=d_model,
            num_layers=num_layers,
            num_heads=num_heads,
            d_ff=d_ff,
        )
        # Data Generation 
        inputs = torch.randint(0, vocab_size, (batch_size, context_length))
        # Test data 
        targets = torch.randint(0, vocab_size, (batch_size, context_length))

    params = model.parameters()

    optim = AdamW(params=params)
    
    return model, inputs, targets, optim

def run_step(model, inputs, targets, optim, mode, no_grad):
    
    if mode == "forward_backward":
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        model.zero_grad()
    elif mode == "full_step":
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        optim.step()
        model.zero_grad()

    elif mode == "forward":
        ctx = torch.no_grad() if no_grad else nullcontext()
        with ctx:
            logits = model(inputs)
    else:
        raise ValueError("This mode is not supported")



if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=512)
    parser.add_argument("--d_model", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--num_heads", type=int, default=4)
    parser.add_argument("--d_ff", type=int, default=128)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--warmup_steps", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=10)
    parser.add_argument("--mode", type=str, default="forward", choices=["forward", "forward_backward", "full_step"])
    parser.add_argument("--size", type=str, default=None, choices=["small", "medium", "large", "xl", "10B"])
    parser.add_argument("--dump_times", action="store_true")
    parser.add_argument("--no_grad", action="store_true")

    args = parser.parse_args()
    if args.num_steps < 2:
        parser.error("std needs atleast 2 samples")
    if args.no_grad and args.mode != "forward":
        parser.error("No grad only works with forward mode")
    config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "batch_size": args.batch_size,
        "device" : args.device
    }
    warmup_steps = args.warmup_steps
    num_steps = args.num_steps
    mode = args.mode

    if args.size:
        size = args.size 
        for key, val in model_configs[size].items():
            config[key] = val 
    
    
    model, inputs, targets, optim = setup(**config)

    print(sum(p.numel() for p in model.parameters()),file=sys.stderr)
    no_grad = args.no_grad
    for _ in range(warmup_steps):
        run_step(model, inputs, targets, optim, mode, no_grad)
        sync_gpu()

    total_time = []

    for _ in range(num_steps):
        before = timeit.default_timer()
        run_step(model, inputs, targets, optim, mode, no_grad)
        sync_gpu()
        after = timeit.default_timer()
        time_value = (after - before) * 1e3 
        total_time.append(time_value)
    
    
    mean_ms = statistics.mean(total_time)
    std_ms = statistics.stdev(total_time)

    if args.dump_times:
        print([round(x,2) for x in total_time], file=sys.stderr)
    print(f"The mean of time is {mean_ms} ms and std is {std_ms} ms",file=sys.stderr)
    final_dict = {}
    for key, val in config.items():
        final_dict[key] = val 
    final_dict["size"] = args.size 
    final_dict["warmup_steps"] = warmup_steps
    final_dict["num_steps"] = num_steps
    final_dict["mode"] = mode 
    final_dict["mean_ms"] = mean_ms
    final_dict["std_ms"] = std_ms
    final_dict["device_name"] = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
    final_dict["dump_times"] = args.dump_times
    final_dict["no_grad"] = args.no_grad
    print(json.dumps(final_dict))