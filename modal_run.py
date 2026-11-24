import json
import os
import subprocess
import sys

import modal

image = (
    modal.Image.debian_slim(python_version="3.12")
    .pip_install(
        "einops>=0.8",  # tensor operations
        "einx>=0.4",  # more general tensor operations
        "jaxtyping>=0.3",  # type hints for pytorch/jax/numpy
        "numpy>=2.4",
        "psutil>=7",
        "pytest>=9.0",
        "pytest-timeout",
        "regex>=2026.3.32",  # more powerful regex than the builtin `re` module
        "tiktoken>=0.12.0",
        "torch~=2.11.0",
        "tqdm>=4.67",
        "wandb>=0.25",
        "ty>=0.0.26",
        "ruff>=0.15.8",
    )
    .add_local_python_source("cs336_basics")
    .add_local_python_source("cs336_systems")
)

volume = modal.Volume.from_name("cs336-data-am-2", create_if_missing=True)

app = modal.App("cs336-assignment-2")

WORKSPACE = "/result"


@app.function(image=image, gpu="H100", volumes={WORKSPACE: volume}, timeout=60 * 60 * 3)
def gpu_run():
    sizes = ['large']  # ["small", "medium", "large", "xl", "10B"]
    modes = ["forward", "forward_backward", "full_step"]
    warmups = [0,1,2]
    num_steps = 10
    output_list = []

    file_path = os.path.join(WORKSPACE, "result.jsonl")
    with open(file_path, "a") as f:
        for size in sizes:
            for mode in modes:
                for warmup_steps in warmups:
                    output = subprocess.run(
                        [sys.executable, "-m", "cs336_systems.benchmark", "--size", size, "--mode", mode, "--num_steps", str(num_steps), "--warmup_steps", str(warmup_steps), "--dump_times"],
                        capture_output=True,
                        text=True,
                    )
                
                    if output.returncode:
                        error_dict = {"size": size, "mode": mode, "status": "failed", "stderr": output.stderr}
                        err = json.dumps(error_dict)
                        f.write(err + "\n")
                    print(output.stderr)
                    if output.stdout:
                        f.write(output.stdout)
                    f.flush()
                    volume.commit()
                    output_list.append(output.stdout)
    return output_list
@app.function(image=image, gpu="A10",  timeout=60 * 5)
def toy_dtypes():
    output = subprocess.run(
                    [sys.executable, "-m", "cs336_systems.bechmarking_mixed_precision"],
                    capture_output=True,
                    text=True,
                )
    print(output.stderr)
    print(output.stdout)



@app.local_entrypoint()
def start():
    output_list = gpu_run.remote()
    print(output_list)

@app.local_entrypoint()
def toy():
    toy_dtypes.remote()
