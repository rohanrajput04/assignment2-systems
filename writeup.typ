#set page(margin: 2cm, numbering: "1")
#set text(size: 10pt)
#set heading(numbering: none)
#show figure.where(kind: table): set figure.caption(position: top)

#align(center)[
  #text(size: 16pt, weight: "bold")[CS336 Assignment 2 — Systems]
  #v(0.2em)
  Rohan Rajput
]

#v(1em)

*Hardware:* NVIDIA H100 80GB HBM3 (Modal). \
*Defaults:* vocab 10000, context length 512, batch size 4, FP32.

= 2.1 Profiling and Benchmarking

== Problem (benchmarking_script) (b)

#figure(
    table(
    columns: 4,
    align: (left, right, right, right),
    table.header([*Size*], [*Forward*], [*Fwd+Bwd*], [*Full step*]),
        [small], [20.1 ± 0.04], [61.4 ± 0.05], [69.4 ± 0.05],
    [medium], [55.1 ± 0.08], [168.6 ± 0.06], [191.1 ± 0.11],
    [large], [123.4 ± 0.05], [384.9 ± 0.13], [436.2 ± 0.13],
    [xl], [335.6 ± 0.18], [1072.1 ± 0.12], [1238.0 ± 0.12],
    [10B],    [OOM], [OOM], [OOM],
    ),
    caption: [Mean ± std runtime in ms over 10 measured steps after 5 warm-up steps, FP32.]
)


Backward costs roughly twice the FLOPs of forward: each linear layer computes one matmul going forward ($Y = X W$) but two coming back, one for the input gradient ($d X = d Y W^T$) and one for the weight gradient ($d W = X^T d Y$). Forward+backward should therefore cost about $3 times$ forward, and the measured ratios — 3.05, 3.06, 3.12 and 3.20 for small through xl — are consistent with that. The slight excess over 3, which grows with model size, comes from non-matmul backward kernels (softmax, SwiGLU and RMSNorm gradients) and from re-reading the activations saved during the forward pass.

Variability is very low: the standard deviation is under 0.2% of the mean for every configuration.

These numbers required a measurement fix first. The warm-up loop was initially unsynchronized, so the CPU ran ahead of the GPU and the first `torch.cuda.synchronize()` in the measurement loop waited on undrained warm-up work, charging it to the first measured step. Adding a synchronization after each warm-up step moved `small`/forward from 22.8 ± 8.2 ms to 20.1 ± 0.04 ms — a 13% lower reported mean and a 211× smaller standard deviation. The code did not become faster; the measurement became correct.

The 10B configuration ran out of memory in all three modes. Its 51.3 GB of FP32 parameters loaded onto the 80 GB H100 successfully, but the forward pass failed at the attention-score einsum: with autograd enabled roughly 1 GB of activations is retained per layer across 50 layers, exhausting the remaining capacity partway through.

== Problem (benchmarking_script) (c)


#figure(
  table(
    columns: 4,
    align: (left, right, right, right),
    table.header([*Warm-up steps*], [*Forward*], [*Fwd+Bwd*], [*Full step*]),
    [0], [149.2 ± 80.18],  [460.1 ± 239.65], [495.1 ± 189.45],
    [1], [123.3 ± 0.19],   [385.7 ± 0.15],   [442.5 ± 23.39],
    [2], [123.1 ± 0.21],   [384.9 ± 0.17],   [436.8 ± 0.25],
    [5], [123.4 ± 0.05],   [384.9 ± 0.13],   [436.2 ± 0.13],
  ),
  caption: [Effect of warm-up steps on the `large` model (36 layers, $d_"model"=1280$). Mean ± std runtime in ms over 10 measured steps.],
)
- Zero warm-up inflates the mean by 13–21% and the standard deviation by three to four orders of magnitude, since one or two cold steps dominate a ten-step average.
- Forward and forward+backward converge after a single warm-up step.
- The full training step needs two. AdamW allocates its moment tensors lazily on the first `step()` call — two fresh tensors per parameter, roughly 7.8 GB for the `large` model — and that one-time allocator growth spills past a single warm-up iteration.


== Problem (nsys_profile)

=== (a) Total forward time vs. Python timing
=== (b) Highest cumulative-time kernel
=== (c) Non-matmul kernels
=== (d) Full training step vs. inference
=== (e) Softmax vs. matmul runtime

== Problem (mixed_precision_accumulation)

#figure(
  table(
    columns: 3,
    align: (left, left, right),
    table.header([*Accumulator*], [*Addend*], [*Result*]),
    [FP32], [FP32],                [10.0001],
    [FP16], [FP16],                [9.9531],
    [FP32], [FP16],                [10.0021],
    [FP32], [FP16, cast to FP32],  [10.0021],
  ),
  caption: [Summing 0.01 one thousand times. The exact value is 10.0.],
)

The exact answer is 10.0. Accumulating in FP16 loses 0.47%, while accumulating in FP32 with FP16 addends loses only 0.02% — a twentyfold difference produced purely by the accumulator's dtype, since the addend is identical in both cases.

The dominant error in the all-FP16 case is swamping: once the running sum grows large, the spacing between adjacent representable values exceeds the increment being added. At $s approx 8$ the FP16 spacing is $2^3 dot 2^(-10) = 0.0078$, comparable to the 0.01 addend, so each addition loses a substantial fraction of its value and the sum drifts below the true result.

Cases 3 and 4 are identical, since implicit type promotion and an explicit `.type(torch.float32)` cast do the same thing. Their residual error is pure representation rather than accumulation: the nearest FP16 value to 0.01 is 0.0100021, and $1000 dot 0.0100021 = 10.0021$ exactly matches the observed sum. The lesson is that the accumulator's precision matters more than the operands', which is why `torch.autocast` keeps reductions and accumulations in FP32.

== Problem (benchmarking_mixed_precision)

=== (a) Data types under FP16 autocast
- Model parameters: fp16
- Output of `fc1`: fp16
- Output of layer norm: fp32
- Logits: fp16
- Loss: fp32
- Gradients: fp32

=== (b) Layer norm sensitivity
=== (c) BF16 mixed-precision timings

== Problem (memory_profiling)

=== (a) Memory timelines
=== (b) Peak memory per context length
=== (c) Mixed-precision peak memory
=== (d) Residual-stream activation size
=== (e) Largest allocations
=== (f) Nsight memory profiling