# Simida

**LLM inference systems · GPU performance**

I’ve worked on LLM serving on Ascend, focusing on request routing and KV cache reuse. My public work includes CUDA and Triton kernel experiments on NVIDIA Blackwell.

At **HiSilicon Turing**, I developed NIC firmware features for LACP link aggregation offload and QoS. For NIC chip verification, I wrote SystemVerilog/UVM sequences and implemented core reference-model logic for VirtIO packed queues.

**Currently studying** [CMU 11-768 · AI Agents][agents]. I’m exploring agent harnesses and evaluation, with [notes on how these methods apply to AI infrastructure][agents-notes].

[Open-source explorations](#open-source-explorations) · [Kernel studies](#kernel-studies) · [Notes](#notes) · [Coursework](#coursework)

## Open-source explorations

### DeepInfer LocalEngram on B300

I ran DeepSeek V4.1 Flash with its full weights on four B300s using the NVLink LocalEngram path from [community PR #246][engram-upstream]. I profiled the stock gather, a fused Triton gather prototype, layer prefetch, and their combination in one allocation. At 8K context, the fused prototype reduced median decode step intervals by **2.30%** at one request per rank and **1.84%** at eight in a fixed-order run. Generated tokens matched the stock path.

[Reproduction, measurements, and traces][local-engram]

| Project | What I investigated | Public evidence |
| --- | --- | --- |
| vLLM / MiniMax M3 | WMHPC team follow-up on sparse decode. In a one-layer dummy-weight replay, CUDA Graph padding crossed the backend-selection threshold in 24 of 41 CUTLASS decode steps. | [Workload replay and analysis][msa-replay] |
| PyTorch FSDP2 / NCCL | Compared SM and copy-engine all-gather with symmetric staging fixed on two B300s. The median training step fell 0.71%, while P95 rose slightly. | [Benchmark and Nsight evidence][fsdp2-ce] |

## Kernel studies

AI-assisted experiments from the WMHPC × LCPU AI Infra Seminars.

### MiniMax M3 sparse decode

In this team project, we kept the original Triton partial kernel and tiled the merge over output channels with one warp per CTA. Nsight Compute showed fewer shared-memory wavefronts in the BF16 cases we profiled.

On B300, the full partial + merge path ran **1.117× faster** than the original path (geometric mean across 16 configurations with PDL off). Four configurations regressed with PDL on.

In a follow-up on B300, two PDL regressions seen in 64-call CUDA Graphs were near zero in single-call graphs. A separate one-layer vLLM replay traced how graph padding affected the choice between Triton and CUTLASS.

[Triton code][msa-code] · [Benchmark and correctness checks][msa-results] · [PDL and vLLM follow-up][msa-follow-up]

### Blackwell BF16 GEMM

I tuned a CUDA/PTX kernel with a five-stage TMA pipeline and double-buffered TMEM to overlap compute and writeback. For **4096³ on B300**, it reached **91.94–93.54%** of the fastest cuBLAS path I measured.

I used Nsight Compute and SASS to guide tuning of a CUTLASS-derived two-CTA kernel, testing dynamic tile scheduling, pipeline depth, and TMA issue order. In single-GPU tests on two B300s, the tuned kernel delivered **2.35–4.18% higher throughput** than default cuBLAS GemmEx and reached **97.45–99.69%** of a fixed cuBLASLt baseline.

[Implementations and experiment notes][gemm-overview] · [CUTLASS benchmarks][gemm-results]

## Notes

[AI Infra notes and knowledge map](https://github.com/csdiywithme/AI_infra) · Course notes and experiment logs spanning LLM serving, training systems, GPU kernels, and parallel computing. Includes notes from CS149, CS336, CMU 11-763, the LCPU AI Infra Seminars, and HPCGame 2026.

## Coursework

### AI Infra and Inference

- **WMHPC × LCPU AI Infra Seminars** · [Assignments and experiments](https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars)
- **CMU 11-763 · Inference Algorithms for Language Modeling** · [Course repository](https://github.com/csdiywithme/lminference-fall2025-code) · [HW3: KV cache and speculative sampling](https://github.com/csdiywithme/lminference-fall2025-code/blob/main/homework3/STATUS.zh-CN.md) · [Reproduction guide](https://github.com/csdiywithme/lminference-fall2025-code/blob/main/homework3/REPRODUCE.md)

### CS149 · Parallel Computing

- [Assignment 1](https://github.com/csdiywithme/cs149-asst1) · CPU parallelism, SIMD, and profiling
- [Assignment 2](https://github.com/csdiywithme/cs149-asst2) · Parallel task runtime and dependency scheduling
- [Assignment 3](https://github.com/csdiywithme/cs149-asst3) · CUDA circle renderer
- [CS149 GPT](https://github.com/csdiywithme/cs149-cs149gpt) · CPU attention optimization for nanoGPT
- [Big Graphs](https://github.com/csdiywithme/cs149-biggraphs-ec) · Parallel BFS with OpenMP

### CS336 · Language Modeling from Scratch

- [Assignment 1 · Basics](https://github.com/csdiywithme/cs336-assignment1-basics) · Language model fundamentals
- [Assignment 2 · Systems](https://github.com/csdiywithme/cs336-assignment2-systems) · Language model performance and distributed training
- [Assignment 3 · Scaling](https://github.com/csdiywithme/cs336-assignment3-scaling) · Language model scaling experiments
- [Lectures](https://github.com/csdiywithme/cs336-lectures) · Lecture notes, examples, and slides

### Other Courses and Experiments

- [MIT 6.824](https://github.com/csdiywithme/MIT-6.824) · Distributed Systems
- [UC Berkeley CS61A](https://github.com/csdiywithme/UCB-CS61A) · Structure and Interpretation of Computer Programs
- [libfabric / EFA](https://github.com/csdiywithme/libfabric-efa-demo) · Networking demo

[agents]: https://github.com/csdiywithme/11-768-ai-agents
[agents-notes]: https://github.com/csdiywithme/11-768-ai-agents/tree/main/research/ai-infra
[msa-code]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment02/team/c2_msa_decode/candidate.py
[msa-results]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment02/team/c2_msa_decode/experiments/CANDIDATE_EXPERIMENTS.md
[msa-follow-up]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/4ba7bb18f0f8c847b46fbbd824e81171f608ec61/assignment02/team/c2_msa_decode/experiments/FRONTIER_MSA.md
[gemm-overview]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/ae1c056d60f916097f3a2cf0cdc53abfc86875de/assignment02/experiments/bf16_frontier_20261113/OVERVIEW.md
[gemm-results]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/ae1c056d60f916097f3a2cf0cdc53abfc86875de/assignment02/experiments/bf16_frontier_20261113/pack192_input_order/METHOD.md
[local-engram]: https://github.com/csdiywithme/deepinfer-localengram-profiling
[engram-upstream]: https://github.com/deepseek-ai/DeepInfer/pull/246
[msa-replay]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/4ba7bb18f0f8c847b46fbbd824e81171f608ec61/assignment02/team/c2_msa_decode/experiments/FRONTIER_MSA_WORKLOAD.md
[fsdp2-ce]: https://github.com/csdiywithme/cs336-assignment2-systems/blob/main/output/fsdp2-ce/README.md
