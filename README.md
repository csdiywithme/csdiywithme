# simida

**LLM inference systems · GPU performance**

I build LLM serving systems on Ascend, with a focus on request routing and KV cache reuse. My public work includes CUDA and Triton kernel experiments on NVIDIA Blackwell.

[Kernel studies](#kernel-studies) · [Notes](#notes) · [Coursework](#coursework)

## Kernel studies

AI-assisted experiments from the WMHPC × LCPU AI Infra Seminars.

### MiniMax M3 sparse decode

In this team project, we kept the original Triton partial kernel and tiled the merge over output channels with one warp per CTA. Nsight Compute showed fewer shared-memory wavefronts in the BF16 cases we profiled.

On B300, the full partial + merge path ran **1.117× faster** than the original path (geometric mean across 16 configurations with PDL off). Four configurations regressed with PDL on.

[Triton code][msa-code] · [Benchmark and correctness checks][msa-results]

### Blackwell BF16 GEMM

The epilogue reads more accumulators per TMEM load and rearranges them in shared memory for coalesced stores. In the same B300 run at **M = N = K = 4096**, the persistent baseline measured **654 TFLOP/s**, the epilogue variant **970 TFLOP/s**, and the variant with a 128-wide output tile **1,140 TFLOP/s**.

[CUDA code][gemm-code] · [Benchmark and Nsight Compute results][gemm-results]

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

[msa-code]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment02/team/c2_msa_decode/candidate.py
[msa-results]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment02/team/c2_msa_decode/experiments/CANDIDATE_EXPERIMENTS.md
[gemm-code]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment02/cuda/m4_gemm/04h_bn128.cu
[gemm-results]: https://github.com/csdiywithme/wmhpc-training-camp-x-lcpu-ai-infra-seminars/blob/main/assignment02/experiments/results/m4-epilogue-tiles-20260923T044929Z/OBSERVATIONS.md
