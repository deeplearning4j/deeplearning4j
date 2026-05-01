---
name: cpu-has-full-dsp
description: CPU has full DSP including emulated DSP — ALL configs must work on CPU, NEVER dismiss DSP failures as platform-specific
type: feedback
---

CPU has full DSP support including emulated DSP. ALL execution configurations (SLOT_BY_SLOT, CUDA_GRAPHS, TRITON_*, etc.) are expected to work on CPU — they use OpenVINO, oneDNN Graph, and CPU Triton backends.

**Why:** TRITON configs trigger OpenVINO and DSP compilation on CPU, not CUDA Triton. Dismissing these failures as "expected because there's no GPU" is completely wrong.

**How to apply:**
- NEVER conclude that a DSP/Triton/CUDA_GRAPHS config failure is "expected" on CPU
- NEVER dismiss config failures as platform-specific without investigating
- ALL 6 test configurations must pass on BOTH CPU and CUDA
- When a config fails on CPU, investigate the actual DSP backend cascade (oneDNN Graph → OpenVINO → CPU Triton → native slot)
- Fix the root cause of KERNEL_FAILURE — don't rationalize it away
- NEVER work around DSP errors — FIX them
