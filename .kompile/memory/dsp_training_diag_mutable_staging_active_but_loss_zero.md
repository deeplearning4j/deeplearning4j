---
name: dsp_training_diag_mutable_staging_active_but_loss_zero
description: DSP training diagnostics show mutable staging active, but loss-zero parity failure remains.
type: project
---

**Test:** `DspTrainingE2ETest#testDspTrainingParityWithNonDsp` with EXECUTE/STREAM_SYNC diagnostics at full level.\n**Result:** 3 tests run, 3 failures. DSP final loss remains 0.0 for SGD, Adam, Nesterovs.\n**Key diagnostic:** replay path shows mutable external staging active (`STAGING_D2D: copied=8 ... total=8` and frozen_fast_path uses staged pointers for 9 externals).\n\n**Implication:** The training-only mutable external input hook fires sufficiently for native replay staging. Remaining root cause is likely loss output extraction/write semantics or frozen/replay classification of training graph outputs, not simply stale trainable weights.
