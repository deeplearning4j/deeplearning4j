# MTP Reference-Contract Repair Plan — merged from the external audit + session evidence
# Created 2026-09-17 after audit of ag_new_release_updates_2 @ e2aa602a9b
# Governing principle: poor draft accuracy must reduce ACCELERATION, never degrade target
# output. Losslessness first, acceptance second, speed third. No K/threshold tuning until
# the contract bugs below are fixed and gated.

## Verified findings (audit + session evidence, cross-checked)

F1 [P1] Hidden-state handoffs are PRE-normalization; references (llama.cpp, vLLM Qwen3.5) are
   POST-normalization, in BOTH places:
   - target -> first draft: LLaMAArchitecture exports target_hidden_states BEFORE the final
     model RMSNorm (references export after)
   - predictor -> recursive draft: mtp_hidden_states exported BEFORE shared_head_norm
     (references return the normalized state and reuse it for drafting)
   Learned per-channel gain means a later RMSNorm cannot undo a missing norm. Affects slot-0
   accuracy (first handoff) and slots 1..K disproportionately (second). Applies to ModelOpt
   path too (same buildGraph). NOTE: do NOT also add +1 to norm weights — ModelOpt importer
   already one-centers those weights; handoff location and weight representation are separate.
   Historical 77% acceptance does NOT validate the pre-norm export choice.

F2 [P1] Predictor bootstrap inserts an artificial first KV row: prepareBundledMtp keeps prompt
   ids unshifted, shifts hidden states right, zeros hidden[0], and commits that zero-conditioned
   row to the persistent predictor cache. vLLM shifts ids LEFT, appends the first sampled token
   with h_{N-1} — no artificial row ever exists. Extra VISIBLE cache entry changes attention.
   Minimal reproducer: one-token prompt (reference has exactly the meaningful pair; we have a
   zero row + warmup row). Fix the WHOLE bootstrap contract together: token shift, hidden rows,
   positions, cache length, mask, warmup boundary.

F3 [P1] CUDA stop-matching consumes the PROVISIONAL token before the rerun replaces it
   (autoregressive_decode.cu ~2530-2750): stopMatcher.accept(verify argmax) -> shouldStop ->
   rerun -> final token may DIFFER -> emitted token replaced but matcher already advanced.
   Three concrete failures: (a) verify EOS + rerun normal = premature stop after non-EOS token,
   (b) verify normal + rerun EOS = missing termination, (c) multi-token stop sequence advanced
   through a token never emitted. Fix: select the authoritative emitted token FIRST, then feed
   the matcher once.

F4 [P1] Single-token commit (consumedCount=1 + rerun) is a WORKAROUND, not a repaired verifier:
   it disables multi-token emission entirely, and active-length input alone does not prove
   window-execution == independent-scalar-execution (GEMM geometry, attention geometry, in-place
   KV writes, recurrent inputs unproven). Keep multi-token commit DISABLED until the full-state
   parity gate (see T3) passes. Production boundary: validated scalar path where needed.

F5 [P1 CPU] CPU path keeps multi-token emission and SKIPS repair of predictor rows already
   written during recursive drafting: repair fires only when repairPosition > mtpProcessedThrough,
   so rows base+1..base+K-1 retain SELF-generated (recursively carried) hidden states instead of
   verified target-conditioned states. Fix: repair every retained row produced from recursive
   carry; keep 'written horizon' and 'correctly conditioned horizon' as separate boundaries.

F6 [P2] Metrics/tests blind spots: acceptance counts can reflect the pre-rerun decision; the
   lifecycle tests are self-consistency, not mathematical oracles (both sides can share a wrong
   contract); TestQwen35Pipeline SLOT_BY_SLOT filter can null the golden baseline and silently
   skip parity; MTP_KV_SELFROW decodes BF16 as FP16 (diagnostic-only); fingerprints sample too
   little state. Historical 77% must not be used to choose hidden-state contracts.

## Cross-provider context (what matches already)
Separate embedding-norm and hidden-norm, concat(embedding, hidden), project to model width,
full predictor attention block, shared output head — all match vLLM/llama.cpp/SGLang. We fix
BOUNDARY TENSORS and COMMIT SEMANTICS, not the predictor architecture. Quantization nuance:
target quant and predictor quant are separate concerns (SGLang supports BF16 predictor on
quantized targets); cross-provider comparisons must match actual predictor weights/policy.

## Repair order (four bounded stages, each with a gate before the next)

T1 [FINAL-OUTPUT CONSISTENCY] CUDA stop-ordering (F3)
   - Forced verification/rerun disagreement tests FIRST (deterministic): EOS-in-verify/rerun
     normal; normal/EOS; multi-token stop sequence; callbacks; continuation across calls.
   - Move persistent stop matching after authoritative token selection.
   - Gate: those tests green on CUDA (and CPU for the same semantics).
   Milestones: one per test class run. Tests in platform-tests only, -Dbackend.artifactId set.

T2 [PREDICTOR CONTRACT] Both handoffs + bootstrap (F1 + F2)
   - LLaMAArchitecture: export target_hidden_states AFTER final model RMSNorm; return
     mtp_hidden_states AFTER shared_head_norm and carry THAT forward recursively.
   - prepareBundledMtp: adopt vLLM alignment (shift ids left, no artificial zero row, cache
     geometry updated consistently — token/hidden/positions/length/mask/warmup as ONE change).
   - External fixed-input fixture: compare normalized target state, predictor post-norm+proj
     input, predictor hidden output, logits, predictor KV against reference tensors
     (llama.cpp/vLLM checkpoint). DL4J-fresh-vs-compiled is necessary, NOT sufficient.
   - Expect: first-draft agreement improves slot-0; recursive carries fix slots 1..K.
   - Gate: fixture comparisons within declared tolerance; 8-token parity stays green.

T3 [CROSS-BACKEND COMMIT CONTRACT] CPU repair + full-state parity (F5 + F4)
   - CPU: repair every retained recursively-carried predictor row; separate written vs
     conditioned horizons; exercise accepted lengths 0..K including EOS and budget truncation.
   - Full-state parity gate (the missing oracle): at a committed prefix, compare
     A) independent scalar target execution vs B) window execution with active length 1 vs
     C) draft->verify->reject->continue — over next-token logits AND all next-step state:
     target KV, GDN state, conv state, positions, masks, pending input. Teacher-forced
     identical prefixes to avoid first-divergence confounds.
   - Only after T3 passes: restore multi-token CUDA emission.

T4 [ACCELERATION] p-min + K + replay — ONLY AFTER T1-T3
   - Implement spec-draft-p-min (chain truncation when draft prob < P) — llama.cpp knob.
   - Add draft/top-k correlation diagnostic (DRAFT_TOPK/TARGET_TOPK/DRAFT_VS_TARGET/CARRY_STATS)
     if still needed to close the accuracy question after T2.
   - Add impossible-token guard (never-trained last lm_head row => corruption signal).
   - Metrics separation: first-draft agreement / conditional depth-j agreement / emitted
     accepted tokens / tokens per target forward / verify-vs-rerun disagreement counts.
   - Then: K tuning, p-min tuning, whole-plan CUDA-graph replay for per-unit overhead.
   - STOP tuning K/thresholds until T1-T3 gates pass (user directive).

## Test matrix (minimum regression set; platform-tests only, milestones recorded per run)
- bootstrap: one-token, two-token, padded, long prefill -> predictor cache holds exactly the
  intended token/hidden pairs
- drafts vs external fixture: first + recursive -> both handoffs match reference contract
- scalar-vs-window parity from same nonzero state -> matching continuation, no prefix contamination
- forced acceptance lengths 0..K -> only consumed prefix affects committed state
- EOS / multi-token stop / rerun replacement -> matcher, callbacks, finish reason consistent
- one-shot vs chunked/resumed -> same committed-prefix semantics across the session seam
- production-path gate: GraphOptimizer + native steady-state path IN the final acceptance run;
  diagnostic graphs are localization tools only

## Session-lane rules (mandatory, from feedback memory)
One 27B GPU run at a time globally; launcher runs nothing concurrent until exit; lane check =
nvidia-smi only monitor PIDs + no benchTokens/ForkedBooter java + >=60GB free; watcher observes,
never launches; 8-token gate before any 250-token run; milestone every run.
