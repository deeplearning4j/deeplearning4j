# ADR 0126: Random state in SameDiff execution

## Status

Accepted (2026-10-02). Amends ADR 0089 (CUDA graph capture and replay): slots that
draw random state are never captured. Amended 2026-10-03: the generator is Philox4x32-10
(see The generator).

## Context

`Nd4j.getRandom().setSeed(...)` is how ND4J callers make random draws reproducible,
and `Nd4j.exec(CustomOp)` honors it: `DefaultOpExecutioner.initOpContext` copies the
thread generator's state into the op's context and the executioner copies the advanced
state back. SameDiff did neither:

- The standard session runs every op through an `OpContext` of its own. A native
  context's generator is seeded from the clock (`RandomGenerator(0, 0)`), so random
  ops ignored `setSeed`, and two runs of a graph never drew the same values.
- A DSP plan executes its slots with step contexts that are seeded the same way.
  Once a plan captured a CUDA graph, its replays repeated the draws of the capture.
- Unseeded dropout (seed argument 0) built `RandomGenerator(3019, 0)` on every
  execution: the same mask forever, so dropout did not drop anything new.
- `getOpTraits` returned 32 bits and cut off `OP_TRAIT_STATEFUL` (bit 32), so Java
  could not tell a random op from a deterministic one. Constant folding guessed from
  the Java class's package and folded `random_crop`, whose class lives elsewhere.

The kernels themselves had defects that a seeded generator exposes: gamma samplers
shared a function-level static draw index (a data race across OpenMP threads), CUDA
gamma started every element of a row at the same uniform, CUDA Poisson read its
`DOUBLE` temporaries as the output type, Poisson drew one uniform per row and looped
forever once `exp(-lambda)` underflowed, integer uniform and Poisson never advanced
the generator, and `set_seed` seeded a copy of it.

## Decision

### One generator per thread

A random op draws from its context's generator, and whoever executes the op seeds
that generator from the thread's `Nd4j.getRandom()` state and takes the advanced
state back:

- The standard session does this for every op whose descriptor declares
  `OP_TRAIT_STATEFUL` (`InferenceSession.execWithThreadRandom`; `TrainingSession`
  shares it).
- A DSP plan execution does it once for the plan entry's context
  (`DynamicShapePlanExecutor`, and `NativeExecutionBinding` between
  `beginNativeUse` and `completeNativeUse`). The plan entry makes that context's
  generator the thread's execution generator (`DspExecutionRandomScope`), and every
  slot that draws random state runs with it (`SlotRandomStateScope`): its step
  context takes the execution's state and hands back what the op advanced.

### Random slots run live

`NativeSlot::drawsRandomState()` is a stateful op that writes no input. Ops that
write inputs (recurrent state, KV caches) are stateful through tensors a replay
updates as well, and they stay capturable. Amended 2026-10-03: the decision is the
op's, for the slot's arguments (`DeclarableOp::drawsRandomStateFor`, resolved once
when a plan compiles the slot). The default is the rule above; an op that draws for
some arguments only overrides it. `dot_product_attention_v2` writes its KV caches and
draws its dropout mask from its context's generator with a dropout rate above 0 while
training, so those slots run live with the execution generator and the others stay
capturable. Its slots used to keep a generator seeded from the clock, so
`Nd4j.getRandom().setSeed` did not reproduce attention dropout, a gradient check of
it compared different masks, and a capture would have replayed one mask. `isCapturable()`, the single source of
truth for capture, rejects a slot that draws random state: a replay would repeat the
draws of its capture. Compiled segments keep such a slot as a live gap. The Vulkan
recorder, which uploads a random op's state as a replay input
(`VULKAN_EMITTER_TRAIT_RANDOM_STATE`), seeds that state from the slot context's
generator; feeding it the execution generator is open work.

### Seeds

Every random op reads its seed argument the same way
(`helpers::applySeedArgument`): a nonzero seed fixes the generator's states, so the
op's draws depend on the seed alone; 0 or no seed uses the context's generator.
Dropout keeps its fixed mask for a nonzero seed and draws a new mask from the
context's generator for seed 0. `set_seed` seeds the context's own generator, which
the executor hands on to the ops that follow.

### Traits reach Java whole

`getOpTraitMask` returns all 64 bits (`getOpTraits` stays for its callers and returns
the low 32). Java reads traits through `OpTraits`, one memoized lookup shared by the
DSP compiler, the session and the optimizers. Constant folding skips `RandomOp` and
every custom op that declares `OP_TRAIT_STATEFUL`; `dropout` and `random_crop` now
declare it.

### Samplers

Gamma (Marsaglia and Tsang, with the shape-below-one boost) and Poisson (Knuth below
lambda 10, Hormann's PTRS from 10 on) live once in `helpers/random.h` as host/device
functions used by the CPU and CUDA helpers. Element `e` of a fill draws its own
counter range of the generator (`e * 2^16 + j`), so the result does not depend on how
threads or blocks split the work, and a fill advances the generator once
(`rewindH`). The draws of one element are independent because the generator's
values at neighbouring indices are (see The generator). Parameters are broadcast
and converted to the compute type (float, or double for a double output) once per
fill (`randomParameter`).

### The generator

`RandomGenerator`'s value at an index was a hash of the index and the low 32 bits of
each state. Neighbouring indices hashed into correlated values (a lag-1 correlation
of -0.0475 in the samplers' draw layout), which biased Gamma(0.5) and Poisson(0.5),
and states that differed only in their high bits drew the same stream. The value at
an index is now a Philox4x32-10 block (Salmon et al., SC'11; the generator of cuRAND
and Random123): the key is the root state and the counter is (index, node state), so
every bit of the index and of both states reaches every output bit.

- A float is the top 23 bits of word 0 (`relativeT<float>`), a double the top 52 bits
  of words 1 and 0 (`relativeT<double>`; it used to widen a float unless built with
  `__DOUBLE_RNG__`). Integer and half-precision values derive from these as before.
- The Vulkan lowering of the `RANDOM` recipes computes the same block with 32 x 32 ->
  64-bit multiplies (`arith.mului_extended`), so it needs no Int64 capability; its
  state words carry both halves of each state.
- A seed s sets the states (s, s ^ 0xdeadbeef) everywhere: `applySeedArgument` and the
  backends' `NativeRandom.setSeed`. CPU and CUDA sign-extended the constant to 64 bits
  and Vulkan did not, which the old hash could not see.

## Consequences

- `Nd4j.getRandom().setSeed(s)` reproduces a SameDiff graph's random outputs on both
  paths, and every execution draws new values.
- A plan with random ops keeps those slots out of its captured graphs; plans without
  them are unchanged. Each plan execution makes a few JNI calls to hand the generator
  over.
- Results of the gamma and Poisson ops differ from before for the same state: the
  algorithms and the draw layout changed.
- Every seeded random output differs from before (uniform, normal and Bernoulli fills,
  dropout masks, shuffles and the samplers), on every backend alike. Tests compare
  draws with `PhiloxReference` instead of recorded values.
- Backends whose generated bindings were not regenerated report
  `UnsupportedOperationException` from `getOpTraitMask` until they are.

## References

- `libnd4j/include/graph/RandomGenerator.h` (`philoxBlock`),
  `graph/vulkan/VulkanOpLowerings.cpp` (`RANDOM` recipes)
- `libnd4j/include/graph/DspExecutionRandom.h`, `NativeDynamicShapePlan.h`
  (`drawsRandomState`, `isCapturable`)
- `libnd4j/include/ops/declarable/helpers/random.h`, `helpers/impl/random.cpp`,
  `helpers/impl/random_crop.cpp`
- `InferenceSession.execWithThreadRandom`, `DynamicShapePlanExecutor`
  (`seedContextRandom`, `takeContextRandom`), `OpTraits`,
  `ConstantFunctionOptimizations`
- `platform-tests`: `SameDiffRandomStateTest`, `RandomGeneratorPhiloxTest` (with
  `PhiloxReference`, checked against Random123's known answers)
