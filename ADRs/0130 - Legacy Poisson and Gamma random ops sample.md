# ADR 0130: Legacy Poisson and Gamma random ops sample

## Status

Accepted (2026-10-03). Builds on ADR 0126 (random state in SameDiff execution)
and on the Philox generator (`RandomGenerator`, one independent value per
index).

## Context

The legacy random ops 15 (`PoissonDistribution`) and 16 (`GammaDistribution`)
in `RANDOM_OPS` did not sample. Given an input x they evaluated a distribution
function at it: `igammac(floor(x), lambda)`, the Poisson CDF below x, and
`igamma(alpha, beta x)`, the Gamma CDF. Without one they evaluated it at a
uniform draw over (-max / 10, max / 10), so almost every element was 0 or about
1. No Java class exposed them, the SameDiff wrapper (`LegacyRandomOp`) rejected
them as unknown op numbers, and the Vulkan port mirrored the CDFs.

The `random_poisson` and `random_gamma` custom ops already sample, through
`helpers::samplePoisson` and `helpers::sampleGamma`.

## Decision

Ops 15 and 16 sample with those same per-element samplers. The samplers and
their draw streams move from `helpers/random.h` into
`helpers/random_samplers.h`, which holds only scalar math and the generator,
so the legacy op header (`random_ops.h`) includes it; the helpers include it
too, and the per-backend copies of the compute type alias
(`RandomComputeT`) are replaced by its single definition.

- `PoissonDistribution`: element e of z is a Poisson(lambda) sample, lambda the
  first extra argument, or element e of x when x is given. Knuth's
  multiplication method below lambda 10, Hormann's transformed rejection with
  squeeze (PTRS) from 10 on. lambda 0 gives 0, a negative or NaN lambda NaN.
- `GammaDistribution`: element e of z is a Gamma(alpha, beta) sample with shape
  alpha and rate beta (mean alpha / beta), the two extra arguments; element e of
  x replaces alpha when x is given, and element e of y replaces beta when y is
  given too. Marsaglia and Tsang's method, a shape below 1 boosted through
  Gamma(alpha + 1) U^(1 / alpha). A shape or rate that is not positive gives NaN.
- Element e draws from its own stream: draw j is the generator's value at
  e * 2^16 + j, so a sample does not depend on how threads or CUDA blocks split
  the work, and the generator is rewound once per fill as for every legacy
  random op.
- Samples are computed in float, or in double for a DOUBLE output; HALF and
  BFLOAT16 outputs are rounded from the float sample.

Java exposes them as `PoissonDistribution` and `GammaDistribution` in
`org.eclipse.deeplearning4j.nd4j.linalg.api.ops.random.impl` (op names
`distribution_poisson` and `distribution_gamma`), registered for deserialization
and in `LegacyOpMapper`. Their parameter arrays must have the output's shape and
are cast to the output's type. `LegacyRandomOp` runs both in SameDiff graphs
(scalar inputs or floating point arguments, as the other distributions), through
`RandomLauncher::fillPoisson` and `fillGamma`.

A legacy random op reads x and y as its output's type. The CPU and CUDA entry
points (`NativeOpExecutioner::execRandom`) now reject an x or y of another type;
before, its bits were read as the output's type.

The Vulkan lowering of ops 15 and 16 implements the same samplers and streams
(the Vulkan backend's legacy random lowering).

## Consequences

- Ops 15 and 16 return samples; nothing called them for their old values.
- A legacy random op given a parameter array of another type than its output
  fails instead of computing from reinterpreted bits.

Tests: `LegacyPoissonGammaSamplerTest` (Knuth samples reproduced exactly from the
Philox reference for FLOAT, DOUBLE, HALF and BFLOAT16, per-element rates, moments
and a chi-square test of the PTRS path, Gamma moments for shapes below and
above 1 and per-element parameters, degenerate parameters, reproducibility,
SameDiff execution and the type check).
