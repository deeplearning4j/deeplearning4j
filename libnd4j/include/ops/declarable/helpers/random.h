/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

//
//  @author sgazeos@gmail.com
//
//
// Declaration of distribution helpers
//
#ifndef __RANDOM_HELPERS__
#define __RANDOM_HELPERS__
#include <array/NDArray.h>
#include <graph/Context.h>
#include <helpers/helper_random.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

/**
 * Applies a random op's seed argument: a nonzero seed fixes both generator states, so the op's
 * draws depend on the seed alone; 0 keeps the context's generator, which its caller seeds
 * (SameDiff from Nd4j.getRandom()).
 */
SD_INLINE void applySeedArgument(graph::RandomGenerator& rng, LongType seed) {
  if (seed != 0) rng.setStates(seed, seed ^ 0xdeadbeef);
}

/**
 * Draw streams of the rejection samplers. Draw j (j = 0, 1, ...) of element e of a fill is the
 * generator's value at spreadDrawIndex(e * kRandomDrawsPerElement + j), so every element has its
 * own reproducible sequence however threads or CUDA blocks split the work. A fill rewinds the
 * generator once afterwards (RandomGenerator::rewindH) so the next fill draws anew.
 */
constexpr LongType kRandomDrawsPerElement = LongType{1} << 16;

/**
 * splitmix64's finalizer, a bijection. RandomGenerator hashes nearby indices into correlated
 * values (adjacent indices correlate at about -0.05), which biases a sampler that combines an
 * element's consecutive draws: Box-Muller pairs, Knuth's product, the shape boost. Spreading the
 * draw indices over the whole index space removes that dependence.
 */
SD_HOST_DEVICE SD_INLINE uint64_t spreadDrawIndex(uint64_t index) {
  index += 0x9E3779B97F4A7C15ULL;
  index = (index ^ (index >> 30)) * 0xBF58476D1CE4E5B9ULL;
  index = (index ^ (index >> 27)) * 0x94D049BB133111EBULL;
  return index ^ (index >> 31);
}

/** The next uniform value of an element's stream, in (0, 1], so log() of it is finite. */
template <typename T>
SD_HOST_DEVICE SD_INLINE T streamUniform(graph::RandomGenerator& rng, LongType element, LongType& draw) {
  const uint64_t streamIndex = static_cast<uint64_t>(element) * static_cast<uint64_t>(kRandomDrawsPerElement) +
                               static_cast<uint64_t>(draw++ & (kRandomDrawsPerElement - 1));
  return T(1) - rng.relativeT<T>(static_cast<LongType>(spreadDrawIndex(streamIndex)));
}

/** The next standard normal value of an element's stream (Box-Muller). */
template <typename T>
SD_HOST_DEVICE SD_INLINE T streamNormal(graph::RandomGenerator& rng, LongType element, LongType& draw) {
  const T radius = math::sd_sqrt<T, T>(T(-2) * math::sd_log<T, T>(streamUniform<T>(rng, element, draw)));
  return radius * math::sd_cos<T, T>(T(6.283185307179586) * streamUniform<T>(rng, element, draw));
}

/**
 * A Gamma(alpha) sample divided by rate, computed in T (float or double): Marsaglia and Tsang's
 * method (2000); a shape below 1 is boosted through Gamma(alpha + 1) * U^(1 / alpha). A shape or
 * rate that is not positive has no distribution: NaN.
 */
template <typename T>
SD_HOST_DEVICE SD_INLINE T sampleGamma(graph::RandomGenerator& rng, LongType element, T alpha, T rate) {
  if (!(alpha > T(0)) || !(rate > T(0))) return DataTypeUtils::nanOrZero<T>();
  LongType draw = 0;
  const bool boost = alpha < T(1);
  const T d = (boost ? alpha + T(1) : alpha) - T(1) / T(3);
  const T c = T(1) / math::sd_sqrt<T, T>(T(9) * d);
  T sample;
  for (;;) {
    T x, v;
    do {
      x = streamNormal<T>(rng, element, draw);
      v = T(1) + c * x;
    } while (v <= T(0));
    v = v * v * v;
    const T u = streamUniform<T>(rng, element, draw);
    const T x2 = x * x;
    if (u < T(1) - T(0.0331) * x2 * x2 ||
        math::sd_log<T, T>(u) < T(0.5) * x2 + d * (T(1) - v + math::sd_log<T, T>(v))) {
      sample = d * v;
      break;
    }
  }
  if (boost) sample *= math::sd_pow<T, T, T>(streamUniform<T>(rng, element, draw), T(1) / alpha);
  return sample / rate;
}

/**
 * A Poisson(lambda) sample, computed in T (float or double): Knuth's multiplication method below
 * lambda 10 and Hormann's transformed rejection with squeeze (PTRS, 1993) from 10 on, where
 * exp(-lambda) would bound a sequential search ever more tightly and underflows past about 87.
 * Lambda 0 gives 0; a negative or NaN lambda has no distribution: NaN.
 */
template <typename T>
SD_HOST_DEVICE SD_INLINE T samplePoisson(graph::RandomGenerator& rng, LongType element, T lambda) {
  if (lambda == T(0)) return T(0);
  if (!(lambda > T(0))) return DataTypeUtils::nanOrZero<T>();
  LongType draw = 0;
  if (lambda < T(10)) {
    const T bound = math::sd_exp<T, T>(-lambda);
    T product = streamUniform<T>(rng, element, draw);
    T count = T(0);
    while (product > bound) {
      count += T(1);
      product *= streamUniform<T>(rng, element, draw);
    }
    return count;
  }
  const T slam = math::sd_sqrt<T, T>(lambda);
  const T logLambda = math::sd_log<T, T>(lambda);
  const T b = T(0.931) + T(2.53) * slam;
  const T a = T(-0.059) + T(0.02483) * b;
  const T logInvAlpha = math::sd_log<T, T>(T(1.1239) + T(1.1328) / (b - T(3.4)));
  const T vr = T(0.9277) - T(3.6224) / (b - T(2));
  for (;;) {
    const T u = streamUniform<T>(rng, element, draw) - T(0.5);
    const T v = streamUniform<T>(rng, element, draw);
    const T us = T(0.5) - math::sd_abs<T, T>(u);
    const T k = math::sd_floor<T, T>((T(2) * a / us + b) * u + lambda + T(0.43));
    if (us >= T(0.07) && v <= vr) return k;
    if (k < T(0) || (us < T(0.013) && v > us)) continue;
    if (math::sd_log<T, T>(v) + logInvAlpha - math::sd_log<T, T>(a / (us * us) + b) <=
        -lambda + k * logLambda - math::sd_lgamma<T, T>(k + T(1)))
      return k;
  }
}

/**
 * A distribution parameter as a C-ordered array of computeType holding values broadcast to
 * shapeInfo's shape, so a sampler reads parameter p at index p whatever the input's dtype, order
 * or strides. The caller deletes it.
 */
SD_LIB_HIDDEN NDArray* randomParameter(NDArray* values, const LongType* shapeInfo, DataType computeType,
                                       LaunchContext* context);

SD_LIB_HIDDEN void fillRandomGamma(LaunchContext* context, graph::RandomGenerator& rng, NDArray* alpha, NDArray* beta,
                                   NDArray* output);
SD_LIB_HIDDEN void fillRandomPoisson(LaunchContext* context, graph::RandomGenerator& rng, NDArray* lambda,
                                     NDArray* output);
SD_LIB_HIDDEN void fillRandomUniform(LaunchContext* context, graph::RandomGenerator& rng, NDArray* min, NDArray* max,
                                     NDArray* output);
SD_LIB_HIDDEN void fillRandomMultiNomial(LaunchContext* context, graph::RandomGenerator& rng, NDArray& input,
                                         NDArray& output, const LongType numOfSamples, const int dimC);
}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif
