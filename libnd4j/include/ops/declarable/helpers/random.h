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
#include <ops/declarable/helpers/random_samplers.h>
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
