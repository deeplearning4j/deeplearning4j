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
// @author raver119@gmail.com, created on 6/30/2018
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include <helpers/OmpLaunchHelper.h>
#include <math/templatemath.h>
#include <system/Environment.h>

namespace sd {

////////////////////////////////////////////////////////////////////////////////
OmpLaunchHelper::OmpLaunchHelper(const LongType N, float desiredNumThreads) {
  auto maxItersPerThread = Environment::getInstance().elementwiseThreshold();

  if (N < maxItersPerThread)
    _numThreads = 1;
  else {
#ifdef _OPENMP
    if (desiredNumThreads == -1)
      desiredNumThreads = omp_get_max_threads();
    else if (desiredNumThreads < 1)
      desiredNumThreads = 1;
    else
      desiredNumThreads = sd::math::sd_min<int>(omp_get_max_threads(), desiredNumThreads);
#else
    desiredNumThreads = Environment::getInstance().maxThreads();
#endif
    _numThreads = sd::math::sd_min<int>(N / maxItersPerThread, desiredNumThreads);
  }

  _itersPerThread = N / _numThreads;
  _remainder = N % _numThreads;  // last thread may contain bigger number of iterations
}

LongType OmpLaunchHelper::betterSpan(LongType N) { return betterSpan(N, betterThreads(N)); }

LongType OmpLaunchHelper::betterSpan(LongType N, LongType numThreads) {
  auto r = N % numThreads;
  auto t = N / numThreads;

  if (r == 0)
    return t;
  else {
    // breaks alignment
    return t + 1;
  }
}

int OmpLaunchHelper::betterThreads(LongType N) {
#ifdef _OPENMP
  return betterThreads(N, omp_get_max_threads());
#else
  return betterThreads(N, Environment::getInstance().maxThreads());
  ;
#endif
}

int OmpLaunchHelper::betterThreads(LongType N, int maxThreads) {
  auto t = Environment::getInstance().elementwiseThreshold();
  if (N < t)
    return 1;
  else {
    return static_cast<int>(sd::math::sd_min<LongType>(N / t, maxThreads));
  }
}

int OmpLaunchHelper::tadThreads(LongType tadLength, LongType numTads) {
#ifdef _OPENMP
  auto maxThreads = omp_get_max_threads();
#else
  auto maxThreads = Environment::getInstance().maxThreads();
#endif

  // if there's only 1 thread allowed - nothing to do here
  if (maxThreads <= 1) return 1;

  auto totalLength = tadLength * numTads;

  // if array is tiny - no need to spawn any threeds
  if (totalLength < Environment::getInstance().elementwiseThreshold()) return 1;

  // by default we're spawning as many threads we can, but not more than number of TADs
  return sd::math::sd_min<int>(numTads, maxThreads);
}
}  // namespace sd
