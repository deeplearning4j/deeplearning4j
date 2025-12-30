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
// @author Adam Gibson
// CPU implementations for mixed precision training helpers
//

#include <ops/declarable/helpers/mixedPrecision.h>
#include <math/templatemath.h>
#include <execution/Threads.h>
#include <helpers/LoopKind.h>

namespace sd {
namespace ops {
namespace helpers {

void scaleGradients(sd::LaunchContext* context, NDArray* gradients, double scale) {
  if (gradients == nullptr || gradients->isEmpty()) {
    return;
  }

  // Use in-place scalar multiply
  gradients->applyScalar(sd::scalar::Multiply, scale, *gradients);
}

bool hasInfOrNan(sd::LaunchContext* context, const NDArray* arr) {
  if (arr == nullptr || arr->isEmpty()) {
    return false;  // Empty arrays are considered valid
  }

  auto length = arr->lengthOf();
  bool foundBad = false;

  // Use atomic for thread-safe detection
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop && !foundBad; i++) {
      double val = arr->e<double>(i);
      if (sd::math::sd_isnan<double>(val) || sd::math::sd_isinf<double>(val)) {
        foundBad = true;
      }
    }
  };

  // For small arrays, just do sequential check
  if (length < 1024) {
    for (sd::LongType i = 0; i < length; i++) {
      double val = arr->e<double>(i);
      if (sd::math::sd_isnan<double>(val) || sd::math::sd_isinf<double>(val)) {
        return true;
      }
    }
    return false;
  }

  // For larger arrays, use parallel check
  samediff::Threads::parallel_for(func, 0, length);
  return foundBad;
}

void castAndScale(sd::LaunchContext* context, const NDArray* input,
                  NDArray* output, double scale) {
  if (input == nullptr || output == nullptr) {
    return;
  }

  if (input->isEmpty()) {
    return;
  }

  auto length = input->lengthOf();

  // Fast path: same type and no scaling needed
  if (input->dataType() == output->dataType() && scale == 1.0) {
    output->assign(input);
    return;
  }

  // Fast path: same type, just scale
  if (input->dataType() == output->dataType()) {
    input->applyScalar(sd::scalar::Multiply, scale, *output);
    return;
  }

  // Different types: cast and scale
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      double val = input->e<double>(i) * scale;
      output->p(i, val);
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

bool unscaleGradientsAndCheck(sd::LaunchContext* context, NDArray* gradients,
                               double lossScale) {
  if (gradients == nullptr || gradients->isEmpty()) {
    return true;  // Empty gradients are valid
  }

  if (lossScale == 1.0) {
    // No scaling needed, just check for overflow
    return !hasInfOrNan(context, gradients);
  }

  auto length = gradients->lengthOf();
  double invScale = 1.0 / lossScale;
  bool allFinite = true;

  // Unscale and check in one pass
  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop && allFinite; i++) {
      double val = gradients->e<double>(i) * invScale;
      if (sd::math::sd_isnan<double>(val) || sd::math::sd_isinf<double>(val)) {
        allFinite = false;
      } else {
        gradients->p(i, val);
      }
    }
  };

  // Sequential for small arrays
  if (length < 1024) {
    for (sd::LongType i = 0; i < length; i++) {
      double val = gradients->e<double>(i) * invScale;
      if (sd::math::sd_isnan<double>(val) || sd::math::sd_isinf<double>(val)) {
        return false;
      }
      gradients->p(i, val);
    }
    return true;
  }

  samediff::Threads::parallel_for(func, 0, length);
  return allFinite;
}

void updateMasterWeight(sd::LaunchContext* context, NDArray* masterWeight,
                        const NDArray* gradient, double learningRate,
                        double lossScale) {
  if (masterWeight == nullptr || gradient == nullptr) {
    return;
  }

  if (masterWeight->isEmpty() || gradient->isEmpty()) {
    return;
  }

  auto length = masterWeight->lengthOf();
  double scaledLr = learningRate / lossScale;

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      double w = masterWeight->e<double>(i);
      double g = gradient->e<double>(i);
      // w = w - lr * (g / lossScale) = w - (lr/lossScale) * g
      masterWeight->p(i, w - scaledLr * g);
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

void syncMasterToCompute(sd::LaunchContext* context, const NDArray* masterWeight,
                         NDArray* computeWeight) {
  if (masterWeight == nullptr || computeWeight == nullptr) {
    return;
  }

  if (masterWeight->isEmpty()) {
    return;
  }

  // If same type, just assign
  if (masterWeight->dataType() == computeWeight->dataType()) {
    computeWeight->assign(masterWeight);
    return;
  }

  // Different types - need to cast
  auto length = masterWeight->lengthOf();

  auto func = PRAGMA_THREADS_FOR {
    for (auto i = start; i < stop; i++) {
      computeWeight->p(i, masterWeight->e<double>(i));
    }
  };

  samediff::Threads::parallel_for(func, 0, length);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
