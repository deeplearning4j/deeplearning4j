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


#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_choose)

#include <array/NDArrayFactory.h>
#include <ops/declarable/helpers/choose.h>
#include <ops/ops.h>

namespace sd {
namespace ops {
namespace helpers {
template <typename T>
static NDArray* processCondition_(int mode, NDArray* arg, NDArray* comp, NDArray& compScalar);

template <typename T>
static T processElementCondition(int mode, T d1, T d2);

template <typename T>
NDArray* processCondition_(int mode, NDArray* arg, NDArray* comp, NDArray* output, NDArray* numResult,
                           NDArray& compScalar) {
  // Convert to straight ndarray based on input

  int numResults = 0;
  if (comp != nullptr) {
    if (comp->isScalar()) {
      // Other input for compare could be an ndarray or a secondary scalar
      // for comparison
      //                sd::NDArray arg1 = *arg;
      //                sd::NDArray comp1 = *comp;
      for (LongType i = 0; i < arg->lengthOf(); i++) {
        T result2 = processElementCondition(mode, arg->e<T>(i), comp->e<T>(0));
        if (result2 > static_cast<T>(0)) {
          if (output != nullptr) output->p(numResults, arg->e<T>(i));
          numResults++;
        }
      }
    } else {
      // Other input for compare could be an ndarray or a secondary scalar
      // for comparison
      NDArray arg1 = *arg;
      for (LongType i = 0; i < arg->lengthOf(); i++) {
        T result2 = processElementCondition(mode, arg->e<T>(i), comp->e<T>(i));
        if (result2 > static_cast<T>(0)) {
          if (output != nullptr) output->p(numResults, arg->e<T>(i));
          numResults++;
        }
      }
    }

  } else {
    //        sd::NDArray arg1 = *arg;
    // Other input for compare could be an ndarray or a secondary scalar
    // for comparison
    for (LongType i = 0; i < arg->lengthOf(); i++) {
      T result2 = processElementCondition(mode, arg->e<T>(i), compScalar.e<T>(0));
      if (result2 > static_cast<T>(0)) {
        if (output != nullptr) output->p(numResults, arg->e<T>(i));
        numResults++;
      }
    }
  }

  if (numResult != nullptr) numResult->p(0, numResults);

  return output;
}

NDArray* processCondition(LaunchContext* context, int mode, NDArray* arg, NDArray* comp, NDArray* output,
                          NDArray* numResult, NDArray& compScalar) {
  arg->syncToHost();

  if (comp != nullptr) comp->syncToHost();

  if (output != nullptr) output->syncToHost();

  if (numResult != nullptr) numResult->syncToHost();

  compScalar.syncToHost();

  BUILD_SINGLE_SELECTOR(arg->dataType(), return processCondition_, (mode, arg, comp, output, numResult, compScalar),
                        SD_FLOAT_TYPES);

  arg->syncToDevice();

  if (comp != nullptr) comp->syncToDevice();

  if (output != nullptr) output->syncToDevice();

  if (numResult != nullptr) numResult->syncToDevice();

  compScalar.syncToDevice();
  return nullptr;
}
BUILD_SINGLE_TEMPLATE(template NDArray* processCondition_,
                      (int mode, sd::NDArray* arg, sd::NDArray* comp, sd::NDArray* output, sd::NDArray* numResult,
                          sd::NDArray& compScalar),
                      SD_FLOAT_TYPES);

template <typename T>
T processElementCondition(int mode, T d1, T d2) {
  T input[3] = {d2, (T)SD_EPSILON, (T)mode};
  T res = simdOps::MatchCondition<T, T>::op(d1, input);
  return res;
}

void chooseFunctorArray(LaunchContext* context, NDArray* arg, NDArray* comp, int mode, NDArray* result,
                        NDArray* numResults) {
  if (arg->isScalar() || comp->isScalar()) {
    if (arg->isScalar()) {
      processCondition(context, mode, comp, nullptr, result, numResults, *arg);
    } else {
      processCondition(context, mode, arg, nullptr, result, numResults, *comp);
    }
  } else {
    auto zero = NDArrayFactory::create<float>(0);
    processCondition(context, mode, arg, comp, result, numResults, zero);
  }
}

void chooseFunctorScalar(LaunchContext* context, NDArray* arg, double scalar, int mode, NDArray* result,
                         NDArray* numResults) {
  auto scalarA = NDArrayFactory::create(scalar);
  processCondition(context, mode, arg, nullptr, result, numResults, scalarA);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif