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
  auto argBuf = arg->bufferAsT<T>();
  T* outputBuf = (output != nullptr) ? output->bufferAsT<T>() : nullptr;

  if (comp != nullptr) {
    if (comp->isScalar()) {
      T compVal = comp->bufferAsT<T>()[0];
      for (LongType i = 0; i < arg->lengthOf(); i++) {
        T result2 = processElementCondition(mode, argBuf[i], compVal);
        if (result2 > static_cast<T>(0)) {
          if (outputBuf != nullptr) outputBuf[numResults] = argBuf[i];
          numResults++;
        }
      }
    } else {
      auto compBuf = comp->bufferAsT<T>();
      for (LongType i = 0; i < arg->lengthOf(); i++) {
        T result2 = processElementCondition(mode, argBuf[i], compBuf[i]);
        if (result2 > static_cast<T>(0)) {
          if (outputBuf != nullptr) outputBuf[numResults] = argBuf[i];
          numResults++;
        }
      }
    }

  } else {
    T compScalarVal = compScalar.bufferAsT<T>()[0];
    for (LongType i = 0; i < arg->lengthOf(); i++) {
      T result2 = processElementCondition(mode, argBuf[i], compScalarVal);
      if (result2 > static_cast<T>(0)) {
        if (outputBuf != nullptr) outputBuf[numResults] = argBuf[i];
        numResults++;
      }
    }
  }

  if (output != nullptr) {
    output->tickWriteHost();
    output->syncToDevice();
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
BUILD_SINGLE_TEMPLATE( NDArray* processCondition_,
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
    processCondition(context, mode, arg, comp, result, numResults, *zero);
    delete zero;
  }
}

void chooseFunctorScalar(LaunchContext* context, NDArray* arg, double scalar, int mode, NDArray* result,
                         NDArray* numResults) {
  auto scalarA = NDArrayFactory::create(scalar);
  processCondition(context, mode, arg, nullptr, result, numResults, *scalarA);
  delete scalarA;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
#endif