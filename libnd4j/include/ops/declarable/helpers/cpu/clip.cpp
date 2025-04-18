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
// @author Yurii Shyrma (iuriish@yahoo.com)
// @author sgazeos@gmail.com
// @author raver119@gmail.com
//

#include <execution/Threads.h>
#include <ops/declarable/helpers/transforms.h>

#if NOT_EXCLUDED(OP_clip)
namespace sd {
namespace ops {
namespace helpers {

//////////////////////////////////////////////////////////////////////////
void clipByNorm(LaunchContext* context, NDArray* input, NDArray* output, const std::vector<LongType>& dimensions,
                NDArray* clipNorm, const bool isInplace, const bool useAverage) {
  NDArray* z = nullptr;

  if (isInplace) {
    z = input;
  } else {
    output->assign(input);
    z = output;
  }

  if (dimensions.empty()) {
    std::vector<sd::LongType> emptyVec = {};

    NDArray actualNorm = useAverage ? z->reduceAlongDimension(reduce::Norm2, &emptyVec) / z->lengthOf()
                                          : z->reduceAlongDimension(reduce::Norm2, &emptyVec);
    int idx = 0;
    if (actualNorm.e<float>(0) > clipNorm->e<float>(0)) *z *= *clipNorm / actualNorm;
  } else {
    auto listOfSubArrs = z->allTensorsAlongDimension(dimensions);

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        std::vector<sd::LongType> emptyVec = {};
         NDArray actualNorm =
            useAverage ? listOfSubArrs.at(i)->reduceAlongDimension(reduce::Norm2, &emptyVec) / listOfSubArrs.at(i)->lengthOf()
                       : listOfSubArrs.at(i)->reduceAlongDimension(reduce::Norm2, &emptyVec);
        if (actualNorm.e<float>(0) > clipNorm->e<float>(0)) *listOfSubArrs.at(i) *= *clipNorm / actualNorm;
      }
    };
    samediff::Threads::parallel_tad(func, 0, listOfSubArrs.size());
  }
}

//////////////////////////////////////////////////////////////////////////
template <typename T>
static void clipByNormBp_(NDArray *input, NDArray *gradO, NDArray *gradI,
                          const std::vector<LongType>& dimensions, NDArray *clipNorm, const bool useAverage) {
  const int rank = input->rankOf();

  auto norm2 = input->reduceAlongDimension(reduce::Norm2, &dimensions);
  auto sums = input->reduceAlongDimension(reduce::Sum, &dimensions);

  if (norm2.lengthOf() == 1) {
    const T norm = useAverage ? norm2.e<T>(0) / input->lengthOf() : norm2.e<T>(0);

    auto clipVal = clipNorm->e<T>(0);

    if (norm > clipVal) {
      const T sum = sums.e<T>(0);  // reduce to scalar
      const T factor1 = clipVal / norm;
      const T factor2 = static_cast<T>(1.f) / (norm * norm);  // 1 / (norm*norm*norm)

      auto lambda = LAMBDA_TT(x, y, sum, factor1, factor2) {
        return factor1 * y * (static_cast<T>(1.f) - factor2 * x * sum);
      });

      input->applyPairwiseLambda<T>(gradO, lambda, gradI);
    } else
      gradI->assign(gradO);
  } else {
    auto gradISubArrs = gradI->allTensorsAlongDimension({dimensions});
    auto gradOSubArrs = gradO->allTensorsAlongDimension({dimensions});
    auto inputSubArrs = input->allTensorsAlongDimension({dimensions});

    auto clipVal = clipNorm->e<T>(0);

    auto func = PRAGMA_THREADS_FOR {
      for (auto i = start; i < stop; i++) {
        auto gradOSubArr = gradOSubArrs.at(i);
        auto gradISubArr = gradISubArrs.at(i);

        const T norm = useAverage ? norm2.e<T>(i) / gradISubArr->lengthOf() : norm2.e<T>(i);

        if (norm > clipVal) {
          auto inputSubArr = inputSubArrs.at(i);

          const T sum = sums.e<T>(i);  // reduce to scalar
          const T factor1 = clipVal / norm;
          const T factor2 = static_cast<T>(1.f) / (norm * norm);  // 1 / (norm*norm*norm)

          auto lambda = LAMBDA_TT(x, y, sum, factor1, factor2) {
            return factor1 * y * (static_cast<T>(1.f) - factor2 * x * sum);
          });

          inputSubArr->applyPairwiseLambda<T>(gradOSubArr, lambda, gradISubArr);
        } else
          gradISubArr->assign(gradOSubArr);
      }
    };
    samediff::Threads::parallel_tad(func, 0, gradISubArrs.size());
  }
}
BUILD_SINGLE_TEMPLATE(template void clipByNormBp_,
                      (NDArray *input, NDArray *gradO, NDArray *gradI, const std::vector<sd::LongType>& dimensions,
                       NDArray *clipNorm, const bool useAverage),
                      SD_FLOAT_TYPES);

//////////////////////////////////////////////////////////////////////////
void clipByNormBp(sd::LaunchContext* context, NDArray *input, NDArray  *gradO, NDArray *gradI,
                  const std::vector<LongType>& dimensions, NDArray* clipNorm, const bool useAverage) {
  BUILD_SINGLE_SELECTOR(gradI->dataType(), clipByNormBp_, (input, gradO, gradI, dimensions, clipNorm, useAverage),
                        SD_FLOAT_TYPES);
}

template <typename T>
static void clipByGlobalNorm_(std::vector<NDArray*>& inputs, double clipNorm, sd::memory::Workspace* workspace,
                              std::vector<NDArray*>& outputs, bool isInplace) {
  T globalNorm = 0;
  for (size_t i = 0; i < inputs.size(); i++) {
    auto input = inputs[i];
    auto l2norm = input->reduceNumber(reduce::Norm2);
    globalNorm += l2norm.t<T>(0) * l2norm.t<T>(0);
  }

  auto normS = sd::math::sd_sqrt<T, T>(globalNorm);
  outputs[inputs.size()]->p(0, normS);

  const T factor = clipNorm / normS;

  for (size_t e = 0; e < inputs.size(); e++) {
    // all-reduce
    auto input = inputs[e];
    auto output = outputs[e];

    if (normS <= clipNorm) {
      output->assign(input);
    } else {
      auto lambda = LAMBDA_T(_x, factor) { return _x * factor; });
      input->applyLambda<T>(lambda, output);
    }
  }
}
void clipByGlobalNorm(LaunchContext* context, std::vector<NDArray*>& inputs, double clipNorm,
                      memory::Workspace* workspace, std::vector<NDArray*>& outputs, bool isInplace) {
  BUILD_SINGLE_SELECTOR(outputs[0]->dataType(), clipByGlobalNorm_, (inputs, clipNorm, workspace, outputs, isInplace),
                        SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void clipByGlobalNorm_,
                      (std::vector<NDArray*> & inputs, double clipNorm, sd::memory::Workspace* workspace,
                       std::vector<NDArray*>& outputs, bool isInplace),
                      SD_FLOAT_TYPES);

template <typename T>
static void clipByValue_(NDArray* input, double leftBound, double rightBound, NDArray* output) {
  auto routine = LAMBDA_T(_x, leftBound, rightBound) {
    if (_x > rightBound) return rightBound;
    if (_x < leftBound) return leftBound;
    return _x;
  });

  input->applyLambda<T>(routine, output);
}

void clipByValue(LaunchContext* context, NDArray* input, double leftBound, double rightBound, NDArray* output) {
  BUILD_SINGLE_SELECTOR(input->dataType(), clipByValue_, (input, leftBound, rightBound, output), SD_FLOAT_TYPES);
}

BUILD_SINGLE_TEMPLATE(template void clipByValue_,
                      (NDArray * input, double leftBound, double rightBound, NDArray* output);
                      , SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif