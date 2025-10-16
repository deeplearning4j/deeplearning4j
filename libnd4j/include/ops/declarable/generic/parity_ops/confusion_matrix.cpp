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
// @author @cpuheater
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_confusion_matrix)

#include <array/NDArray.h>
#include <array/NDArrayList.h>
#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/confusion.h>

#include <array>

namespace sd {
namespace ops {

CUSTOM_OP_IMPL(confusion_matrix, 2, 1, false, 0, -2) {
  auto labels = INPUT_VARIABLE(0);
  auto predictions = INPUT_VARIABLE(1);
  NDArray *weights = nullptr;
  if (block.width() > 2) {
    weights = INPUT_VARIABLE(2);
    REQUIRE_TRUE(weights->isSameShape(predictions), 0,
                 "CONFUSION_MATRIX: Weights and predictions should have equal shape");
  }
  auto output = OUTPUT_NULLIFIED(0);

  auto* minPredictionArr = predictions->reduceNumber(reduce::Min);
  int minPrediction = minPredictionArr->e<int>(0);
  delete minPredictionArr;
  
  auto* minLabelArr = labels->reduceNumber(reduce::Min);
  int minLabel = minLabelArr->e<int>(0);
  delete minLabelArr;

  REQUIRE_TRUE(minLabel >= 0, 0, "CONFUSION_MATRIX: Labels contains negative values !");
  REQUIRE_TRUE(minPrediction >= 0, 0, "CONFUSION_MATRIX: Predictions contains negative values !");
  REQUIRE_TRUE(labels->isVector(), 0, "CONFUSION_MATRIX: Labels input should be a Vector, but got %iD instead",
               labels->rankOf());
  REQUIRE_TRUE(predictions->isVector(), 0, "CONFUSION_MATRIX: Predictions input should be Vector, but got %iD instead",
               predictions->rankOf());
  REQUIRE_TRUE(labels->isSameShape(predictions), 0, "CONFUSION_MATRIX: Labels and predictions should have equal shape");

  helpers::confusionFunctor(block.launchContext(), labels, predictions, weights, output);

  return sd::Status::OK;
}

DECLARE_SHAPE_FN(confusion_matrix) {
  auto labels = INPUT_VARIABLE(0);
  auto predictions = INPUT_VARIABLE(1);
  auto dtype = block.numD() ? D_ARG(0) : sd::DataType::INT64;
  int numClasses = 0;

  if (block.getIArguments()->size() > 0) {
    numClasses = INT_ARG(0);
  } else {
    auto* maxPredictionArr = predictions->reduceNumber(reduce::Max);
    int maxPrediction = maxPredictionArr->e<int>(0);
    delete maxPredictionArr;
    
    auto* maxLabelArr = labels->reduceNumber(reduce::Max);
    int maxLabel = maxLabelArr->e<int>(0);
    delete maxLabelArr;
    
    numClasses = (maxPrediction >= maxLabel) ? maxPrediction + 1 : maxLabel + 1;
  }

  std::array<sd::LongType, 2> shape = {{numClasses, numClasses}};
  auto newShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', 2, shape.data(),0);
  return SHAPELIST(newShape);
}

DECLARE_TYPES(confusion_matrix) {
  getOpDescriptor()
      ->setAllowedInputTypes({ALL_INDICES})
      ->setAllowedOutputTypes({ALL_INDICES});
}

}  // namespace ops
}  // namespace sd

#endif
