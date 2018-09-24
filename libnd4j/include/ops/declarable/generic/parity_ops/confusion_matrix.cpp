/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_confusion_matrix)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>
#include <NDArray.h>
#include <array/NDArrayList.h>
#include <array>
#include <ops/declarable/helpers/confusion.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(confusion_matrix, 2, 1, false, 0, -2) {

            auto labels = INPUT_VARIABLE(0);
            auto predictions = INPUT_VARIABLE(1);
            NDArray *weights = nullptr;
            if(block.width() > 2){
                weights = INPUT_VARIABLE(2);
                REQUIRE_TRUE(weights->isSameShape(predictions),0, "CONFUSION_MATRIX: Weights and predictions should have equal shape");
            }
            auto output = OUTPUT_VARIABLE(0);

            int minPrediction = predictions->reduceNumber(reduce::Min).e<int>(0);
            int minLabel = labels->reduceNumber(reduce::Min).e<int>(0);

            REQUIRE_TRUE(minLabel >=0, 0, "CONFUSION_MATRIX: Labels contains negative values !");
            REQUIRE_TRUE(minPrediction >=0, 0, "CONFUSION_MATRIX: Predictions contains negative values !");
            REQUIRE_TRUE(labels->isVector(), 0, "CONFUSION_MATRIX: Labels input should be a Vector, but got %iD instead", labels->rankOf());
            REQUIRE_TRUE(predictions->isVector(), 0, "CONFUSION_MATRIX: Predictions input should be Vector, but got %iD instead", predictions->rankOf());
            REQUIRE_TRUE(labels->isSameShape(predictions),0, "CONFUSION_MATRIX: Labels and predictions should have equal shape");

            helpers::confusionFunctor(labels, predictions, weights, output);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(confusion_matrix) {
            auto labels = INPUT_VARIABLE(0);
            auto predictions = INPUT_VARIABLE(1);

            int numClasses = 0;

            if (block.getIArguments()->size() > 0) {
                numClasses = INT_ARG(0);
            }
            else  {
                int maxPrediction = predictions->reduceNumber(reduce::Max).e<int>(0);
                int maxLabel = labels->reduceNumber(reduce::Max).e<int>(0);
                numClasses = (maxPrediction >= maxLabel) ?  maxPrediction+1 : maxLabel+1;
            }

            Nd4jLong *newShape;
            std::array<Nd4jLong, 2> shape = {{numClasses,numClasses}};
            ALLOCATE(newShape, block.getWorkspace(), shape::shapeInfoLength(2), Nd4jLong);
            shape::shapeBuffer(2, block.dataType(), shape.data(), newShape);

            return SHAPELIST(newShape);
        }
    }
}

#endif