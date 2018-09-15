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
//  @author GS <sgazeos@gmail.com>
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_dynamic_stitch)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/dynamic.h>

namespace nd4j {
namespace ops {
    CUSTOM_OP_IMPL(dynamic_stitch, 2, 1, false, 0, 0) {
        int numOfData = block.width();
//        int k = 0;
        REQUIRE_TRUE(numOfData % 2 == 0, 0, 
            "dynamic_stitch: The input params should contains"
            " both indeces and data lists with same length.");
        numOfData /= 2;

        auto output = OUTPUT_VARIABLE(0);
        std::vector<NDArray*> inputs(numOfData);
        std::vector<NDArray*> indices(numOfData);
        for (int e = 0; e < numOfData; e++) {
            auto data = INPUT_VARIABLE(numOfData + e);
            auto index = INPUT_VARIABLE(e);
            inputs[e] = data;
            indices[e] = index;
        }

        return helpers::dynamicStitchFunctor(inputs, indices, output);
    }

    DECLARE_SHAPE_FN(dynamic_stitch) {
        int maxValue = 0;
        auto numOfData = block.width();
        numOfData /= 2; // only index part it's needed to review
        auto restShape = inputShape->at(numOfData);
        auto firstShape = inputShape->at(0);
        for(int i = 0; i < numOfData; i++) {
            auto input = INPUT_VARIABLE(i);

            // FIXME: we have reduce::Max, cinsider using it instead
            /*
            for (int e = 0; e < input->lengthOf(); ++e) {
                if (T(maxValue) < (*input)(e))
                    maxValue = static_cast<int>((*input)(e));
            }
             */
            throw std::runtime_error("Not implemented yet");
        }

        Nd4jLong *outShapeInfo;
        int outRank = shape::rank(restShape) - shape::rank(firstShape) + 1;
        ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);

        outShapeInfo[0] = outRank;
        outShapeInfo[1] = maxValue + 1;
        for(int i = 1; i < outRank; ++i)
            outShapeInfo[i + 1] = shape::sizeAt(restShape, i);

        shape::updateStrides(outShapeInfo, shape::order(firstShape));

        //shape::shapeVector(maxValue + 1, newShape);

        return SHAPELIST(outShapeInfo);
    }
}
}

#endif