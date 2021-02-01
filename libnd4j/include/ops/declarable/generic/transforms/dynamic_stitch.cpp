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
//  @author GS <sgazeos@gmail.com>
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_dynamic_stitch)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/dynamic.h>

namespace sd {
namespace ops {
    CUSTOM_OP_IMPL(dynamic_stitch, 2, 1, false, 0, 0) {
        int numOfData = block.width();
//        int k = 0;
        // checking input data size
        REQUIRE_TRUE(numOfData % 2 == 0, 0, 
            "dynamic_stitch: The input params should contains"
            " both indeces and data lists with same length.");
        // split input data list on two equal parts
        numOfData /= 2;

        // form input lists to use with helpers - both indices and float data inputs
        auto output = OUTPUT_VARIABLE(0);
        std::vector<NDArray*> inputs(numOfData);
        std::vector<NDArray*> indices(numOfData);

        for (int e = 0; e < numOfData; e++) {
            auto data = INPUT_VARIABLE(numOfData + e);
            auto index = INPUT_VARIABLE(e);

            inputs[e] = data;
            indices[e] = index;
        }
        // run helper
        return helpers::dynamicStitchFunctor(block.launchContext(), inputs, indices, output);
    }

    DECLARE_TYPES(dynamic_stitch) {
        getOpDescriptor()
                ->setAllowedInputTypes(sd::DataType::ANY)
                ->setAllowedOutputTypes({ALL_INTS, ALL_FLOATS});
    }

    DECLARE_SHAPE_FN(dynamic_stitch) {
        Nd4jLong maxValue = 0;
        auto numOfData = block.width();
        numOfData /= 2; // only index part it's needed to review
        auto restShape = inputShape->at(numOfData);
        auto firstShape = inputShape->at(0);
        // check up inputs to avoid non-int indices and calculate max value from indices to output shape length
        for(int i = 0; i < numOfData; i++) {
            auto input = INPUT_VARIABLE(i);
            REQUIRE_TRUE(input->isZ(), 0, "dynamic_stitch: Indices should be integer, but %d type given.", (int)input->dataType() );
            auto maxV = input->reduceNumber(reduce::Max);
            if (maxV.e<Nd4jLong>(0) > maxValue) maxValue = maxV.e<Nd4jLong>(0);
        }
        // calculate output rank - difference between indices shape and data shape
        int outRank = shape::rank(restShape) - shape::rank(firstShape) + 1; // at least 1D tensor
        std::vector<Nd4jLong> outShape(outRank);
        // fill up output shape template: the first to max index, and rests - to vals from the first data input
        outShape[0] = maxValue + 1;
        for(int i = 1; i < outRank; ++i)
            outShape[i] = shape::sizeAt(restShape, i);

        return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(ShapeDescriptor(ArrayOptions::dataType(restShape), shape::order(firstShape), outShape)));
    }
}
}

#endif