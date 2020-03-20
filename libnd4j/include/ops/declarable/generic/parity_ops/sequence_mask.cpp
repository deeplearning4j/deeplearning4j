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
// Created to use with batched tensor by GS <sgazeos@gmail.com> 3/27/2018
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/sequence_mask.h>

namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(sequence_mask, 1, 1, false, 0, 0) {
            auto input  = INPUT_VARIABLE(0);
            auto output = OUTPUT_NULLIFIED(0);
            const int inRank = input->rankOf();

            //REQUIRE_TRUE(inRank >= 1, 0, "sequence_mask: input array must have rank >= 1, but %i given!", inRank);
            Nd4jLong maxInd = input->argMax();
            float max = input->e<float>(maxInd);
            if (block.getIArguments()->size() > 0) {
                maxInd = INT_ARG(0);
                if (maxInd < max)
                    maxInd = static_cast<Nd4jLong>(max);
            }
            else if (block.width() > 1) {
                auto maxlen = INPUT_VARIABLE(1);
                //REQUIRE_TRUE(maxlen->lengthOf() == 1, "sequence_mask: 2nd input (max length) should be a scalar array.");
                float tmaxlen = maxlen->e<float>(0);
                if (tmaxlen > max)
                    maxInd = static_cast<Nd4jLong>(tmaxlen);
            }
            else
                maxInd = static_cast<Nd4jLong>(max);

            helpers::sequenceMask(block.launchContext(), input, output, maxInd);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(sequence_mask) {

            Nd4jLong* outShapeInfo = nullptr;
            auto in = inputShape->at(0);
            int outRank = shape::rank(in) + 1;
            auto input = INPUT_VARIABLE(0);
            auto dtype = DataType::BOOL;
            auto argMaxInd = input->argMax();
            Nd4jLong max = input->e<Nd4jLong>(argMaxInd);
            Nd4jLong maxInd = max;

            if (block.numD() > 0)
                dtype = D_ARG(0);

            if (block.width() > 1) {
                auto maxlen = INPUT_VARIABLE(1);
                Nd4jLong tmaxlen = maxlen->e<Nd4jLong>(0);
                if (tmaxlen > max)
                    maxInd = static_cast<Nd4jLong>(tmaxlen);
                if (block.numI() > 0) {
                    dtype = (DataType) INT_ARG(0);
                }
            }
            else {
                if (block.numI() > 0) {
                    maxInd = INT_ARG(0);
                }
                if (maxInd < max)
                    maxInd = max;
                if (block.numI() > 1)
                    dtype = (DataType)INT_ARG(1); // to work with legacy code
            }

            int lastDimension = maxInd;
            ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);
            outShapeInfo[0] = outRank;
            for(int i = 0; i < outRank - 1; ++i)
                outShapeInfo[i + 1] = shape::sizeAt(in, i);
            outShapeInfo[outRank] = lastDimension;

            ShapeUtils::updateStridesAndType(outShapeInfo, dtype, shape::order(in));

            return SHAPELIST(CONSTANT(outShapeInfo));
    }

        DECLARE_TYPES(sequence_mask) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_INTS})
                    ->setAllowedOutputTypes(sd::DataType::ANY);
        }
}
}

