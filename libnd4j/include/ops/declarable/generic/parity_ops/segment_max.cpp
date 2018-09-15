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
// Created by george@skymind.io on 2/21/2018.
//

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/segment.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(segment_max, 2, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto idxSegments = INPUT_VARIABLE(1);
            auto segmentedOutput = OUTPUT_VARIABLE(0);
            REQUIRE_TRUE(idxSegments->isVector(), 0, "segment_max: segment indexes array should be a vector, but it rank is %i.", idxSegments->rankOf());
            REQUIRE_TRUE(idxSegments->lengthOf() == input->sizeAt(0), 0, "segment_max: segment indexes array length should be equal to the input first dimension, but %i != %i.", idxSegments->lengthOf(), input->sizeAt(0));

            // FIXME: float?
            double expected = 0.f, wrong = 0.f;

            REQUIRE_TRUE(helpers::segmentIndicesValidate(idxSegments, expected, wrong), 0, "segment_max: segment indices should be arranged, but %2.1f > %2.1f", expected, wrong);

            helpers::segmentMaxFunctor(input, idxSegments, segmentedOutput);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(segment_max) {
            auto idxVector = INPUT_VARIABLE(1);

            auto in = inputShape->at(0);
            int outRank = shape::rank(in);
            Nd4jLong* outputShape = nullptr;
            int val = (*idxVector).getScalar<int>(idxVector->lengthOf() - 1);

            int numOfClasses = val + 1;

            ALLOCATE(outputShape, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);

            outputShape[0] = outRank;
            outputShape[1] = numOfClasses;
            for(int i = 1; i < outRank; ++i)
                outputShape[i + 1] = shape::sizeAt(in, i);

            shape::updateStrides(outputShape, shape::order(in));

            return SHAPELIST(outputShape);
        }
    }

}
