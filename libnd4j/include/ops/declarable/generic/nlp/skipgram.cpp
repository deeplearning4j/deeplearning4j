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
//  @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_skipgram)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/sg_cb.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(skipgram, 3, 1, true, 0, 0) {
            // required part
            auto indices = INPUT_VARIABLE(0);
            auto codes = INPUT_VARIABLE(1);
            auto syn0 = INPUT_VARIABLE(2);

            auto syn1 = INPUT_VARIABLE(3);
            auto syn1neg = INPUT_VARIABLE(4);
            auto expTable = INPUT_VARIABLE(5);

            auto inferenceVector = INPUT_VARIABLE(6);

            auto isInference = block.numB() > 0 ? B_ARG(0) : true;

            REQUIRE_TRUE(syn0->dataType() == syn1->dataType() && syn0->dataType() == syn1neg->dataType(), 0, "SkipGram: all syn tables must have the same data type");

            nd4j::ops::helpers::skipgram(*syn0, *syn1, *syn1neg, *indices, *codes, *inferenceVector);

            return Status::OK();
        }

        DECLARE_TYPES(skipgram) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, nd4j::DataType::INT32)
                    ->setAllowedInputTypes(1, nd4j::DataType::INT8)
                    ->setAllowedInputTypes(2, {ALL_FLOATS})
                    ->setAllowedInputTypes(3, {ALL_FLOATS})
                    ->setAllowedOutputTypes(nd4j::DataType::INT8);
        }

        DECLARE_SHAPE_FN(skipgram) {
            return SHAPELIST(ShapeBuilders::createScalarShapeInfo(DataType::INT8, block.getWorkspace()));
        }
    }
}

#endif