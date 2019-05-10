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
#if NOT_EXCLUDED(OP_cbow)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/sg_cb.h>

namespace nd4j {
    namespace ops {
        CONFIGURABLE_OP_IMPL(cbow, 15, 15, true, 0, 0) {
            auto target = INPUT_VARIABLE(0);
            auto ngStarter = INPUT_VARIABLE(1);

            // required part
            auto context = INPUT_VARIABLE(2);
            auto indices = INPUT_VARIABLE(3);
            auto codes = INPUT_VARIABLE(4);

            auto syn0 = INPUT_VARIABLE(5);
            auto syn1 = INPUT_VARIABLE(6);
            auto syn1neg = INPUT_VARIABLE(7);

            auto expTable = INPUT_VARIABLE(8);
            auto negTable = INPUT_VARIABLE(9);

            auto alpha = INPUT_VARIABLE(10);
            auto randomValue = INPUT_VARIABLE(11);
            auto numLabels = INPUT_VARIABLE(12);

            auto lockedWords = INPUT_VARIABLE(13);

            auto inferenceVector = INPUT_VARIABLE(14);

            auto numWorkers = block.numI() > 0 ? INT_ARG(0) : omp_get_max_threads();
            auto nsRounds = block.numI() > 1 ? INT_ARG(1) : 0;

            auto trainWords = block.numB() > 0 ? B_ARG(0) : true;
            auto isInference = block.numB() > 1 ? B_ARG(1) : false;

            REQUIRE_TRUE(block.isInplace(), 0, "CBOW: this operation requires inplace execution only");

            REQUIRE_TRUE(syn0->dataType() == syn1->dataType() && syn0->dataType() == syn1neg->dataType(), 0, "CBOW: all syn tables must have the same data type");
            REQUIRE_TRUE(syn0->dataType() == expTable->dataType(), 0, "CBOW: expTable must have the same data type as syn0 table");


            nd4j::ops::helpers::cbow(*syn0, *syn1, *syn1neg, *expTable, *negTable, *target, *ngStarter, nsRounds, *context, *lockedWords, *indices, *codes, *alpha, *randomValue, *numLabels, *inferenceVector, trainWords, numWorkers);


            return Status::OK();
        }

        DECLARE_TYPES(cbow) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, nd4j::DataType::INT32)
                    ->setAllowedInputTypes(1, nd4j::DataType::INT32)
                    ->setAllowedInputTypes(2, nd4j::DataType::INT32)
                    ->setAllowedInputTypes(3, nd4j::DataType::INT32)
                    ->setAllowedInputTypes(4, nd4j::DataType::INT8)
                    ->setAllowedInputTypes(5, {ALL_FLOATS})
                    ->setAllowedInputTypes(6, {ALL_FLOATS})
                    ->setAllowedInputTypes(7, {ALL_FLOATS})
                    ->setAllowedInputTypes(8, {ALL_FLOATS})
                    ->setAllowedInputTypes(9, {ALL_FLOATS})
                    ->setAllowedInputTypes(10, {ALL_FLOATS})
                    ->setAllowedInputTypes(11, nd4j::DataType::INT64)
                    ->setAllowedInputTypes(12, nd4j::DataType::INT32)
                    ->setAllowedInputTypes(13, nd4j::DataType::INT32)
                    ->setAllowedInputTypes(14, {ALL_FLOATS});
        }
    }
}

#endif