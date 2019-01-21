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

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(skipgram, 3, 1, true, 0, 0) {
            auto syn0 = INPUT_VARIABLE(0);
            auto syn1 = INPUT_VARIABLE(1);
            auto syn1neg = INPUT_VARIABLE(2);
            auto expTable = INPUT_VARIABLE(3);

            auto indices = INPUT_VARIABLE(4);
            auto codes = INPUT_VARIABLE(5);

            auto isInference = block.numB() > 0 ? B_ARG(0) : true;

            return Status::OK();
        }

        DECLARE_TYPES(skipgram) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

        DECLARE_SHAPE_FN(skipgram) {
            return SHAPELIST(ShapeBuilders::createScalarShapeInfo(DataType::INT8, block.getWorkspace()));
        }
    }
}

#endif