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
// Created by raver119 on 01.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_rank)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(rank, 1, 1, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto output = OUTPUT_VARIABLE(0);

            REQUIRE_TRUE(output->isScalar(), 0, "Rank output should be scalar");

            output->assign(static_cast<Nd4jLong>(input->rankOf()));

            return Status::OK();
        }
        DECLARE_SHAPE_FN(rank) {
            return SHAPELIST(ShapeBuilders::createScalarShapeInfo(nd4j::DataType::DataType_INT32, block.workspace()));
        }
    }
}

#endif