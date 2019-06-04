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
// @author George A. Shulinok <sgazeos@gmail.com), created on 5/15/2019.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_cell_contains)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace nd4j {
    namespace ops {

        CUSTOM_OP_IMPL(cell_contains, 3, 1, false, 0, 1) {
            auto corner = INPUT_VARIABLE(0);
            auto width = INPUT_VARIABLE(1);
            auto point = INPUT_VARIABLE(2);

            auto output = OUTPUT_VARIABLE(0);
            auto dimension = INT_ARG(0);
            output->assign(helpers::cell_contains(corner, width, point, dimension));
            return Status::OK();
        }

        DECLARE_TYPES(cell_contains) {
            getOpDescriptor()
                    ->setAllowedInputTypes(nd4j::DataType::ANY)
                    ->setAllowedOutputTypes(nd4j::DataType::BOOL)
                    ->setSameMode(false);
        }

        DECLARE_SHAPE_FN(cell_contains) {
            return SHAPELIST(CONSTANT(ShapeBuilders::createScalarShapeInfo(nd4j::DataType::BOOL, block.workspace())));
        }
    }
}

#endif