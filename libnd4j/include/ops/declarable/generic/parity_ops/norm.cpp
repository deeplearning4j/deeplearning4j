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
#if NOT_EXCLUDED(OP_norm)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>

namespace nd4j {
    namespace ops {
        REDUCTION_OP_IMPL(norm, 1, 1, false, 1, -2) {
            auto input = INPUT_VARIABLE(0);
            NDArray *output = nullptr;

            auto mode = (int) T_ARG(0);
            std::vector<int> dims = *block.getIArguments();
            bool overwrite = false;

            if (block.width() == 1) {
                output = OUTPUT_VARIABLE(0);
            } else {
                auto axisVector = INPUT_VARIABLE(1);
                dims.resize(axisVector->lengthOf());
                helpers::adjustAxis(input, axisVector, dims);

                auto shape = ShapeUtils::evalReduceShapeInfo(input->ordering(), dims, *input, false, true);
                ArrayOptions::setDataType(shape, input->dataType());
                output = new NDArray(shape, false, block.getWorkspace());

                overwrite = true;
                RELEASE(shape, input->getWorkspace());
            }

            switch(mode) {
                case 0: {
                    REQUIRE_TRUE(dims.size() == 2 || (input->rankOf() == 2 && dims.size() == 0), 0, "Norm: Frobenius is defined for 2D matrices or TADS only");
                    // fro
                    input->reduceAlongDimension(reduce::NormFrobenius, output, dims, false, true);
                }
                break;
                case 1: {
                    // euclidean
                    if ((input->rankOf() == 2 && dims.size() == 0) || dims.size() == 2) {
                        input->reduceAlongDimension(reduce::NormFrobenius, output, dims, false, true);
                    } else {
                        input->reduceAlongDimension(reduce::Norm2, output, dims, false, true);
                    }
                }
                break;
                case 2: {
                    // 1
                    input->reduceAlongDimension(reduce::Norm1, output, dims, false, true);
                }
                break;
                case 3: {
                    // 2 
                    input->reduceAlongDimension(reduce::Norm2, output, dims, false, true);
                }
                break;
                case 4: {
                    // inf-norm
                    input->reduceAlongDimension(reduce::NormMax, output, dims, false, true);
                }
                break;
                default: {
                    // p-norm
                    REQUIRE_TRUE(block.getIArguments()->size() > 1, 0, "P-Norm reductions requires 2 TArguments, but only 1 was provided");
                    // FIXME: p is required here
                    //T p = T_ARG(1);
                    input->reduceAlongDimension(reduce::NormP, output, dims, false, true, nullptr);
                }
            }

            if (overwrite) {
                OVERWRITE_RESULT(output);
            }

            return ND4J_STATUS_OK;
        };
    }
}

#endif