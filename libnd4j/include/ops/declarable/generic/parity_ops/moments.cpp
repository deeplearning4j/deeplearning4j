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
// Created by george@skymind.io on 26.01.2018.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_moments)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>

namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(moments, 1, 2, false, 0, -2) {
            auto input = INPUT_VARIABLE(0);
            auto means = OUTPUT_VARIABLE(0);
            auto variances = OUTPUT_VARIABLE(1);

            std::vector<int> axis = *block.getIArguments();
            const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;

            // axis might be dynamic (i.e. tf mode)
            if (block.width() > 1 && axis.size() == 0) {
                auto axisVector = INPUT_VARIABLE(1);
                axis.resize(axisVector->lengthOf());
                helpers::adjustAxis(input, axisVector, axis);
//                for (int e = 0; e < axisVector->lengthOf(); e++) {
//                    int ca = (int) axisVector->e(e);
//                    if (ca < 0)
//                        ca += input->rankOf();
//
//                    axis.emplace_back(ca);
//                }

            }

            std::vector<int>& dims = axis;
            input->varianceAlongDimension(variance::SummaryStatsVariance, variances, false, axis);
            input->reduceAlongDimension(reduce::Mean, means, axis, keepDims);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(moments) {
            auto axis = *block.getIArguments();
            auto input = INPUT_VARIABLE(0);

            // axis might be dynamic (i.e. tf mode)
            if (block.width() > 1 && axis.size() == 0) {
                auto axisVector = INPUT_VARIABLE(1);

                for (int e = 0; e < axisVector->lengthOf(); e++) {
                    int ca = axisVector->e<int>(e);
                    if (ca < 0)
                        ca += input->rankOf();

                    axis.emplace_back(ca);
                }

            }
            //std::vector<int> dims = ShapeUtils::convertAxisToTadTarget(input->rankOf(), {axis});
            const bool keepDims = block.getTArguments()->size() > 0 ? (bool)T_ARG(0) : false;
    
            auto meanShape = ShapeUtils::evalReduceShapeInfo('c', axis, *input, keepDims, false, block.workspace());
            auto varianceShape = ShapeUtils::evalReduceShapeInfo('c', axis, *input, keepDims, false, block.workspace());
            return SHAPELIST(meanShape, varianceShape); 
        }
    }

}

#endif