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
// Created by george@skymind.io on 2/21/2018.
// Modified by sgazeos@gmail.com on 4/4/2018

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_sufficient_statistics)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/axis.h>
namespace sd {
    namespace ops {
        CUSTOM_OP_IMPL(sufficient_statistics, 2, 3, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto axisVector = INPUT_VARIABLE(1);
            auto dataCount = OUTPUT_VARIABLE(0);

            auto sum = OUTPUT_VARIABLE(1);
            auto squares = OUTPUT_VARIABLE(2);

            std::vector<int> axis(axisVector->lengthOf());//*block.getIArguments();

            // axis might be dynamic (i.e. tf mode)
            helpers::adjustAxis(input->rankOf(), axisVector, axis);

            input->reduceAlongDimension(reduce::SquaredNorm, *squares, axis);
            input->reduceAlongDimension(reduce::Sum, *sum, axis);
            auto count = NDArrayFactory::create(input->dataType(), input->lengthOf() / sum->lengthOf());
            dataCount->assign(count);
            if (block.numT() > 0) {
                auto shift = OUTPUT_VARIABLE(3);
                shift->assign(T_ARG(0));
            }

            return Status::OK();
        }

        DECLARE_TYPES(sufficient_statistics) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS});
            getOpDescriptor()
                    ->setAllowedInputTypes(1, {DataType::INT32, DataType::INT64});
            getOpDescriptor()
                    ->setAllowedOutputTypes(0, DataType::INHERIT);
            getOpDescriptor()
                    ->setAllowedOutputTypes(1, DataType::INHERIT);
            getOpDescriptor()
                    ->setAllowedOutputTypes(2, DataType::INHERIT);
        }

        DECLARE_SHAPE_FN(sufficient_statistics) {
            auto axisVector = INPUT_VARIABLE(1);
            std::vector<int> axis(axisVector->lengthOf());

            auto input = INPUT_VARIABLE(0);
            helpers::adjustAxis(input->rankOf(), axisVector, axis);

            //std::vector<int> dims = ShapeUtils::evalDimsToExclude(input->rankOf(), {axis});
            auto scalarShape = ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(inputShape->at(0)));
            auto sumShape = ShapeUtils::evalReduceShapeInfo('c', axis, *input, false, false, block.workspace());

            auto squareShape = ShapeUtils::evalReduceShapeInfo('c', axis, *input, false, false, block.workspace());

            auto shapeList = SHAPELIST(scalarShape, sumShape, squareShape);
            if (block.numT() > 0)
                shapeList->push_back(ConstantShapeHelper::getInstance().scalarShapeInfo(ArrayOptions::dataType(inputShape->at(0))));

            return shapeList;
        }
    }

}

#endif