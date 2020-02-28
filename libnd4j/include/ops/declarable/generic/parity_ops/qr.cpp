/*******************************************************************************
 * Copyright (c) 2019-2020 Konduit K.K.
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
// Created by GS <sgazeos@gmail.com> at 12/20/2019
//

#include <op_boilerplate.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/qr.h>

#if NOT_EXCLUDED(OP_qr)
namespace nd4j {
    namespace ops {
        CUSTOM_OP_IMPL(qr, 1, 2, false, 0, 0) {
            auto input = INPUT_VARIABLE(0);
            auto outputQ = OUTPUT_VARIABLE(0);
            auto outputR = OUTPUT_VARIABLE(1);
            auto fullMatricies = false;
            if (block.getBArguments()->size())
                fullMatricies = B_ARG(0);
            REQUIRE_TRUE(input->rankOf() >=2, 0, "qr: The rank of input array should not be less than 2, but %i is given", input->rankOf());
            REQUIRE_TRUE((fullMatricies && outputQ->sizeAt(-1) == input->sizeAt(-2)) || (!fullMatricies && outputQ->isSameShape(input)), 0, "qr: The last dimmensions should be equal to result Q, but %i and %i are given", outputQ->sizeAt(-1), input->sizeAt(-2));
            REQUIRE_TRUE((fullMatricies && outputR->sizeAt(-1) == input->sizeAt(-1)) || (!fullMatricies && outputR->sizeAt(-1) == outputR->sizeAt(-2)), 0, "qr: The last dimmensions should be equal to result R, but %i and %i are given", outputR->sizeAt(-1), input->sizeAt(-1));
            if (!input->isEmpty() && !outputQ->isEmpty() && !outputR->isEmpty())
                helpers::qr(block.launchContext(), input, outputQ, outputR, fullMatricies);

            return Status::OK();
        }

        DECLARE_SHAPE_FN(qr) {
            auto inShape = inputShape->at(0);

            Nd4jLong* shapeQ;
            Nd4jLong* shapeR;
            int targetRank = shape::rank(inShape); // last two dimensions will be reduced to scalar

            auto fullMatricies = false;
            if (block.getBArguments()->size())
                fullMatricies = B_ARG(0);

            auto shape = ShapeUtils::shapeAsVector(inShape);

            if (!fullMatricies) { // outputs are: Q is MxN and R is NxN
                shape[targetRank - 1] = shape::sizeAt(inShape, -1);
                shape[targetRank - 2] = shape[targetRank - 1];
                shapeQ = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape),
                                                                             shape::order(inShape), targetRank,
                                                                             shape::shapeOf(inShape));
                shapeR = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape),
                                                                             shape::order(inShape), shape);

            }
            else {// otherwise outputs are Q is MxM and R is MxN with zero filled rows
                shape[targetRank - 1] = shape::sizeAt(inShape, -2);
                shape[targetRank - 2] = shape[targetRank - 1];
                shapeR = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape),
                                                                             shape::order(inShape), targetRank,
                                                                             shape::shapeOf(inShape));
                shapeQ = ConstantShapeHelper::getInstance()->createShapeInfo(ArrayOptions::dataType(inShape),
                                                                             shape::order(inShape), shape);
            }

            return SHAPELIST(shapeQ, shapeR);

        }

        DECLARE_TYPES(qr) {
            getOpDescriptor()
                    ->setAllowedInputTypes({ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }
    }
}

#endif
