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
// Created by raver119 on 07.10.2017.
//

#include <ops/declarable/DeclarableReductionOp.h>
#include <ops/declarable/DeclarableOp.h>
#include <helpers/TAD.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        DeclarableReductionOp::DeclarableReductionOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs) : nd4j::ops::DeclarableOp(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs) {
            //
        }

        DeclarableReductionOp::~DeclarableReductionOp()  {
            //
        }


        nd4j::ShapeList* DeclarableReductionOp::calculateOutputShape(nd4j::ShapeList* inputShape, nd4j::graph::Context& block)  {
           // int numDims = INT_ARG(0);
            std::vector<int> dims;
            for (int e = 0; e < block.getIArguments()->size(); e++)
                dims.push_back(INT_ARG(e));

            if (dims.size() > 1)
                std::sort(dims.begin(), dims.end());

            // special case - output is scalar
            if (dims.size() == 0 || (dims.size() == 1 && dims.at(0) == MAX_INT)) {
                Nd4jLong* newShape;
                ALLOCATE(newShape, block.getWorkspace(), 8, Nd4jLong);

                newShape[0] = 2;
                newShape[1] = 1;
                newShape[2] = 1;
                newShape[3] = 1;
                newShape[4] = 1;
                newShape[5] = 0;
                newShape[6] = 1;
                newShape[7] = 99;
                ArrayOptions::setDataType(newShape, ArrayOptions::dataType(inputShape->at(0)));

                return SHAPELIST(newShape);
            }

            shape::TAD tad(inputShape->at(0), dims.data(), dims.size());
            tad.createTadOnlyShapeInfo();

            Nd4jLong tadLength = shape::tadLength(inputShape->at(0), dims.data(), dims.size());
            Nd4jLong numTads = shape::length(inputShape->at(0)) /  tadLength;

            auto newShape = ShapeUtils::evalReduceShapeInfo('c', dims, inputShape->at(0), false, true, block.getWorkspace());

            return SHAPELIST(newShape);
        }
    }
}