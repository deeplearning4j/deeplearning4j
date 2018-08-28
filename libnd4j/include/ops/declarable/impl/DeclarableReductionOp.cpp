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
        template <typename T>
        DeclarableReductionOp<T>::DeclarableReductionOp(int numInputs, int numOutputs, const char *opName, bool allowsInplace, int tArgs, int iArgs) : nd4j::ops::DeclarableOp<T>(numInputs, numOutputs, opName, allowsInplace, tArgs, iArgs) {
            //
        }

        template <typename T>
        DeclarableReductionOp<T>::~DeclarableReductionOp()  {
            //
        }


        template <typename T>
        nd4j::ShapeList* DeclarableReductionOp<T>::calculateOutputShape(nd4j::ShapeList* inputShape, nd4j::graph::Context<T>& block)  {
            
            Nd4jLong* inShapeInfo = inputShape->at(0);

            const int minNumTArgs     = this->getOpDescriptor()->getNumberOfTArgs();
            const int currentNumTArgs = block.getTArguments()->size();
            
            const bool keepDims = currentNumTArgs > minNumTArgs ? static_cast<bool>(T_ARG(currentNumTArgs - 1)) : false;

            Nd4jLong* outShapeInfo = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inShapeInfo), *block.getIArguments(), inShapeInfo, keepDims, false, block.workspace());            

            return SHAPELIST(outShapeInfo);
        }

        template class ND4J_EXPORT DeclarableReductionOp<float>;
        template class ND4J_EXPORT DeclarableReductionOp<float16>;
        template class ND4J_EXPORT DeclarableReductionOp<double>;
    }
}