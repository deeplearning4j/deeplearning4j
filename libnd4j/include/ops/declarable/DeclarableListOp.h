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

#ifndef LIBND4J_DECLARABLE_LIST_OP_H
#define LIBND4J_DECLARABLE_LIST_OP_H

#include <array/ResultSet.h>
#include <graph/Context.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/DeclarableOp.h>

using namespace nd4j::graph;

namespace nd4j {
    namespace ops {
        class ND4J_EXPORT DeclarableListOp : public nd4j::ops::DeclarableOp {
        protected:
            virtual Nd4jStatus validateAndExecute(Context& block) = 0;

            nd4j::NDArray* getZ(Context& block, int inputId);
        public:
            DeclarableListOp(int numInputs, int numOutputs, const char* opName, int tArgs, int iArgs);
            ~DeclarableListOp();

            
            Nd4jStatus execute(Context* block) override;
            

            ResultSet* execute(NDArrayList* list, std::initializer_list<NDArray*> inputs, std::initializer_list<double> tArgs, std::initializer_list<int> iArgs);
            ResultSet* execute(NDArrayList* list, std::vector<NDArray*>& inputs, std::vector<double>& tArgs, std::vector<int>& iArgs);

            ShapeList* calculateOutputShape(ShapeList* inputShape, nd4j::graph::Context& block) override;
        };
    }
}

#endif