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
// Created by raver119 on 13.10.2017.
//

#ifndef LIBND4J_BOOLEANOP_H
#define LIBND4J_BOOLEANOP_H

#include <graph/Context.h>
#include "OpDescriptor.h"
#include "DeclarableOp.h"

namespace sd {
    namespace ops {
        class ND4J_EXPORT BooleanOp : public DeclarableOp {
        protected:
            OpDescriptor * _descriptor;

            bool prepareOutputs(Context& block);
            Nd4jStatus validateAndExecute(Context& block) override = 0;
        public:
            BooleanOp(const char *name, int numInputs, bool scalar);

            bool verify(const std::vector<sd::NDArray*>& args);
            bool verify(sd::graph::Context& block);

            Nd4jStatus execute(Context* block) override;

            ShapeList *calculateOutputShape(ShapeList *inputShape, sd::graph::Context& block) override;
        };
    }
}



#endif //LIBND4J_BOOLEANOP_H