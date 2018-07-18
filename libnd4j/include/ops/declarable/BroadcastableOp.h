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
// Created by raver on 6/6/2018.
//

#ifndef LIBND4J_BROADCASTABLEOP_H
#define LIBND4J_BROADCASTABLEOP_H

#include <graph/Context.h>
#include "OpDescriptor.h"
#include "DeclarableOp.h"
#include "DeclarableCustomOp.h"

namespace nd4j {
    namespace ops {
        template <typename T>
        class ND4J_EXPORT BroadcastableOp : public DeclarableCustomOp<T>{
        protected:
            virtual Nd4jStatus validateAndExecute(Context<T> &block) = 0;
        public:
            BroadcastableOp(const char *name, int numTArgs, int numIArgs);
            ~BroadcastableOp();

            ShapeList *calculateOutputShape(ShapeList *inputShape, nd4j::graph::Context<T> &block) override;
        };
    }
}


#endif //LIBND4J_BROADCASTABLEOP_H
