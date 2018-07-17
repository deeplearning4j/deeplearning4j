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


#include <ops/declarable/CustomOperations.h>
#include <op_boilerplate.h>

namespace nd4j {
    namespace ops {
        /**
         * This operation is, basically IF statement
         * 
         * arg_0 is our "signal"
         * arg_1 is condition that will determine transition
         */
        // TODO: make this op a placeholder too
        DIVERGENT_OP_IMPL(Switch, 2, 2, true) {
            auto input = INPUT_VARIABLE(0);
            auto condition = INPUT_VARIABLE(1);

            // we'll store signal to both ends
            //STORE_2_RESULTS(*input, *input);

            // but we'll ensure only one node is active, and other is disabled
            if (condition->getScalar(0) == (T) 0.0f) {
                block.setBranch(0);
                this->storeResult(block, 0, input->dup());
            } else {
                block.setBranch(1);
                this->storeResult(block, 1, *input->dup());
            }

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(switch, Switch);
        DECLARE_SYN(if, Switch);


        /**
         *  This op is a placeholder.
         *  Actual WHILE implementation is in GraphExecutioner
         */
        LOGIC_OP_IMPL(While);
        DECLARE_SYN(while, While);

        /**
         *  This op is a placeholder.
         *  Actual Scope implementation is in Graph and GraphExecutioner
         */
        LOGIC_OP_IMPL(Scope);
        DECLARE_SYN(scope, Scope);

        /**
         *  This op is a placeholder.
         *  Actual Conditional implementation is in Graph and GraphExecutioner
         */
        LOGIC_OP_IMPL(Conditional);
        DECLARE_SYN(cond, Conditional);


        /**
         * This op is a placeholder
         * Actual implementation is in LogicReturn class
         */
        LOGIC_OP_IMPL(Return);
        DECLARE_SYN(return, Return);
    }
}