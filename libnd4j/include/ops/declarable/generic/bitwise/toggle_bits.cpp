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
// Created by raver119 on 23.11.17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_toggle_bits)

#include <ops/declarable/CustomOperations.h>
#include <helpers/BitwiseUtils.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(toggle_bits, -1, -1, true) {

            for (int i = 0; i < block.width(); i++) {
                auto x = INPUT_VARIABLE(i);
                auto z = OUTPUT_VARIABLE(i);

             //   auto lambda = LAMBDA_T(_x) {
             //       return BitwiseUtils::flip_bits<T>(_x);
             //   };
                
             //   x->applyLambda(lambda, z);
                return ND4J_STATUS_OK;
            }
        }
    }
}

#endif