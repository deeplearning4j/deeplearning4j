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

#ifndef LIBND4J_CUSTOMOPERATIONS_H
#define LIBND4J_CUSTOMOPERATIONS_H

#include <ops/declarable/headers/activations.h>
#include <ops/declarable/headers/boolean.h>
#include <ops/declarable/headers/broadcastable.h>
#include <ops/declarable/headers/convo.h>
#include <ops/declarable/headers/list.h>
#include <ops/declarable/headers/recurrent.h>
#include <ops/declarable/headers/transforms.h>
#include <ops/declarable/headers/parity_ops.h>
#include <ops/declarable/headers/shape.h>
#include <ops/declarable/headers/random.h>
#include <ops/declarable/headers/nn.h>
#include <ops/declarable/headers/blas.h>
#include <ops/declarable/headers/bitwise.h>
#include <ops/declarable/headers/loss.h>
#include <ops/declarable/headers/datatypes.h>
#include <ops/declarable/headers/third_party.h>
#include <ops/declarable/headers/tests.h>
#include <dll.h>
#include <Status.h>
#include <helpers/ArrayUtils.h>


namespace nd4j {
    struct ND4J_EXPORT _loader {
        _loader();
    };

    namespace ops {

        // logic ops 
        DECLARE_DIVERGENT_OP(Switch, 2, 2, true);
        DECLARE_LOGIC_OP(While);
        DECLARE_LOGIC_OP(Scope);
        DECLARE_LOGIC_OP(Conditional);
        DECLARE_LOGIC_OP(Return);


        /**
         * This operations exposes given arguments as it's own outputs, but does it only once.
         * Subsequent calls will be served directly by this op.
         *
         * PLEASE NOTE: This operation is internal graph operation, and shouldn't be used directly usually.
         */
        DECLARE_CUSTOM_OP(expose, -1, -1, true, 0, 0);
    }
}

#endif //LIBND4J_CUSTOMOPERATIONS_H
