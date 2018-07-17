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
// Created by raver119 on 12.10.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_zeros_as)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(zeros_as, 1, 1, false) {
            // auto input = INPUT_VARIABLE(0);

            auto out = OUTPUT_VARIABLE(0);

            *out = static_cast<T>(0.f);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(zeroslike, zeros_as);
        DECLARE_SYN(zeros_like, zeros_as);
    }
}

#endif