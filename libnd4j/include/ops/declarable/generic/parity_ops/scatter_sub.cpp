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
// Created by raver119 on 24.11.17.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_scatter_sub)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/generic/helpers/ScatterHelper.h>

namespace nd4j {
    namespace ops {
        OP_IMPL(scatter_sub, 3, 1, true) {
            auto input = INPUT_VARIABLE(0);
            auto indices = INPUT_VARIABLE(1);
            auto updates = INPUT_VARIABLE(2);

            auto output = OUTPUT_VARIABLE(0);

            if (!block.isInplace())
                output->assign(input);

            ScatterHelper<T>::template scatter_apply<simdOps::Subtract<T>>(output, indices, updates);        

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(ScatterSub, scatter_sub);
    }
}

#endif