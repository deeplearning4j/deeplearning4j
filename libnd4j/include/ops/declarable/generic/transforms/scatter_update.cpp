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
#if NOT_EXCLUDED(OP_scatter_update)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
    namespace ops {
        /**
         * scatter update operation
         *
         * IArgs map:
         * IArgs[0] - update operation: 0 - add; 1 - sub; 2 - mul; 3 - div; 4 - rsub; 5 - rdiv; 6 - assign
         * IArgs[1] - number of dimensions
         * IArgs[...] - dimensions
         * IArgs[...] - number of indices
         * IArgs[...] - indices
         *
         * @tparam T
         */
        CONFIGURABLE_OP_IMPL(scatter_update, 2, 1, true, 0, -1) {
            
            NDArray<T> *operand = INPUT_VARIABLE(0);
            NDArray<T> *updates = INPUT_VARIABLE(1);
            
            helpers::scatterUpdate(*operand, *updates, block.getIArguments());
            
            return Status::OK();
        }
        DECLARE_SYN(scatterupdate, scatter_update);
    }
}

#endif