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
// Created by raver119 on 06.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_create_list)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(create_list, 1, 2, 0, -2) {
            int height = 0;
            bool expandable = false;
            if (block.getIArguments()->size() == 1) {
                height = INT_ARG(0);
                expandable = (bool) INT_ARG(1);
            } else if (block.getIArguments()->size() == 2) {
                height = INT_ARG(0);
            } else {
                height = 0;
                expandable = true;
            }

            auto list = new NDArrayList(height, expandable);

            // we recieve input array for graph integrity purposes only
            auto input = INPUT_VARIABLE(0);

            OVERWRITE_RESULT(list);

            auto scalar = NDArray::scalar(list->counter());
            block.pushNDArrayToVariableSpace(block.getNodeId(), 1, scalar);

            return ND4J_STATUS_OK;
        }
        DECLARE_SYN(TensorArrayV3, create_list);
        DECLARE_SYN(tensorarrayv3, create_list);
        DECLARE_SYN(TensorArrayCreateV3, create_list);
        DECLARE_SYN(tensorarraycreatev3, create_list);
    }
}

#endif