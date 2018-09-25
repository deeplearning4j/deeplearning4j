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
// @author raver119@gmail.com
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_write_list)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(write_list, 2, 1, 0, -2) {
            auto list = INPUT_LIST(0);

            // nd4j mode
            if (block.getIArguments()->size() == 1) {
                auto input = INPUT_VARIABLE(1);
                auto idx = INT_ARG(0);

                Nd4jStatus result = list->write(idx, input->dup());

                auto res = NDArrayFactory::scalar_(list->counter(), block.workspace());
                OVERWRITE_RESULT(res);

                return result;
            } else if (block.width() >= 3) {
                auto input = INPUT_VARIABLE(block.width() - 2);
                auto idx = INPUT_VARIABLE(block.width() - 1);

                REQUIRE_TRUE(idx->isScalar(), 0, "Index should be Scalar");

                Nd4jStatus result = list->write(idx->e<int>(0), input->dup());

                auto res = NDArrayFactory::scalar_(list->counter(), block.workspace());
                OVERWRITE_RESULT(res);

                return result;
            } else return ND4J_STATUS_BAD_INPUT;
        }
        DECLARE_SYN(TensorArrayWriteV3, write_list);
        DECLARE_SYN(tensorarraywritev3, write_list);
    }
}

#endif