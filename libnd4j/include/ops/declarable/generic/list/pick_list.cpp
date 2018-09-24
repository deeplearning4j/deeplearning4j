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
#if NOT_EXCLUDED(OP_pick_list)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(pick_list, 1, 1, 0, -2) {
            auto list = INPUT_LIST(0);

            std::vector<int> indices;
            if (block.width() > 1 && block.getVariable(1)->getNDArray()->isVector()) {
                auto ia = INPUT_VARIABLE(1);
                for (int e = 0; e < ia->lengthOf(); e++)
                    indices.emplace_back(ia->e<int>(e));
            } else if (block.getIArguments()->size() > 0) {
                indices = *(block.getIArguments());
            } else return ND4J_STATUS_BAD_ARGUMENTS;

            for (auto& v: indices) {
                if (v >= list->height()) {
                    nd4j_printf("Requested index [%i] is higher (or equal) then ArrayList height: [%i]", v,
                                list->height());
                    return ND4J_STATUS_BAD_ARGUMENTS;
                }
            }
            auto result = list->pick(indices);

            OVERWRITE_RESULT(result);

            return Status::OK();
        }
    }
}

#endif