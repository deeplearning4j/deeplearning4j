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
#if NOT_EXCLUDED(OP_scatter_list)

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(scatter_list, 1, 1, 0, -2) {
            NDArrayList *list = nullptr;
            NDArray *array = nullptr;
            NDArray *indices = nullptr;

            bool hasList = false;
            auto w = block.width();

            if (w == 4){
                list = INPUT_LIST(0);
                indices = INPUT_VARIABLE(1);
                array = INPUT_VARIABLE(2);
                hasList = true;
            } else {
                array = INPUT_VARIABLE(1);
                indices = INPUT_VARIABLE(2);
                list = new NDArrayList(indices->lengthOf(), false);
                block.trackList(list);
            }

            REQUIRE_TRUE(indices->isVector(), 0, "ScatterList: Indices for Scatter should be a vector")
            REQUIRE_TRUE(indices->lengthOf() == array->sizeAt(0), 0, "ScatterList: Indices length should be equal number of TADs along dim0, but got %i instead", indices->lengthOf());

            std::vector<int> axis = ShapeUtils::convertAxisToTadTarget(array->rankOf(), {0});
            auto tads = array->allTensorsAlongDimension( axis);
            for (int e = 0; e < tads->size(); e++) {
                auto idx = indices->e<int>(e);
                if (idx >= tads->size())
                    return ND4J_STATUS_BAD_ARGUMENTS;

                auto arr = tads->at(e)->dup(array->ordering());
                auto res = list->write(idx, arr);
                if (res != ND4J_STATUS_OK)
                    return res;
            }

            if (!hasList)
                OVERWRITE_RESULT(list);

            delete tads;

            return Status::OK();
        }
        DECLARE_SYN(TensorArrayScatterV3, scatter_list);
        DECLARE_SYN(tensorarrayscatterv3, scatter_list);
    }
}

#endif