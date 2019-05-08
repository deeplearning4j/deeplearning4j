/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
#if NOT_EXCLUDED(OP_gather_list)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {
        LIST_OP_IMPL(gather_list, 2, 1, 0, -2) {
            auto list = INPUT_LIST(0);
            auto indices = INPUT_VARIABLE(1);

            indices->printShapeInfo("indices shape");
            indices->printIndexedBuffer("indices");

            REQUIRE_TRUE(indices->isVector() || indices->rankOf() == 1, 0, "Indices for Gather operation should be a vector");
            REQUIRE_TRUE(list->height() > 0, 0, "Number of elements in list should be positive prior to Gather call");
            REQUIRE_TRUE(list->height() == indices->lengthOf(), 1, "Number of indicies should be equal to number of elements in list, but got [%i] indices instead", indices->lengthOf());

            // first of all we need to get shapes
            std::vector<Nd4jLong> shape({0});
            shape[0] = indices->lengthOf();
            for (int e = 0; e < list->height(); e++) {
                auto array = list->readRaw(e);

                // now we should fill other dimensions 
                if (e == 0) {
                    for (int d = 0; d < array->rankOf(); d++)
                        shape.emplace_back(array->sizeAt(d));
                }
            }

            auto result = NDArrayFactory::create_('c', shape, list->dataType());
            std::vector<Nd4jLong> indicesList((list->readRaw(0)->rankOf() + 1) * 2, 0);
            int skipPosition = 0;
            for (int e = 0; e < indices->lengthOf(); e++) {
                auto idx = indices->e<int>(e);
                auto array = list->readRaw(idx);
                                
                // first dimension
                indicesList[0] = skipPosition;
                indicesList[1] = skipPosition++ + 1;                

                auto subarray = (*result)(indicesList, true);
                subarray.assign(array);
            }

            //OVERWRITE_RESULT(result);
            setupResult(result, block);
            return Status::OK();
        }
        DECLARE_SYN(TensorArrayGatherV3, gather_list);
        DECLARE_SYN(tensorarraygatherv3, gather_list);
    }
}

#endif