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
#if NOT_EXCLUDED(OP_mergemaxindex)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops  {

OP_IMPL(mergemaxindex, -1, 1, false) {

    REQUIRE_OK(this->validateInputDimensionsMatch(block));
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    std::vector<NDArray<T>*> inArrs(block.width());
    
    for(int i = 0; i < block.width(); ++i)
        inArrs[i] = INPUT_VARIABLE(i);

    helpers::mergeMaxIndex(inArrs, *output);

    return Status::OK();
}

DECLARE_SYN(MergeMaxIndex, mergemaxindex);

}
}

#endif