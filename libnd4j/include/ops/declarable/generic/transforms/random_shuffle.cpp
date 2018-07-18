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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 26.01.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_random_shuffle)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops  {

OP_IMPL(random_shuffle, 1, 1, true) {
    
    NDArray<T>* input  = INPUT_VARIABLE(0);
    const bool isInplace = block.isInplace();
    NDArray<T>* output = isInplace ? nullptr : OUTPUT_VARIABLE(0);

    nd4j::random::RandomBuffer* rng = block.getRNG();   
    
    REQUIRE_TRUE(rng != nullptr, 0, "RANDOM_SHUFFLE op: RNG should be defined in Graph !");

    helpers::randomShuffle(*input, *output, *rng, isInplace);
    
    return Status::OK();
}


}
}

#endif