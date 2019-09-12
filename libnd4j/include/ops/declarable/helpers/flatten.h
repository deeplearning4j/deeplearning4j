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
//  @author raver119@gmail.com
//

#ifndef DEV_TESTS_FLATTEN_H
#define DEV_TESTS_FLATTEN_H

#include <vector>
#include <NDArray.h>

namespace nd4j    {
namespace ops     {
namespace helpers {


//////////////////////////////////////////////////////////////////////
void flatten(nd4j::LaunchContext *context, std::vector<NDArray*> &inputs, NDArray *output, char order);


//////////////////////////////////////////////////////////////////////
INLINEDEF _CUDA_HD Nd4jLong getIndexOffsetOrdered(Nd4jLong index, const Nd4jLong *shapeInfo, const char order) {

    Nd4jLong offset = 0;

    if (order == 'c') {

        for(uint i = shapeInfo[0]; i > 1; --i) {
            offset += (index % shapeInfo[i]) * shapeInfo[i + shapeInfo[0]];
            index /= shapeInfo[i];
        }

        offset += index * shapeInfo[1 + shapeInfo[0]];  // last iteration
    }
    else {

        for(uint i = 1; i < shapeInfo[0]; ++i) {
            offset += (index % shapeInfo[i]) * shapeInfo[i + shapeInfo[0]];
            index /= shapeInfo[i];
        }

        offset += index * shapeInfo[2 * shapeInfo[0]];  // last iteration
    }

    return offset;
}


}
}
}

#endif //DEV_TESTS_FLATTEN_H
