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
// @author Yurii Shyrma, created on 16.04.2018
//

#include <ops/declarable/helpers/reverse.h>
#include <helpers/ShapeUtils.h>
#include <array/ResultSet.h>


namespace nd4j    {
namespace ops     {
namespace helpers {

    /////////////////////////////////////////////////////////////////////////////////////
    // this legacy op is written by raver119@gmail.com
    template<typename T>
    void reverseArray(void *vinArr, Nd4jLong *inShapeBuffer, void *voutArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse) {

    }


    ///////////////////////////////////////////////////////////////////
    template <typename T>
    static void _reverseSequence(const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim){

    }

    void reverseSequence(const NDArray* input, const NDArray* seqLengths, NDArray* output, int seqDim, const int batchDim) {
        BUILD_SINGLE_SELECTOR(input->dataType(), _reverseSequence, (input, seqLengths, output, seqDim, batchDim), LIBND4J_TYPES);
    }

    //////////////////////////////////////////////////////////////////////////
    void reverse(const NDArray* input, NDArray* output, const std::vector<int>* intArgs, bool isLegacy) {

    }

BUILD_SINGLE_TEMPLATE(template void reverseArray, (void *inArr, Nd4jLong *inShapeBuffer, void *outArr, Nd4jLong *outShapeBuffer, int numOfElemsToReverse), LIBND4J_TYPES);

}
}
}

