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
//  @author sgazeos@gmail.com
//  @brief helpers fuctions for segment_* ops (segment_max, segment_min, etc.)
//  @brief helpers fuctions for unsorted_segment_* ops (unsorted_segment_max, etc.)
//
#ifndef __SEGMENT_HELPERS__
#define __SEGMENT_HELPERS__
#include <op_boilerplate.h>
#include <NDArray.h>

namespace nd4j {
namespace ops {
namespace helpers {

    template <typename T>
    bool segmentIndicesValidate(NDArray<T>* indices, Nd4jLong& expected, Nd4jLong& output);
    template <typename T>
    bool unsortedSegmentIndicesValidate(NDArray<T>* indices, Nd4jLong numOfClasses, Nd4jLong& output);

    template <typename T>
    void segmentMaxFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

    template <typename T>
    void segmentMinFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

    template <typename T>
    void segmentMeanFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

    template <typename T>
    void segmentSumFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

    template <typename T>
    void segmentProdFunctor(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* output);

    template <typename T>
    void unsortedSegmentSqrtNFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    void unsortedSegmentMaxFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    void unsortedSegmentMinFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    void unsortedSegmentMeanFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    void unsortedSegmentSumFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    void unsortedSegmentProdFunctor(NDArray<T>* input, NDArray<T>* indices, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    int segmentMaxFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output);

    template <typename T>
    int segmentMinFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output);

    template <typename T>
    int segmentMeanFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output);

    template <typename T>
    int segmentSumFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output);

    template <typename T>
    int segmentProdFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, NDArray<T>* output);

    template <typename T>
    int unsortedSegmentSqrtNFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    int unsortedSegmentMaxFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    int unsortedSegmentMinFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    int unsortedSegmentMeanFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    int unsortedSegmentSumFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output);

    template <typename T>
    int unsortedSegmentProdFunctorBP(NDArray<T>* input, NDArray<T>* indices, NDArray<T>* gradOut, Nd4jLong numOfClasses, NDArray<T>* output);

}
}
}
#endif
