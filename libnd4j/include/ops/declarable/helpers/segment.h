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

    bool segmentIndicesValidate(NDArray* indices, NDArray& expected, NDArray& output);

    bool unsortedSegmentIndicesValidate(NDArray* indices, Nd4jLong numOfClasses, Nd4jLong& output);

    void segmentMaxFunctor(NDArray* input, NDArray* indices, NDArray* output);

    void segmentMinFunctor(NDArray* input, NDArray* indices, NDArray* output);

    void segmentMeanFunctor(NDArray* input, NDArray* indices, NDArray* output);

    void segmentSumFunctor(NDArray* input, NDArray* indices, NDArray* output);

    void segmentProdFunctor(NDArray* input, NDArray* indices, NDArray* output);

    void unsortedSegmentSqrtNFunctor(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMaxFunctor(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMinFunctor(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMeanFunctor(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentSumFunctor(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentProdFunctor(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    int segmentMaxFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentMinFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentMeanFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentSumFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentProdFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int unsortedSegmentSqrtNFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMaxFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMinFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMeanFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentSumFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentProdFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

}
}
}
#endif
