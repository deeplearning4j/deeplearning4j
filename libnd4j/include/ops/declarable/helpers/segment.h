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

    bool segmentIndicesValidate(nd4j::LaunchContext * context, NDArray* indices, NDArray& expected, NDArray& output);

    bool unsortedSegmentIndicesValidate(nd4j::LaunchContext * context, NDArray* indices, Nd4jLong numOfClasses, Nd4jLong& output);

    void segmentMaxFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentMinFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentMeanFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentSumFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentProdFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void unsortedSegmentSqrtNFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMaxFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMinFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMeanFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentSumFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentProdFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    int segmentMaxFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentMinFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentMeanFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentSumFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentProdFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int unsortedSegmentSqrtNFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMaxFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMinFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMeanFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentSumFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentProdFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

}
}
}
#endif
