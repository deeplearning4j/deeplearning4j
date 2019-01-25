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

    bool segmentIndicesValidate(graph::LaunchContext* context, NDArray* indices, NDArray& expected, NDArray& output);

    bool unsortedSegmentIndicesValidate(graph::LaunchContext* context, NDArray* indices, Nd4jLong numOfClasses, Nd4jLong& output);

    void segmentMaxFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentMinFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentMeanFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentSumFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentProdFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* output);

    void unsortedSegmentSqrtNFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMaxFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMinFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMeanFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentSumFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentProdFunctor(graph::LaunchContext* context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    int segmentMaxFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentMinFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentMeanFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentSumFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentProdFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int unsortedSegmentSqrtNFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMaxFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMinFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMeanFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentSumFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentProdFunctorBP(graph::LaunchContext* context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

}
}
}
#endif
