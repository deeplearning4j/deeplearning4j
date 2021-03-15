/* ******************************************************************************
 *
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 *  See the NOTICE file distributed with this work for additional
 *  information regarding copyright ownership.
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
#include <system/op_boilerplate.h>
#include <array/NDArray.h>

namespace sd {
namespace ops {
namespace helpers {

    bool segmentIndicesValidate(sd::LaunchContext * context, NDArray* indices, NDArray& expected, NDArray& output);

    bool unsortedSegmentIndicesValidate(sd::LaunchContext * context, NDArray* indices, Nd4jLong numOfClasses, Nd4jLong& output);

    void segmentMaxFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentMinFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentMeanFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentSumFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void segmentProdFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output);

    void unsortedSegmentSqrtNFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMaxFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMinFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentMeanFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentSumFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    void unsortedSegmentProdFunctor(sd::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output);

    int segmentMaxFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentMinFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentMeanFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentSumFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int segmentProdFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output);

    int unsortedSegmentSqrtNFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMaxFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMinFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentMeanFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentSumFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

    int unsortedSegmentProdFunctorBP(sd::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output);

}
}
}
#endif
