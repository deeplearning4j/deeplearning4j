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
//  @author GS <sgazeos@gmail.com>
//

#include <ops/declarable/helpers/segment.h>

namespace nd4j {
namespace ops {
namespace helpers {

    // segment max
    template <typename T>
    static void segmentMaxFunctor_(NDArray* input, NDArray* indices, NDArray* output) {

    }

    // segmen min 
    template <typename T>
    static void segmentMinFunctor_(NDArray* input, NDArray* indices, NDArray* output) {

    }

    // segmen mean
    template <typename T>
    static void segmentMeanFunctor_(NDArray* input, NDArray* indices, NDArray* output) {

    }

    template <typename T>
    static void segmentSumFunctor_(NDArray* input, NDArray* indices, NDArray* output) {

    }

    template <typename T>
    static void segmentProdFunctor_(NDArray* input, NDArray* indices, NDArray* output) {

    }

    template <typename T>
    static bool segmentIndicesValidate_(NDArray* indices, NDArray& aexpected, NDArray& aoutput) {
        return true;
    }

    void segmentMaxFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentMaxFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    void segmentMinFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentMinFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    void segmentMeanFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentMeanFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    void segmentSumFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentSumFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    void segmentProdFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), segmentProdFunctor_, (input, indices, output), LIBND4J_TYPES);
    }

    bool segmentIndicesValidate(nd4j::LaunchContext * context, NDArray* indices, NDArray& expected, NDArray& output) {
        BUILD_SINGLE_SELECTOR(output.dataType(), return segmentIndicesValidate_, (indices, expected, output), LIBND4J_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template bool segmentIndicesValidate_, (NDArray*, NDArray&, NDArray&), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentProdFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentSumFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentMeanFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentMinFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    BUILD_SINGLE_TEMPLATE(template void segmentMaxFunctor_, (NDArray* input, NDArray* indices, NDArray* output), LIBND4J_TYPES);
    // -------------------------------------------------------------------------------------------------------------- //
    // Unsorted segment ops
    // -------------------------------------------------------------------------------------------------------------- //

    bool unsortedSegmentIndicesValidate(nd4j::LaunchContext * context, NDArray* indices, Nd4jLong expected, Nd4jLong& output) {
        return true;
    }

    template <typename T>
    static void unsortedSegmentMaxFunctor_(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

    void unsortedSegmentMaxFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentMaxFunctor_, (input, indices, numOfClasses, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template void unsortedSegmentMaxFunctor_, (NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    template <typename T>
    static void unsortedSegmentMinFunctor_(NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

    void unsortedSegmentMinFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentMinFunctor_, (input, indices, numOfClasses, output),
                              NUMERIC_TYPES);
    }

    BUILD_SINGLE_TEMPLATE(template void unsortedSegmentMinFunctor_, (NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    void unsortedSegmentMeanFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

    void unsortedSegmentSumFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

    void unsortedSegmentProdFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {
      //  BUILD_SINGLE_SELECTOR(input->dataType(), unsortedSegmentProdFunctor_, (input, indices, numOfClasses, output), NUMERIC_TYPES);
    }
    //BUILD_SINGLE_TEMPLATE(template void unsortedSegmentProdFunctor_, (NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    void unsortedSegmentSqrtNFunctor(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, Nd4jLong numOfClasses, NDArray* output) {

    }

    // -------------------------------------------------------------------------------------------------------------- //
    // Backpropagate ops helpers
    // -------------------------------------------------------------------------------------------------------------- //
    // Sorted backpropagate ops
    //

    // segment max
    template <typename T>
    int segmentMaxFunctorBP_(NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        return Status::OK();
    }

    int segmentMaxFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), return segmentMaxFunctorBP_, (input, indices, gradOut, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int segmentMaxFunctorBP_, (NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output), NUMERIC_TYPES);

    // segmen min
    int segmentMinFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        return Status::OK();
    }

    // segmen mean
    int segmentMeanFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        return Status::OK();
    }

    int segmentSumFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, NDArray* output) {
        return Status::OK();
    }

    // -------------------------------------------------------------------------------------------------------------- //
    // Unsorted backpropagate segment ops
    // -------------------------------------------------------------------------------------------------------------- //

    template <typename T>
    static int unsortedSegmentMaxFunctorBP_(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

    int unsortedSegmentMaxFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentMaxFunctorBP_, (input, indices, gradOut, numOfClasses, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int unsortedSegmentMaxFunctorBP_, (NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    template <typename T>
    static int unsortedSegmentMinFunctorBP_(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

    int unsortedSegmentMinFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentMinFunctorBP_, (input, indices, gradOut, numOfClasses, output), NUMERIC_TYPES);
    }
    BUILD_SINGLE_TEMPLATE(template int unsortedSegmentMinFunctorBP_, (NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output), NUMERIC_TYPES);

    int unsortedSegmentMeanFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

    int unsortedSegmentSumFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

    int unsortedSegmentProdFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

//    template <typename T>
    int unsortedSegmentSqrtNFunctorBP(nd4j::LaunchContext * context, NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
        return Status::OK();
    }

//    int unsortedSegmentSqrtNFunctorBP(NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output) {
//        BUILD_SINGLE_SELECTOR(output->dataType(), return unsortedSegmentSqrtNFunctorBP_, (input, indices, gradOut, numOfClasses, output), FLOAT_TYPES);
//    }
//    BUILD_SINGLE_TEMPLATE(template int unsortedSegmentSqrtNFunctorBP_, (NDArray* input, NDArray* indices, NDArray* gradOut, Nd4jLong numOfClasses, NDArray* output), FLOAT_TYPES);
}
}
}