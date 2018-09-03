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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 31.08.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_histogram_fixed_width)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/histogramFixedWidth.h>

namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(histogram_fixed_width, 2, 1, false, 0, 0) {

    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* range  = INPUT_VARIABLE(1);
    NDArray<T>* output = OUTPUT_VARIABLE(0);    

    const int nbins = block.getIArguments()->empty() ? 100 : INT_ARG(0);

    const T leftEdge  = (*range)(0.);
    const T rightEdge = (*range)(1);

    REQUIRE_TRUE(leftEdge < rightEdge, 0, "HISTOGRAM_FIXED_WIDTH OP: wrong content of range input array, bottom_edge must be smaller than top_edge, but got %f and %f correspondingly !", leftEdge, rightEdge);
    REQUIRE_TRUE(nbins >= 1, 0, "HISTOGRAM_FIXED_WIDTH OP: wrong nbins value, expected value should be >= 1, however got %i instead !", nbins);

    helpers::histogramFixedWidth<T>(*input, *range, *output);

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(histogram_fixed_width) {
            
    Nd4jLong* outShapeInfo;
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(1), Nd4jLong);

    const int nbins = block.getIArguments()->empty() ? 100 : INT_ARG(0);
    shape::shapeVector(nbins, outShapeInfo);
       
    return SHAPELIST(outShapeInfo);
}


}
}

#endif