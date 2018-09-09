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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 17.05.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_percentile)

#include <ops/declarable/CustomOperations.h>
#include <declarable/helpers/percentile.h>


namespace nd4j {
namespace ops  {

CUSTOM_OP_IMPL(percentile, 1, 1, false, 1, -2) {    
    auto input  = INPUT_VARIABLE(0);                                             // tensor with rank > 0
    auto output = OUTPUT_VARIABLE(0);                                            // [bS, oD, oH, oW, iC] (NDHWC) or [bS, iC, oD, oH, oW] (NCDHW)

    const T q = T_ARG(0);                                                             // percentile
    const int interpolation = block.getTArguments()->size() > 1 ? T_ARG(1) : (T)2.;     // 0-"lower", 1-"higher", 2-"nearest"(default)
    const int keepDims = block.getTArguments()->size() > 2 ? T_ARG(2) : (T)0.;          // false is default

    const int axisArrRank = block.getIArguments()->size();
    const int inputArrRank = input->rankOf();

    REQUIRE_TRUE(inputArrRank > 0, 0, "PERCENTILE OP: rank of input array must be positive (>0), but got %i instead !", inputArrRank);
    REQUIRE_TRUE(0.f <= q && q <= 100.f, 0, "PERCENTILE OP: percentile parameter must be within [0, 100] range, but got %f instead !", q);
    REQUIRE_TRUE(interpolation == 0 || interpolation == 1 || interpolation == 2, 0, "PERCENTILE OP: the correct values for interpolation parameter are 0, 1, 2, but got %i instead !", interpolation);
    REQUIRE_TRUE(axisArrRank <= inputArrRank, 0, "PERCENTILE OP: the rank of axis array must be <= rank of input array, but got %i and %i correspondingly !", axisArrRank, inputArrRank);

    for(int i = 0; i < axisArrRank; ++i) {
        int dim = INT_ARG(i) >= 0 ? INT_ARG(i) : INT_ARG(i) + inputArrRank;
        REQUIRE_TRUE(dim < inputArrRank, 0, "PERCENTILE OP: element (dimension) of axis array at position %i is >= rank of input array (%i >= %i), which is unacceptable !", i, dim, inputArrRank);
    }

    std::vector<int> axises = *block.getIArguments();
    helpers::percentile<T>(*input, *output, axises, q, interpolation);

    return Status::OK();
}


DECLARE_SHAPE_FN(percentile) {

    Nd4jLong* inputShapeInfo = inputShape->at(0);
    const int keepDims = block.getTArguments()->size() > 2 ? T_ARG(2) : 0.;        // false is default

    const int axisArrRank = block.getIArguments()->size();
    const int inputArrRank = inputShapeInfo[0];    

    REQUIRE_TRUE(inputArrRank > 0, 0, "PERCENTILE OP: rank of input array must be positive (>0), but got %i instead !", inputArrRank);
    REQUIRE_TRUE(axisArrRank <= inputArrRank, 0, "PERCENTILE OP: the rank of axis array must be <= rank of input array, but got %i and %i correspondingly !", axisArrRank, inputArrRank);    

    for(int i = 0; i < axisArrRank; ++i) {
        int dim = INT_ARG(i) >= 0 ? INT_ARG(i) : INT_ARG(i) + inputArrRank;
        REQUIRE_TRUE(dim < inputArrRank, 0, "PERCENTILE OP: element (dimension) of axis array at position %i is >= rank of input array (%i >= %i), which is unacceptable !", i, dim, inputArrRank);
    }

    std::vector<int> axises = *block.getIArguments();
    Nd4jLong* outputShapeInfo = ShapeUtils<T>::evalReduceShapeInfo(shape::order(inputShapeInfo), axises, inputShapeInfo, keepDims, false, block.getWorkspace());

    return SHAPELIST(outputShapeInfo);
}


}
}

#endif