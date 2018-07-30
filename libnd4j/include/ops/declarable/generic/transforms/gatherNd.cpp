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
// @author Shyrma Yurii (iuriish@yahoo.com), created on 23.01.2018
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_gather_nd)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>

namespace nd4j {
namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gather_nd, 2, 1, false, 0, 0) {

	NDArray<T>* input   = INPUT_VARIABLE(0);
    NDArray<T>* indices = INPUT_VARIABLE(1);
    NDArray<T>* output  = OUTPUT_VARIABLE(0);

    const int rankIn = input->rankOf();
    const int rankInd = indices->rankOf();    

    REQUIRE_TRUE(rankInd > 0, 0, "GATHER_ND op: array of indexes can't be single scalar, the requirement is: rank > 0, but got rank = %i instead!", rankInd);
    int lastIndDim = indices->sizeAt(-1);
    REQUIRE_TRUE(lastIndDim <= rankIn, 0, "GATHER_ND op: the last dimension of indices array must be <= rank of input array but got %i and %i correspondingly!", lastIndDim, rankIn);

    helpers::gatherND(*input, *indices, *output);
    
    return Status::OK();
}


DECLARE_SHAPE_FN(gather_nd) {

	auto inShapeInfoIn = inputShape->at(0);
    auto inShapeInfoInd = inputShape->at(1);
		
    const int rankIn = inShapeInfoIn[0];
    const int rankInd = inShapeInfoInd[0];
    REQUIRE_TRUE(rankInd > 0, 0, "GATHER_ND op: array of indexes can't be single scalar, the requirement is: rank > 0, but got rank = %i instead!", rankInd);    
    const int lastIndDim = inShapeInfoInd[rankInd];    
    REQUIRE_TRUE(lastIndDim <= rankIn, 0, "GATHER_ND op: the last dimension of indices array must be <= rank of input array but got %i and %i correspondingly!", lastIndDim, rankIn);

	int outRank = (rankInd - 1) + (rankIn - lastIndDim);

    Nd4jLong* outShapeInfo = nullptr;
	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);

    outShapeInfo[0] = outRank;

    for(int i = 1; i <= rankInd-1; ++i)
        outShapeInfo[i] = inShapeInfoInd[i];

    for(int i = 0; i < rankIn-lastIndDim; ++i)
        outShapeInfo[rankInd + i] = inShapeInfoIn[lastIndDim + i + 1];

	shape::updateStrides(outShapeInfo, 'c');

    return SHAPELIST(outShapeInfo);    
}




}
}

#endif