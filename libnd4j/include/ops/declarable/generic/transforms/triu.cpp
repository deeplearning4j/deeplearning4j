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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 31.03.2018
//

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>


namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(triu, 1, 1, false, 0, 0) {
	
    NDArray<T>* input  = INPUT_VARIABLE(0);
    NDArray<T>* output = OUTPUT_VARIABLE(0);

    REQUIRE_TRUE(input->rankOf() > 0, 0, "TRIU OP: the rank of input array must be > 0, but got %i instead !", input->rankOf());

    const int diag = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;
    
    helpers::triu(*input, *output, diag);

    return Status::OK();
}


DECLARE_SHAPE_FN(triu) {

	auto inShapeInfo = inputShape->at(0);

    REQUIRE_TRUE(inShapeInfo[0] > 0, 0, "TRIU OP: the rank of input array must be > 0, but got %i instead !", inShapeInfo[0]);

    int rank = (inShapeInfo[0] == 1) ? 2 : inShapeInfo[0];
    
    Nd4jLong *outShapeInfo = nullptr;
	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);    
    memcpy(outShapeInfo, inShapeInfo, (1 + rank) * sizeof(Nd4jLong));                     // copy rank and dimensions values only

    if(inShapeInfo[0] == 1) {
        outShapeInfo[0] = rank; 
        outShapeInfo[1] = inShapeInfo[1];
        outShapeInfo[2] = inShapeInfo[1];
    }

	shape::updateStrides(outShapeInfo, shape::order(inShapeInfo));

    return SHAPELIST(outShapeInfo);    
}



//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(triu_bp, 2, 1, false, 0, 0) {
    
    NDArray<T>* input = INPUT_VARIABLE(0);
    NDArray<T>* gradO = INPUT_VARIABLE(1);              // dLoss/dO

    NDArray<T>* gradI = OUTPUT_VARIABLE(0);              // dLoss/dI

    REQUIRE_TRUE(input->rankOf() > 0, 0, "TRIU_BP OP: the rank of input array must be > 0, but got %i instead !", input->rankOf());

    const int diag = block.getIArguments()->size() > 0 ? INT_ARG(0) : 0;

    helpers::triuBP(*input, *gradO, *gradI, diag);

    return Status::OK();
}


DECLARE_SHAPE_FN(triu_bp) {

    auto gradOShapeInfo = inputShape->at(1);
    int rank = gradOShapeInfo[0];

    Nd4jLong* outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);    
    memcpy(outShapeInfo, gradOShapeInfo, (1 + rank) * sizeof(Nd4jLong));                     // copy rank and dimensions values only    

    shape::updateStrides(outShapeInfo, shape::order(inputShape->at(0)));

    return SHAPELIST(outShapeInfo);    
}


}
}