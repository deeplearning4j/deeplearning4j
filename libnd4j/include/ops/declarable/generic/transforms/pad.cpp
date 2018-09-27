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
// @author Shyrma Yurii (iuriish@yahoo.com), created on 06.11.2017.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_pad)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/transforms.h>
#include <numeric>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(pad, 2, 1, false, 0, 1) {

    auto input    = INPUT_VARIABLE(0);
    auto paddings = INPUT_VARIABLE(1);
    auto output   = OUTPUT_VARIABLE(0);
    std::vector<int>* argI = block.getIArguments();

    const int rank =  input->rankOf();    	

	// input validation
	std::string expectedPaddingsShape = ShapeUtils::shapeAsString({rank, 2});
	std::string currentPaddingsShape  = ShapeUtils::shapeAsString(paddings);
	REQUIRE_TRUE(expectedPaddingsShape == currentPaddingsShape, 0, "PAD op: wrong shape of paddings array, expected is %s, but got %s instead !", expectedPaddingsShape.c_str(), currentPaddingsShape.c_str());

	// FIXME: double
	auto padValue = NDArrayFactory::create(0.);
	// in case of REFLECT and SYMMETRIC modes paddings must obey additional shape requirements 
	// REFLECT case
	if (argI->at(0) == 0) { // CONSTAND mode
	    if (!block.getTArguments()->empty())
	        padValue = T_ARG(0);
    }
    else if(argI->at(0) == 1)
		for(int dim=0; dim < rank; ++dim)
			REQUIRE_TRUE(paddings->e<Nd4jLong>(dim,0) <= (input->shapeOf()[dim]-1) && paddings->e<Nd4jLong>(dim,1) <= (input->shapeOf()[dim]-1), 0, "PAD op: wrong content of paddings array for REFLECT mode !");
	// SYMMETRIC case
	if(argI->at(0) == 2)				
		for(int dim=0; dim < rank; ++dim)
			REQUIRE_TRUE(paddings->e<Nd4jLong>(dim,0) <= input->shapeOf()[dim] && paddings->e<Nd4jLong>(dim,1)  <= input->shapeOf()[dim], 0, "PAD op: wrong content of paddings array for SYMMETRIC mode !");
	// CONSTANT->0, REFLECT->1, SYMMETRIC->2
    REQUIRE_TRUE(!(argI->at(0) < 0 || argI->at(0) > 2), 0, "PAD op: unknown padding mode, there are only three possible legal values -> 0,1,2, but got %i instead !", argI->at(0));

	std::vector<int> dimensions(input->rankOf());
    std::iota(dimensions.begin(), dimensions.end(), 0);   			// fill with 0, 1, ... rank-1
    
	helpers::recursiveLoopForPad(argI->at(0), *input, *paddings, *output, dimensions, 0, 0, 0, padValue);
	
    return Status::OK();
}

DECLARE_SHAPE_FN(pad) {

	// check shape of paddings 
	auto inputShapeInfo = inputShape->at(0);
    auto paddings = INPUT_VARIABLE(1);
    const int rank =  inputShapeInfo[0];    	

    // paddings validation
    std::string expectedPaddingsShape = ShapeUtils::shapeAsString({rank, 2});
	std::string currentPaddingsShape  = ShapeUtils::shapeAsString(paddings);
	REQUIRE_TRUE(expectedPaddingsShape == currentPaddingsShape, 0, "PAD op: wrong shape of paddings array, expected is %s, but got %s instead !", expectedPaddingsShape.c_str(), currentPaddingsShape.c_str());
		
	Nd4jLong* outShapeInfo = nullptr;
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank), Nd4jLong);
    outShapeInfo[0] = rank;
    for(int i=1; i <= rank; ++i)
    	outShapeInfo[i] = inputShapeInfo[i] + paddings->e<Nd4jLong>(i-1,0) + paddings->e<Nd4jLong>(i-1,1);
	
    shape::updateStrides(outShapeInfo, shape::order(inputShapeInfo));
    ArrayOptions::setDataType(outShapeInfo, ArrayOptions::dataType(inputShapeInfo));
//    ArrayOptions::setDataType(outShapeInfo, block.dataType());
    return SHAPELIST(outShapeInfo);
    
}



}
}

#endif