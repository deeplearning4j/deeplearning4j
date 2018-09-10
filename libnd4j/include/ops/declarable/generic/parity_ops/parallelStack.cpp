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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 01.11.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_parallel_stack)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/stack.h>

namespace nd4j {
namespace ops  {


CUSTOM_OP_IMPL(parallel_stack, -1, 1, false, 0, 0) {
	auto input  = INPUT_VARIABLE(0);
	auto output = OUTPUT_VARIABLE(0);
	
	// check whether shapes of all input array are the same				
	for (int i = 0; i < (int) block.width() - 1; ++i)
		REQUIRE_TRUE(shape::equalsSoft((INPUT_VARIABLE(i))->getShapeInfo(), (INPUT_VARIABLE(i+1))->getShapeInfo()), 0, "PARALLEL_STACK op: the shapes of all input arrays must be the same !");
 	 	
 	std::vector<NDArray*> inArrs(block.width());
 	for(int i = 0; i < block.width(); ++i)
		inArrs[i] = INPUT_VARIABLE(i);
	
	const int dim = 0;
	helpers::stack(inArrs, *output, dim);
	 	
  	return Status::OK();
}


DECLARE_SHAPE_FN(parallel_stack) {
	
	auto inShapeInfo = inputShape->at(0);
	int rank = inShapeInfo[0];

	Nd4jLong* outShapeInfo = nullptr;
	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(rank+1), Nd4jLong);

	outShapeInfo[0] = rank + 1;
	outShapeInfo[1] = block.width();
	for(int i = 1; i <= rank; ++i)
		outShapeInfo[i+1] = inShapeInfo[i];
	
	shape::updateStrides(outShapeInfo, shape::order(inShapeInfo));
  	
  	return SHAPELIST(outShapeInfo);
}


}
}

#endif