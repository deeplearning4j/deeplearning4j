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
// @author George A. Shulinok <sgazeos@gmail.com>, created on 4/18/2019.
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_barnes_symmetrized)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/BarnesHutTsne.h>

namespace nd4j {
namespace ops  {
		
		CUSTOM_OP_IMPL(barnes_symmetrized, 3, 1, false, 0, 0) {
    		auto rowP  = INPUT_VARIABLE(0);
            auto colP  = INPUT_VARIABLE(1);
            auto valP  = INPUT_VARIABLE(2);

    		auto output = OUTPUT_VARIABLE(0);

	 	 	helpers::barnes_symmetrize(rowP, colP, valP, output);

		    return Status::OK();
		}

		DECLARE_TYPES(barnes_symmetrized) {
			getOpDescriptor()
				->setAllowedInputTypes(nd4j::DataType::ANY)
				->setSameMode(true);
		}

		DECLARE_SHAPE_FN(barnes_symmetrized) {
    		auto inputShapeInfo = inputShape->at(0);

    		const int inRank = inputShapeInfo[0];

    		// input validation
    		REQUIRE_TRUE(inRank == 2 ||  inRank == 4 || inRank == 6, 0, "DIAG_PART op: input array must have rank among following three possible values: 2, 4, 6, but got %i instead !", inRank);
    		for(int i = 1; i < inRank; ++i)
    			REQUIRE_TRUE(inputShapeInfo[i] == inputShapeInfo[i+1], 0, "DIAG_PART op: wrong shape of input array %s ! All dimensions must be equal !", ShapeUtils::shapeAsString(inputShapeInfo).c_str());

    		Nd4jLong* outShapeInfo = nullptr;

			int outRank = inRank/2;

			ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outRank), Nd4jLong);
	
			outShapeInfo[0] = outRank;
			for(int i = 1; i <= outRank; ++i)
				outShapeInfo[i] = inputShapeInfo[i];

			ShapeUtils::updateStridesAndType(outShapeInfo, inputShapeInfo, shape::order(inputShapeInfo));

    		return SHAPELIST(outShapeInfo);
		}

}
}

#endif