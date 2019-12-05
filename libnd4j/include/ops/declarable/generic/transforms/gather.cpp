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
// @author Shyrma Yurii (iuriish@yahoo.com), created on 16.11.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_gather)

#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/helpers/gather.h>
#include <ops/declarable/helpers/scatter.h>


namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(gather, 1, 1, false, 0, -2) {

	auto input   = INPUT_VARIABLE(0);
    auto indices = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;
	auto output  = OUTPUT_VARIABLE(0);

	const bool checkIndices = block.getBArguments()->empty() ? false : B_ARG(0);

	//Edge case: empty indices -> empty output
	if(indices != nullptr && indices->isEmpty()){
		REQUIRE_TRUE(output->isEmpty(), 0, "Gather op: If indices are empty, output must also be empty");
		return Status::OK();	//No op
	}

	const int numOfIntArgs = block.numI();

    std::vector<int> intArgs;
    if (block.width() > 2) {
    	intArgs = INPUT_VARIABLE(2)->template asVectorT<int>();
    }
    else {
		if (numOfIntArgs == 0)
			intArgs.emplace_back(0);
		else
			for (int i = 0; i < numOfIntArgs; ++i)
				intArgs.emplace_back(block.getIArguments()->at(i));
	}

    const int inputRank = input->rankOf();
	if(intArgs[0] < 0)
        intArgs[0] += inputRank;

	// input validation
    REQUIRE_TRUE(intArgs[0] < inputRank, 0, "GATHER op: input axis must be smaller than input array rank, but got %i and %i correspondingly!", intArgs[0], inputRank);
    REQUIRE_TRUE(indices != nullptr || numOfIntArgs > 1, 0, "GATHER op: indices should be provided either as additional input array or as IntArguments !");

	if(checkIndices) {

		NDArray* pIndices = indices;
		if(indices == nullptr)
            pIndices = new NDArray(input->ordering(), {static_cast<int>(intArgs.size()) - 1}, std::vector<double>(intArgs.begin() + 1, intArgs.end()), DataType::INT64, block.launchContext());
        const Nd4jLong numOfBadIndx = helpers::checkIndices(block.launchContext(), *pIndices, *input, intArgs[0]);
        REQUIRE_TRUE(numOfBadIndx == 0, 0, "GATHER OP: please check elements of indices-array, total number of wrong elements is %lld!", numOfBadIndx);
        if(indices == nullptr)
            delete pIndices;
    }

	helpers::gather(block.launchContext(), input, indices, output, intArgs);

    return Status::OK();
}

DECLARE_TYPES(gather) {
	getOpDescriptor()->setAllowedInputTypes(0, {ALL_INTS, ALL_FLOATS});
	getOpDescriptor()->setAllowedInputTypes(1, {ALL_INTS});
	getOpDescriptor()->setAllowedOutputTypes(0, {ALL_INTS, ALL_FLOATS});
}


DECLARE_SHAPE_FN(gather) {

	// check shape of paddings
	auto inputShapeInfo  = inputShape->at(0);
	Nd4jLong* outputShapeInfo = nullptr;

	int axis = 0;

	if (block.width() > 2) {
		axis = INPUT_VARIABLE(2)->e<int>(0);
	} else
		axis = block.numI() > 0 ? block.getIArguments()->at(0) : 0;

	int inputRank = shape::rank(inputShapeInfo);
	if(axis < 0)
		axis += inputRank;

    REQUIRE_TRUE(axis < inputRank, 0, "GATHER op: input axis must be smaller than input array rank, but got %i and %i correspondingly!", axis, inputRank);

	bool isEmpty = false;

	if (block.width() > 1) {
		auto indicesShapeInfo = inputShape->at(1);

    	int indicesRank = shape::rank(indicesShapeInfo);

    	int outputRank = inputRank + indicesRank - 1;

    	ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outputRank), Nd4jLong);

    	// fill output shapeInfo
    	outputShapeInfo[0] = outputRank;
    	int shapeIdx = 1;

    	for(int i = 0; i < axis; ++i)
    		outputShapeInfo[shapeIdx++] = inputShapeInfo[i+1];

		for(int i = 0; i < indicesRank; ++i)
    		outputShapeInfo[shapeIdx++] = indicesShapeInfo[i+1];

    	for(int i = axis+1; i < inputRank; ++i)
    		outputShapeInfo[shapeIdx++] = inputShapeInfo[i+1];
	}
	else if (block.numI() > 1) {

		int indicesRank = block.numI() == 2 ? 0 : 1;

		int outputRank = inputRank + indicesRank - 1;
		ALLOCATE(outputShapeInfo, block.getWorkspace(), shape::shapeInfoLength(outputRank), Nd4jLong);

		// building shape manually
		outputShapeInfo[0] = outputRank;
    	int shapeIdx = 1;
    	for(int i = 0; i < axis; ++i)
    		outputShapeInfo[shapeIdx++] = inputShapeInfo[i+1];

		if (block.numI() > 2)
			outputShapeInfo[shapeIdx++] = block.numI() - 1;

		for(int i = axis+1; i < inputRank; ++i)
    		outputShapeInfo[shapeIdx++] = inputShapeInfo[i+1];
	}
    else
        REQUIRE_TRUE(false, 0, "GATHER op: indices should be provided either as additional input array or as IntArguments !");

	ShapeUtils::updateStridesAndType(outputShapeInfo, inputShapeInfo, shape::order(inputShapeInfo));

	if(isEmpty){
		ArrayOptions::setPropertyBit(outputShapeInfo, ARRAY_EMPTY);
	}

	auto result = ConstantShapeHelper::getInstance()->createShapeInfo(ShapeDescriptor(outputShapeInfo));
	RELEASE(outputShapeInfo, block.getWorkspace());
    return SHAPELIST(result);

}

}
}


#endif
