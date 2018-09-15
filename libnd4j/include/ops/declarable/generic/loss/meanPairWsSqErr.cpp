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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 24.11.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_mean_pairwssqerr_loss)

#include <ops/declarable/CustomOperations.h>
#include <numeric>
#include <iostream>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(mean_pairwssqerr_loss, 3, 1, false, 0, 0) {
  	auto predictions = INPUT_VARIABLE(0);
    auto weights     = INPUT_VARIABLE(1);
    auto labels      = INPUT_VARIABLE(2);
    auto output      = OUTPUT_VARIABLE(0);

	// input validation    
    REQUIRE_TRUE(labels->isSameShape(predictions), 0, "MEAN_PAIRWSSQERR_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
	REQUIRE_TRUE(!(!weights->isScalar() && weights->rankOf() != labels->rankOf()), 0, "MEAN_PAIRWSSQERR_LOSS OP: weights array must have the same rank as labels array, but got %i and %i correspondingly!", weights->rankOf(), labels->rankOf());
    // check whether broadcast operation is possible for weights array
    if(!weights->isScalar())
    	for (int i = 0; i < weights->rankOf(); ++i)
           	REQUIRE_TRUE(!(weights->shapeOf()[i] != labels->shapeOf()[i] && weights->shapeOf()[i] != 1), 0, "MEAN_PAIRWSSQERR_LOSS OP: shape of weights array %s is not broadcastable to labels array shape %s !", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());

	// perform weights broadcasting/tile to labels if needed	
	auto weightsBroad = weights;
	if(!weights->isScalar() && !weights->isSameShape(predictions)) {
		// evaluate repeat dimensions for tile operation
		std::vector<Nd4jLong> reps;
		for(int i = 0; i < labels->rankOf(); ++i)
			reps.emplace_back(labels->shapeOf()[i] / weights->shapeOf()[i]);
		weightsBroad = new NDArray(weights->tile(reps));
	}	
	
	auto diffs = *predictions - *labels;
	std::vector<int> reductionIdx(diffs.rankOf()-1);
	std::iota(reductionIdx.begin(), reductionIdx.end(), 1);
	auto sumSqrsDiffPerBatch = (diffs*diffs).reduceAlongDims(reduce::Sum, reductionIdx, true);

	NDArray numOfNonZeroWeights(sumSqrsDiffPerBatch.getShapeInfo(), block.getWorkspace());
	if(weights->isScalar()) {
		if((*weights).getScalar<double>(0) != 0.)
			numOfNonZeroWeights.assign((labels->lengthOf()/labels->sizeAt(0)));
	}
	else {
		int sizeAtRestDims =  weightsBroad->lengthOf()/weightsBroad->sizeAt(0);
		/*
		for(int i = 0; i < numOfNonZeroWeights.lengthOf(); ++i)
			for(int j = 0; j < sizeAtRestDims; ++j)
				if((*weightsBroad)(i*sizeAtRestDims + j) != (T)0.)
					++numOfNonZeroWeights(i);
					*/
		throw std::runtime_error("Not implemented yet");
	}
	
	sumSqrsDiffPerBatch.applyPairwiseTransform(pairwise::SafeDivide, &numOfNonZeroWeights, nullptr);

	auto sumDiff = diffs.reduceAlongDims(reduce::Sum, reductionIdx, true);
	auto nonZerosSquared = numOfNonZeroWeights*numOfNonZeroWeights;
	(sumDiff*sumDiff).applyPairwiseTransform(pairwise::SafeDivide, &nonZerosSquared, &sumDiff, nullptr);
	
	auto weightedLosses = (sumSqrsDiffPerBatch - sumDiff) * 2.;

    // multiply weightedLosses on weights
    weightedLosses *= (*weights);
 		
	if(numOfNonZeroWeights.reduceNumber(reduce::Sum).getScalar<float>(0) == 0.f)
		(*output) = 0.f;
	else
		(*output) = weightedLosses.reduceNumber(reduce::Sum);


    STORE_RESULT(*output);

    if(weightsBroad != weights)
    	delete weightsBroad;
	
    return Status::OK();
}


DECLARE_SHAPE_FN(mean_pairwssqerr_loss) {

	auto predictionsShapeInfo = inputShape->at(0);
    auto labelsShapeInfo 	  = inputShape->at(2);

    // labels and predictions must have the same shapes
    REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0, "MEAN_PAIRWSSQERR_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());

    Nd4jLong* outShapeInfo = nullptr;
    // output is scalar
    ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2) /*rank=2*/, Nd4jLong);
    outShapeInfo[0] = 2;
    outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
    outShapeInfo[5] = 0;
    outShapeInfo[6] = 1;
    outShapeInfo[7] = 99;

    return SHAPELIST(outShapeInfo);    

}

// INT_ARG(0) - reduction mode



}
}

#endif