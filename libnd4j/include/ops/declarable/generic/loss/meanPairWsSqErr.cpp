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
	REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0, "MEAN_PAIRWSSQERR_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i correspondingly!", weights->rankOf(), labels->rankOf());
	    // check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0, "MEAN_PAIRWSSQERR_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s instead!", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());

    if(labels->rankOf() == 1) { // If labels and predictions are of rank 1, it means that all data entries are 0-tensor (scalar) so that the result of becomes always zero.
    	*output = 0.;
    	return Status::OK();
    }
    
    // perform weights broadcasting/tile to predictions if needed
	auto weightsBroad = weights;
	if(!weights->isScalar() && !weights->isSameShape(predictions))
		weightsBroad = new NDArray(weights->tileToShape(predictions->getShapeInfo()));	
	
	NDArray diffs = *predictions - *labels;		

	std::vector<int> reductionIdx = ShapeUtils::evalDimsToExclude(diffs.rankOf(), {0});	
	NDArray sumSqrsDiffPerBatch = (diffs*diffs).reduceAlongDims(reduce::Sum, reductionIdx, true);

	NDArray numOfNonZeroWeights(sumSqrsDiffPerBatch.getShapeInfo(), nd4j::DataType::INT64, false, block.getWorkspace());
	if(weights->isScalar()) {
		if((*weights).e<double>(0) != 0.)
			numOfNonZeroWeights.assign((labels->lengthOf()/labels->sizeAt(0)));
	}
	else 		
		numOfNonZeroWeights.assign(weightsBroad->reduceAlongDims(reduce::CountNonZero, reductionIdx));	

	NDArray numOfNonZeroWeightsMinusOne = numOfNonZeroWeights;// - 1LL;
	numOfNonZeroWeightsMinusOne -= 1LL;
	
	sumSqrsDiffPerBatch.applyPairwiseTransform(pairwise::SafeDivide, numOfNonZeroWeightsMinusOne, nullptr);

	auto sumDiff = diffs.reduceAlongDims(reduce::Sum, reductionIdx, true);
	
	auto nonZerosSquared = numOfNonZeroWeights;
	nonZerosSquared.applyPairwiseTransform(pairwise::Multiply, numOfNonZeroWeightsMinusOne, nullptr);
	(sumDiff*sumDiff).applyPairwiseTransform(pairwise::SafeDivide, &nonZerosSquared, &sumDiff, nullptr);
	
	auto E = (sumSqrsDiffPerBatch - sumDiff);
	E *= 2.f;

    // multiply E on weights
    E *= *weights;

	if(numOfNonZeroWeights.reduceNumber(reduce::Sum).e<double>(0) == 0.)
		*output = 0.;
	else
		*output = E.reduceNumber(reduce::Sum);
    
    if(weightsBroad != weights)
    	delete weightsBroad;
	
    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(mean_pairwssqerr_loss) {
		
	getOpDescriptor()->setAllowedInputTypes(nd4j::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(mean_pairwssqerr_loss) {

	auto predictionsShapeInfo = inputShape->at(0);
	auto weightsShapeInfo 	  = inputShape->at(1);
    auto labelsShapeInfo 	  = inputShape->at(2);

    REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0, "MEAN_PAIRWSSQERR_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());    
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0, "MEAN_PAIRWSSQERR_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i correspondingly!", shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
    // check whether broadcast operation is possible for weights array    
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0, "MEAN_PAIRWSSQERR_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s instead!", ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());
    
    DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));
    Nd4jLong* outShapeInfo = ShapeBuilders::createScalarShapeInfo(outType, block.getWorkspace());

    return SHAPELIST(outShapeInfo);    
}




}
}

#endif