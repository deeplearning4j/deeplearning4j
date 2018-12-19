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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 23.11.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_huber_loss)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(huber_loss, 3, 1, false, 1, 1) {
  	auto predictions = INPUT_VARIABLE(0);
    auto weights     = INPUT_VARIABLE(1);
    auto labels      = INPUT_VARIABLE(2);
    auto output      = OUTPUT_VARIABLE(0);

    int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    // FIXME: double?
    double delta = T_ARG(0);

       // input validation
    REQUIRE_TRUE(labels->isSameShape(predictions), 0, "HUBER_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0, "HUBER_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i correspondingly!", weights->rankOf(), labels->rankOf());
    // check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0, "HUBER_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s instead!", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
    // only 4 possible reduction modes exist
    REQUIRE_TRUE(reductionMode==0 || reductionMode==1 || reductionMode==2 || reductionMode==3, 0, "HUBER_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!", reductionMode);
    
	// perform weights broadcasting/tile to predictions if needed
	auto weightsBroad = weights;
	if(!weights->isScalar() && !weights->isSameShape(predictions))
		weightsBroad = new NDArray(weights->tileToShape(predictions->getShapeInfo()));

	auto error = *predictions - *labels;
	error.applyTransform(transform::Abs, nullptr, nullptr);
	NDArray quadratic(error.getShapeInfo(), block.getWorkspace());
	error.applyScalar(scalar::MinPairwise, delta, &quadratic);
 
    NDArray E = quadratic * quadratic * 0.5f + (error - quadratic)*delta;

    // multiply E on weights
     E *= *weightsBroad;
 	
	switch (reductionMode) {		
		case 0: {											// 0 - "none", un-reduced weighted losses with the same shape as labels.
			output->assign(E);
			break;
		}
		case 1: {											// 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
			(*output) = E.reduceNumber(reduce::Sum);
			break;
		}
		case 2: {											// 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of all elements of weightsBroad array
			NDArray sum;
			if (weights->isScalar())
				sum = *weights * E.lengthOf();
			else 
				sum = weightsBroad->reduceNumber(reduce::Sum);
			
			if (sum.e<double>(0) == 0.)
				*output = 0.;
			else 
				output->assign(E.reduceNumber(reduce::Sum) / sum);
			break;
		}
		case 3: {											// 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E array divided by number of non-zero weights
			Nd4jLong numOfNonZeroWeights = 0;
			if(weights->isScalar()) {
				if(weights->e<double>(0) != 0.)
					numOfNonZeroWeights = E.lengthOf();
			}
			else {
				numOfNonZeroWeights = weightsBroad->reduceNumber(reduce::CountNonZero).e<Nd4jLong>(0);
			}

			if (numOfNonZeroWeights == 0)
				(*output) = 0.;
			else 
				(*output) = E.reduceNumber(reduce::Sum) / numOfNonZeroWeights;
			break;
		}		
	}
   
    if(weightsBroad != weights)
    	delete weightsBroad;
	
    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(huber_loss) {
	
	getOpDescriptor()->setAllowedInputTypes(nd4j::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(huber_loss) {

	auto predictionsShapeInfo = inputShape->at(0);
	auto weightsShapeInfo 	  = inputShape->at(1);
    auto labelsShapeInfo  	  = inputShape->at(2);

     // labels and predictions must have the same shapes 
    REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0, "LOG_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());    
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0, "LOG_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i correspondingly!", shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
    // check whether broadcast operation is possible for weights array    
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0, "LOG_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s instead!", ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

    DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));
    Nd4jLong* outShapeInfo = nullptr;

    if(INT_ARG(0) != 0) 			// in this case output is scalar
    	outShapeInfo = ShapeBuilders::createScalarShapeInfo(outType, block.getWorkspace());
    else 							// in this case output has the same shape as labels and predictions
    	outShapeInfo = ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());

    return SHAPELIST(outShapeInfo);    
}


}
}

#endif