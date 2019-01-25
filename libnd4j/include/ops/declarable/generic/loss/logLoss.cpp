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
#if NOT_EXCLUDED(OP_log_loss)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(log_loss, 3, 1, false, 1, 1) {
  	
  	auto predictions = INPUT_VARIABLE(0);
    auto weights     = INPUT_VARIABLE(1);
    auto labels      = INPUT_VARIABLE(2);
    
    auto output      = OUTPUT_VARIABLE(0);

    int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    // FIXME: double?
    double epsilon = T_ARG(0);

    // input validation
    REQUIRE_TRUE(labels->isSameShape(predictions), 0, "LOG_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0, "LOG_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i correspondingly!", weights->rankOf(), labels->rankOf());
    // check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0, "LOG_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s instead!", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
    // only 4 possible reduction modes exist
    REQUIRE_TRUE(reductionMode==0 || reductionMode==1 || reductionMode==2 || reductionMode==3, 0, "LOG_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!", reductionMode);
    
	// perform weights broadcasting/tile to predictions if needed
	auto weightsBroad = weights;
	if(!weights->isScalar() && !weights->isSameShape(predictions))
		weightsBroad = new NDArray(weights->tileToShape(predictions->getShapeInfo()));

	NDArray E = -(*labels)*((*predictions + epsilon).transform(transform::Log)) - (1. - *labels)*(((1. + epsilon) - *predictions).transform(transform::Log));

    // multiply E on weights
    E *= *weightsBroad;
 	
	switch (reductionMode) {
		case 0: {											// 0 - "none", un-reduced weighted losses with the same shape as labels.
			output->assign(E);
			break;
		}
		case 1: {											// 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
			E.reduceNumber(reduce::Sum, *output);
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
				output->assign(E.reduceNumber(reduce::Sum) / double(numOfNonZeroWeights));
			break;
		}
	}

    if(weightsBroad != weights)
    	delete weightsBroad;
	
    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(log_loss) {
			
	getOpDescriptor()->setAllowedInputTypes(nd4j::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(log_loss) {

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






//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(log_loss_grad, 3, 3, false, 1, 1) {
  	
  	auto predictions = INPUT_VARIABLE(0);
    auto weights     = INPUT_VARIABLE(1);
    auto labels      = INPUT_VARIABLE(2);
    
    auto dLdp = OUTPUT_VARIABLE(0);		// dL/dpredictions
    auto dLdw = OUTPUT_VARIABLE(1);		// dL/dweights
    auto dLdl = OUTPUT_VARIABLE(2);		// dL/dlabels

    int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"  
    // take into account Alex's proposition to treat "none" the same as "weighted_sum" mode when calculating gradients
    if(reductionMode == 0)
    	reductionMode = 1;
    
    // FIXME: double?   
    double epsilon = T_ARG(0);

    // input validation
    REQUIRE_TRUE(labels->isSameShape(predictions), 0, "LOG_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0, "LOG_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i and %i correspondingly!", weights->rankOf(), labels->rankOf());
    // check whether broadcast operation is possible for weights array    
    REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0, "LOG_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s instead!", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
    // only 4 possible reduction modes exist
    REQUIRE_TRUE(reductionMode==0 || reductionMode==1 || reductionMode==2 || reductionMode==3, 0, "LOG_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!", reductionMode);
	
	// perform weights broadcasting/tile to labels if needed	
	auto weightsBroad = weights;
	if(!weights->isScalar() && !weights->isSameShape(predictions))
		weightsBroad = new NDArray(weights->tileToShape(predictions->getShapeInfo()));

	NDArray predictPlusEps = *predictions + epsilon;
	NDArray oneMinusLabels = 1. - *labels;
	NDArray onePlusEpsMinusPredict = (1. + epsilon) - *predictions;

	// dE_i/dp_i = (1-y_i)/(1-p_i+eps) - y_i/(p_i+eps)
	dLdp->assign(oneMinusLabels / onePlusEpsMinusPredict - *labels / predictPlusEps);	// dE/dp	
	// dE_i/dy_i = log((1+2eps)/(p_i+eps) - 1)
	((1. + 2. * epsilon) / predictPlusEps  - 1.).applyTransform(transform::Log, dLdl);		// dE/dy

	NDArray E = -(*labels) * predictPlusEps.transform(transform::Log) - oneMinusLabels * onePlusEpsMinusPredict.transform(transform::Log);
	
 	// process 3 possible reduction modes below
	switch (reductionMode) {		
		case 1: {											// 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

			*dLdp *= *weightsBroad;
			*dLdl *= *weightsBroad;
			
			if(weights->isScalar())
				dLdw->assign(E.reduceNumber(reduce::Sum));
			else if(weights != weightsBroad) {
				std::vector<int> axesToReduceAlong = ShapeUtils::evalBroadcastBackwardAxis(weights->getShapeInfo(), weightsBroad->getShapeInfo());				
				E.reduceAlongDimension(reduce::Sum, dLdw, axesToReduceAlong, true, false, false);
			}
			else
				dLdw->assign(E);

			break;
		}
		case 2: {											// 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of all elements of weightsBroad array

			NDArray sum;
			if (weights->isScalar())
				sum = (*weights) * E.lengthOf();
			else 
				sum = weightsBroad->reduceNumber(reduce::Sum);
			
			if (sum.e<double>(0) == 0.) {
				*dLdp = 0.;
				*dLdl = 0.;
				*dLdw = 0.;
			}
			else {

				NDArray temp = *weightsBroad / sum;
				*dLdp *= temp;
				*dLdl *= temp;
				
				if(weights->isScalar())
					*dLdw = 0.;
				else if(weights != weightsBroad) {
					std::vector<int> axesToReduceAlong = ShapeUtils::evalBroadcastBackwardAxis(weights->getShapeInfo(), weightsBroad->getShapeInfo());				
					((E * sum - (E * *weightsBroad).reduceNumber(reduce::Sum)) / (sum*sum)).reduceAlongDimension(reduce::Sum, dLdw, axesToReduceAlong, true, false, false);
				}					
				else 
					dLdw->assign((E * sum - (E * *weightsBroad).reduceNumber(reduce::Sum)) / (sum*sum));					
			}
			break;
		}
		case 3: {											// 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E array divided by number of non-zero weights			
		
			Nd4jLong numOfNonZeroWeights = 0;
			if(weights->isScalar()) {
				if(weights->e<double>(0) != 0.)
					numOfNonZeroWeights = E.lengthOf();
			}
			else 
				numOfNonZeroWeights = weightsBroad->reduceNumber(reduce::CountNonZero).e<Nd4jLong>(0);			

			if (numOfNonZeroWeights == 0) {
				*dLdp = 0.;
				*dLdl = 0.;
				*dLdw = 0.;
			}
			else {
				auto numOfNonZeroWeightsScalar = NDArrayFactory::create(dLdw->dataType(), numOfNonZeroWeights, block.getWorkspace());
				if(weights->isScalar())
					dLdw->assign(E.reduceNumber(reduce::Sum) / numOfNonZeroWeights);
				else if(weights != weightsBroad) {
					std::vector<int> axesToReduceAlong = ShapeUtils::evalBroadcastBackwardAxis(weights->getShapeInfo(), weightsBroad->getShapeInfo());
					E.reduceAlongDimension(reduce::Sum, dLdw, axesToReduceAlong, true, false, false);
					*dLdw /= numOfNonZeroWeightsScalar;
				}
				else
					dLdw->assign(E / numOfNonZeroWeightsScalar);
				
				NDArray temp = *weightsBroad / numOfNonZeroWeightsScalar;
				*dLdp *= temp;
				*dLdl *= temp;
			}
			break;
		}
	}

    if(weightsBroad != weights)
    	delete weightsBroad;
	
    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(log_loss_grad) {
			
	getOpDescriptor()->setAllowedInputTypes(nd4j::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(log_loss_grad) {

    auto predictionsShapeInfo = inputShape->at(0);
	auto weightsShapeInfo 	  = inputShape->at(1);
    auto labelsShapeInfo  	  = inputShape->at(2);

    // labels and predictions must have the same shapes 
    REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0, "LOG_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());    
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0, "LOG_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i and %i correspondingly!", shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
    // check whether broadcast operation is possible for weights array    
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0, "LOG_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s instead!", ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

    DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));    

    Nd4jLong *dLdpShapeInfo = ShapeBuilders::copyShapeInfoAndType(predictionsShapeInfo, outType, false, block.getWorkspace());    
    Nd4jLong *dLdwShapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, outType, false, block.getWorkspace());    
    Nd4jLong *dLdlShapeInfo = ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());    
	
    return SHAPELIST(dLdpShapeInfo, dLdwShapeInfo, dLdlShapeInfo);
}



}
}

#endif