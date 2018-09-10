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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 25.11.2017
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_sigm_cross_entropy_loss)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(sigm_cross_entropy_loss, 3, 1, false, 1, 1) {
 	auto logits  = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto labels  = INPUT_VARIABLE(2);
    auto output  = OUTPUT_VARIABLE(0);

    int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    T labelsSmoothing = T_ARG(0);

    // input validation    
    REQUIRE_TRUE(labels->isSameShape(logits), 0, "SIGM_CROSS_ENTROPY_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly!", ShapeUtils<T>::shapeAsString(labels).c_str(), ShapeUtils<T>::shapeAsString(logits).c_str());        
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
	REQUIRE_TRUE(!(!weights->isScalar() && weights->rankOf() != labels->rankOf()), 0, "SIGM_CROSS_ENTROPY_LOSS OP: weights array must have the same rank as labels array, but got %i and %i correspondingly!", weights->rankOf(), labels->rankOf());
    // check whether broadcast operation is possible for weights array
    if(!weights->isScalar())
    	for (int i = 0; i < weights->rankOf(); ++i)
        	REQUIRE_TRUE(!(weights->shapeOf()[i] != labels->shapeOf()[i] && weights->shapeOf()[i] != 1), 0, "SIGM_CROSS_ENTROPY_LOSS OP: shape of weights array %s is not broadcastable to labels array shape %s !", ShapeUtils<T>::shapeAsString(weights).c_str(), ShapeUtils<T>::shapeAsString(labels).c_str());
    
	// perform weights broadcasting/tile to labels if needed	
	auto weightsBroad = weights;
	if(!weights->isScalar() && !weights->isSameShape(logits)) {
		// evaluate repeat dimensions for tile operation
		std::vector<Nd4jLong> reps;
		for(int i = 0; i < labels->rankOf(); ++i)
			reps.emplace_back(labels->shapeOf()[i] / weights->shapeOf()[i]);
		weightsBroad = new NDArray(weights->tile(reps));
	}	
	
	// If labelsSmoothing is nonzero, smooth the labels towards 1/2:
	auto newLabels = labels;
	if(labelsSmoothing != (T)0.) {
		auto smooth = LAMBDA_T(value, labelsSmoothing) { return value * ((T)1. - labelsSmoothing) + (T)(0.5) * labelsSmoothing; };
    	newLabels = new NDArray(*labels);
    	newLabels->applyLambda(smooth);  
	}
	
	NDArray weightedLosses(newLabels->getShapeInfo(), block.getWorkspace());
	auto sigm_cross_entropy_lossWithLogits = LAMBDA_TT(x, z) { return nd4j::math::nd4j_max(x, (T)0.) - x * z + nd4j::math::nd4j_log((T)1. + nd4j::math::nd4j_exp(-nd4j::math::nd4j_abs(x))); };	
	logits->applyPairwiseLambda(newLabels, sigm_cross_entropy_lossWithLogits, &weightedLosses);

    // multiply weightedLosses on weights
 	if(weights->isScalar())
 		weightedLosses *= (*weights)(0.);
 	else
 		weightedLosses *= (*weights); 	
 	// regard 4 possible reduction modes below
	REQUIRE_TRUE(reductionMode==0 || reductionMode==1 || reductionMode==2 || reductionMode==3, 0, "SIGM_CROSS_ENTROPY_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!", reductionMode);
	switch (reductionMode) {
		case 0:												// 0 - "none", un-reduced weighted losses with the same shape as labels.
			output->assign(&weightedLosses);
			break;
		
		case 1: {											// 1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
			(*output)(0.) = weightedLosses.reduceNumber(reduce::Sum);
			break;
		}
		case 2: {											// 2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
			T sum;
			if (weights->isScalar())
				sum = (*weights)(0.) * weightedLosses.lengthOf();
			else 
				sum = weightsBroad->reduceNumber(reduce::Sum);
			
			if (sum == (T)0.)
				(*output)(0.) = (T)0.;
			else 
				(*output)(0.) = weightedLosses.reduceNumber(reduce::Sum) / sum;
			break;
		}
		case 3: {											// 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
			int numOfNonZeroWeights = 0;
			if(weights->isScalar()) {
				if((*weights)(0.) != (T)0.)
					numOfNonZeroWeights = weightedLosses.lengthOf();
			}
			else {
				for(int i = 0; i < weightsBroad->lengthOf(); ++i)
					if((*weightsBroad)(i) != (T)0.)
						++numOfNonZeroWeights;
			}

			if (numOfNonZeroWeights == 0)
				(*output)(0.) = (T)0.;
			else 
				(*output)(0.) = weightedLosses.reduceNumber(reduce::Sum) / numOfNonZeroWeights;
			break;
		}
	}


    STORE_RESULT(*output);

    if(weightsBroad != weights)
    	delete weightsBroad;
    if(newLabels != labels)
    	delete newLabels;
	
    return Status::OK();
}


DECLARE_SHAPE_FN(sigm_cross_entropy_loss) {

	auto logitsShapeInfo  = inputShape->at(0);
    auto labelsShapeInfo  = inputShape->at(2);

	// labels and logits must have the same shapes 
    REQUIRE_TRUE(shape::shapeEquals(logitsShapeInfo, labelsShapeInfo), 0, "SIGM_CROSS_ENTROPY_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly!", ShapeUtils<T>::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils<T>::shapeAsString(logitsShapeInfo).c_str());    

    Nd4jLong* outShapeInfo = nullptr;
    if(INT_ARG(0) != 0) {			// in this case output is scalar
    	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2) /*rank=2*/, Nd4jLong);
    	outShapeInfo[0] = 2;
    	outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
    	outShapeInfo[5] = 0;
    	outShapeInfo[6] = 1;
    	outShapeInfo[7] = 99;
    }
    else {							// in this case output has the same shape as labels
    	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(labelsShapeInfo[0]), Nd4jLong);
    	outShapeInfo[0] = labelsShapeInfo[0];
    	for(int i = 1; i <= outShapeInfo[0]; ++i)
    		outShapeInfo[i] = labelsShapeInfo[i];
    	shape::updateStrides(outShapeInfo, shape::order(labelsShapeInfo));
    }
 
    return SHAPELIST(outShapeInfo);    

}

// INT_ARG(0) - reduction mode
}
}

#endif