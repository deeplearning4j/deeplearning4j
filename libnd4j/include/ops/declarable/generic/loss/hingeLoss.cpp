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
#if NOT_EXCLUDED(OP_hinge_loss)

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(hinge_loss, 3, 1, false, 0, 1) {
  	auto logits  = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto labels  = INPUT_VARIABLE(2);
    auto output  = OUTPUT_VARIABLE(0);

    // input validation
    // labels and logits must have the same shapes 
    REQUIRE_TRUE(labels->isSameShape(logits), 0, "HINGE_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    REQUIRE_TRUE(!(!weights->isScalar() && weights->rankOf() != labels->rankOf()), 0, "HINGE_LOSS OP: weights array must have the same rank as labels array, but got %i and %i correspondingly!", weights->rankOf(), labels->rankOf());    
    // check whether broadcast operation is possible for weights array
    if(!weights->isScalar())
    	for (int i = 0; i < weights->rankOf(); ++i)
        	REQUIRE_TRUE(!(weights->shapeOf()[i] != labels->shapeOf()[i] && weights->shapeOf()[i] != 1), 0, "HINGE_LOSS OP: shape of weights array %s is not broadcastable to labels array shape %s !", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
    
    int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    
	// perform weights broadcasting/tile to labels if needed	
	auto weightsBroad = weights;
	if(!weights->isScalar() && !weights->isSameShape(logits)) {
		// evaluate repeat dimensions for tile operation
		std::vector<Nd4jLong> reps;
		for(int i = 0; i < labels->rankOf(); ++i)
			reps.emplace_back(labels->shapeOf()[i] / weights->shapeOf()[i]);
		weightsBroad = new NDArray(weights->tile(reps));
	}

	auto ts = NDArrayFactory::create<float>(1.0f);

	 // We first need to convert binary labels to -1/1 labels (as floats)
	NDArray weightedLosses = ts - ((*labels)* 2.f - 1.f)*(*logits);
    weightedLosses.applyTransform(transform::RELU, &weightedLosses, nullptr);

    // multiply weightedLosses on weights
    weightedLosses *= (*weights);
 	
  	// regard 4 possible reduction modes below
  	REQUIRE_TRUE(reductionMode==0 || reductionMode==1 || reductionMode==2 || reductionMode==3, 0, "HINGE_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!", reductionMode);
	switch (reductionMode) {
		case 0:												// 0 - "none", un-reduced weighted losses with the same shape as labels.
			output->assign(&weightedLosses);
			break;
		
		case 1: {											// 1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
			output->assign(weightedLosses.reduceNumber(reduce::Sum));
			break;
		}
		case 2: {											// 2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
			double sum;
			if (weights->isScalar())
				sum = weights->e<double>(0) * weightedLosses.lengthOf();
			else 
				sum = weightsBroad->reduceNumber(reduce::Sum).e<double>(0);
			
			if (sum == 0.)
				(*output) = 0.;
			else 
				(*output) = weightedLosses.reduceNumber(reduce::Sum) / sum;
			break;
		}
		case 3: {											// 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
			Nd4jLong numOfNonZeroWeights = 0;
			if(weights->isScalar()) {
				if (weights->e<double>(0) != 0.)
					numOfNonZeroWeights = weightedLosses.lengthOf();
			}
			else {
				numOfNonZeroWeights = weightsBroad->reduceNumber(reduce::CountNonZero).e<Nd4jLong>(0);
			}

			if (numOfNonZeroWeights == 0)
				(*output) = 0.;
			else 
				(*output) = weightedLosses.reduceNumber(reduce::Sum) / numOfNonZeroWeights;
			break;
		}		
			
	}


    STORE_RESULT(*output);

    if(weightsBroad != weights)
    	delete weightsBroad;
	
    return Status::OK();
}


DECLARE_SHAPE_FN(hinge_loss) {
	
	auto logitsShapeInfo  = inputShape->at(0);    
    auto labelsShapeInfo  = inputShape->at(2);

    // labels and logits must have the same shapes 
    REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, logitsShapeInfo), 0, "HINGE_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());

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