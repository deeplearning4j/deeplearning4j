//
// Created by Yurii Shyrma on 23.11.2017.
//

#include <ops/declarable/CustomOperations.h>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(hingeLoss, 3, 1, false, 0, 1) {

  	NDArray<T>* logits  = INPUT_VARIABLE(0);
    NDArray<T>* weights = INPUT_VARIABLE(1);
    NDArray<T>* labels  = INPUT_VARIABLE(2);
    NDArray<T>* output  = OUTPUT_VARIABLE(0);

    int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    
	// perform weights broadcasting/tile to labels if needed	
	NDArray<T>* weightsBroad = weights;	
	if(!weights->isScalar() && !weights->isSameShape(logits)) {
		// evaluate repeat dimensions for tile operation
		std::vector<int> reps;
		for(int i = 0; i < labels->rankOf(); ++i)
			reps.emplace_back(labels->shapeOf()[i] / weights->shapeOf()[i]);
		weightsBroad = new NDArray<T>(weights->tile(reps));
	}

	 // We first need to convert binary labels to -1/1 labels (as floats)
	NDArray<T> weightedLosses = (T)1. - ((*labels)*(T)2. - (T)1.)*(*logits);
	auto relu = LAMBDA_T(value) { return value > (T)0. ? value : (T)0.; };
    weightedLosses.applyLambda(relu);
    // multiply weightedLosses on weights
 	if(weights->isScalar())
 		weightedLosses *= (*weights)(0);
 	else
 		weightedLosses *= (*weights); 	
 	// regard 4 possible reduction modes below
	switch (reductionMode) {
		case 0:												// 0 - "none", un-reduced weighted losses with the same shape as labels.
			output->assign(&weightedLosses);
			break;
		
		case 1: {											// 1 - "weighted_sum", output is scalar and equal to sum of all elements of weightedLosses array
			(*output)(0) = weightedLosses.template reduceNumber<simdOps::Sum<T>>();
			break;
		}
		case 2: {											// 2 - "weighted_mean", output is scalar and equal to sum of all elements of weightedLosses array divided by sum of all elements of weightsBroad array
			T sum;
			if (weights->isScalar())
				sum = (*weights)(0) * weightedLosses.lengthOf();
			else 
				sum = weightsBroad->template reduceNumber<simdOps::Sum<T>>();
			
			if (sum == (T)0.)
				(*output)(0) = (T)0.;
			else 
				(*output)(0) = weightedLosses.template reduceNumber<simdOps::Sum<T>>() / sum;
			break;
		}
		case 3: {											// 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of weightedLosses array divided by number of non-zero weights
			int numOfNonZeroWeights = 0;
			if(weights->isScalar()) {
				if((*weights)(0) != (T)0.)
					numOfNonZeroWeights = weightedLosses.lengthOf();
			}
			else {
				for(int i = 0; i < weightsBroad->lengthOf(); ++i)
					if((*weightsBroad)(i) != (T)0.)
						++numOfNonZeroWeights;
			}

			if (numOfNonZeroWeights == 0)
				(*output)(0) = (T)0.;
			else 
				(*output)(0) = weightedLosses.template reduceNumber<simdOps::Sum<T>>() / numOfNonZeroWeights;
			break;
		}
		default:
			throw "CUSTOM_OP loss function hingeLoss: reduction mode has not acceptable value, possible values are 0, 1, 2, 3 !";			
	}


    STORE_RESULT(*output);

    if(weightsBroad != weights)
    	delete weightsBroad;
	
    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(hingeLoss) {

	// labels and logits must have the same shapes 
	NDArray<T>* logits  = INPUT_VARIABLE(0);
    NDArray<T>* weights = INPUT_VARIABLE(1);
    NDArray<T>* labels  = INPUT_VARIABLE(2);

    if(!labels->isSameShape(logits))
    	throw "CUSTOM_OP loss function hingeLoss: labels and logits arrays have different shapes!";
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    if(!weights->isScalar() && weights->rankOf() != labels->rankOf())
    	throw "CUSTOM_OP loss function hingeLoss: weights array must have the same rank as labels array!";
    // check whether broadcast operation is possible for weights array
    if(!weights->isScalar())
    	for (int i = 0; i < weights->rankOf(); ++i)
        	if (weights->shapeOf()[i] != labels->shapeOf()[i] && weights->shapeOf()[i] != 1)
            	throw "CUSTOM_OP loss function hingeLoss: shapes of weights array is not broadcastable to labels shape!";

    int* outShapeInfo = nullptr;
    if(INT_ARG(0) != 0) {			// in this case output is scalar
    	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2) /*rank=2*/, int);
    	outShapeInfo[0] = 2;
    	outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
    	outShapeInfo[5] = 0;
    	outShapeInfo[6] = 1;
    	outShapeInfo[7] = 99;
    }
    else {							// in this case output has the same shape as labels
    	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(labels->rankOf()), int);
    	outShapeInfo[0] = labels->rankOf();
    	for(int i = 1; i <= outShapeInfo[0]; ++i)
    		outShapeInfo[i] = labels->shapeOf()[i-1];
    	shape::updateStrides(outShapeInfo, labels->ordering());    
    }
 
    return new ShapeList(outShapeInfo);    

}

// INT_ARG(0) - reduction mode











}
}