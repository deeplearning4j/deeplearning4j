//
// Created by Yurii Shyrma on 22.11.2017.
//

#include <ops/declarable/CustomOperations.h>
#include <helpers/ShapeUtils.h>

namespace nd4j {
    namespace ops {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(cosineDistance, 3, 1, false, 0, 2) {

  	NDArray<T>* predictions = INPUT_VARIABLE(0);
    NDArray<T>* weights 	= INPUT_VARIABLE(1);
    NDArray<T>* labels     	= INPUT_VARIABLE(2);
    NDArray<T>* output      = OUTPUT_VARIABLE(0);

    int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    int dim = INT_ARG(1);					// axis, dimension should be reduced to unity along this axis
    if(dim < 0)
    	dim += labels->rankOf();

    
	// perform weights broadcasting/tile to output if needed	
	NDArray<T>* weightsBroad = weights;	
	if(!weights->isScalar() && !weights->isSameShape(output)) {
		// evaluate repeat dimensions for tile operation
		std::vector<int> reps;
		for(int i = 0; i < output->rankOf(); ++i)
			reps.emplace_back(labels->shapeOf()[i] / weights->shapeOf()[i]);
		weightsBroad = new NDArray<T>(weights->tile(reps));
	}
		
	NDArray<T> weightedLosses = (T)1. - ((*predictions) * (*labels)).template reduceAlongDims<simdOps::Sum<T>>({dim}, true);

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
			throw "CUSTOM_OP loss function cosineDistance: reduction mode has not acceptable value, possible values are 0, 1, 2, 3 !";			
	}


    STORE_RESULT(*output);

    if(weightsBroad != weights)
    	delete weightsBroad;
	
    return ND4J_STATUS_OK;
}


DECLARE_SHAPE_FN(cosineDistance) {

	// labels and predictions must have the same shapes 
	NDArray<T>* predictions = INPUT_VARIABLE(0);
    NDArray<T>* weights 	= INPUT_VARIABLE(1);
    NDArray<T>* labels      = INPUT_VARIABLE(2);   

    int dim = INT_ARG(1);
    if(dim < 0)
    	dim += labels->rankOf();

    if(!labels->isSameShape(predictions))
    	throw "CUSTOM_OP loss function cosineDistance: labels and predictions arrays have different shapes!";
    // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
    if(!weights->isScalar() && weights->rankOf() != labels->rankOf())
    	throw "CUSTOM_OP loss function cosineDistance: weights array must have the same rank as labels array!";
    // input dimension can't be larger than labels/predictions/weights rank
    if(dim >= labels->rankOf())
    	throw "CUSTOM_OP loss function cosineDistance: input reduction dimension can't be larger than labels rank!";

    // check whether broadcast operation is possible for weights array
    if(!weights->isScalar())
    	for (int i = 0; i < weights->rankOf(); ++i)    		           	
            if( (i != dim && weights->shapeOf()[i] != labels->shapeOf()[i] && weights->shapeOf()[i] != 1) || (i == dim && weights->shapeOf()[i] != 1))
            	throw "CUSTOM_OP loss function cosineDistance: shapes of weights array is not broadcastable to losses shape!";
 
 	// evaluate output shapeInfo
    int* outShapeInfo = nullptr;
    if(INT_ARG(0) != 0) {			// in this case output is scalar
    	ALLOCATE(outShapeInfo, block.getWorkspace(), shape::shapeInfoLength(2) /*rank=2*/, int);
    	outShapeInfo[0] = 2;
    	outShapeInfo[1] = outShapeInfo[2] = outShapeInfo[3] = outShapeInfo[4] = 1;
    	outShapeInfo[5] = 0;
    	outShapeInfo[6] = 1;
    	outShapeInfo[7] = 99;
    }
    else {							// in this case output has the same shape as labels reduced  by dim axis    	
    	std::vector<int> dimensions = {dim};
    	outShapeInfo = ShapeUtils<T>::evalReduceShapeInfo(labels->ordering(), dimensions, labels->getShapeInfo(), true, block.getWorkspace());
    }
    
    return new ShapeList(outShapeInfo);    

}


// INT_ARG(0) - reduction mode
// INT_ARG(1) - axis, dimension should be reduced to unity along this axis










}
}