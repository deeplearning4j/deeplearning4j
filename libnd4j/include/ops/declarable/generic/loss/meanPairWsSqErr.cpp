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
// @author Paul Dubs
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
    /*
     * Implementation of mean pairwise squared error loss
     *
     * For context on where this loss function may be useful see:
     *
     * Wei, Z., Zhang, J., Shen, X., Lin, Z., Mech, R., Hoai, M. and Samaras, D., 2018.
     * Good view hunting: learning photo composition from dense view pairs. In Proceedings of the IEEE Conference on
     * Computer Vision and Pattern Recognition (pp. 5437-5446).
     *
     * The paper defines the loss function as:
     *
     * L(y,q) = 1/((n*(n-1))/2) * (sum_(i,j=1..n,i!=j)((y_i - y_j) - (q_i - q_j))^2)
     *
     * with y: predictions, q: labels, n: length of y and q
     *
     * As creating those pairs is computationally expensive, we implement a mathematically equivalent function:
     *
     * L(y,q) = 4/(n*(n-1)) * (n * sum (y_i - q_i)^2 - (sum y_i - q_i)^2)
     *
     * This equivalency can be derived as:
     *
     * sum_(i,j=1..n,i!=j)((y_i - y_j) - (q_i - q_j))^2 = sum_(i,j=1..n,i!=j)((y_i - q_i) - (y_j - q_j))^2
     *
     * To simplify the following equations we use
     *
     * sum_(i,j=1..n,i!=j)(d_i - d_j)^2 = sum_(i,j=1..n,i!=j)(d_i^2 + d_j^2 - 2*d_i*d_j)
     *
     * Due to the pairings each element will appear as both d_i and d_j exactly n-1 times. This allows us to split the sum:
     *
     * sum_(i,j=1..n,i!=j)(d_i^2 + d_j^2 - 2*d_i*d_j) = 2*(n-1)*sum d_i^2 - 2 * sum_(i,j=1..n,i!=j) d_i * d_j
     *                                                = 2*((n-1) * sum d_i^2 - sum_(i,j=1..n,i!=j) d_i * d_j)
     *
     * Now we use the following equivalency:
     *
     * (sum d_i)^2 = sum d_i^2 + sum_(i,j=1..n,i!=j) d_i * d_j
     *
     * This allows us to now use sum d_i^2 and (sum d_i)^2 as a quick way to calculate the sum:
     *
     * (n-1) * sum d_i^2 - sum_(i,j=1..n,i!=j) d_i * d_j = n * sum d_i^2 - (sum d_i)^2
     *
     * And by substituting it into the original definition we get:
     *
     * 1/((n*(n-1))/2) * 2*(n * sum d_i^2 - (sum d_i)^2)
     *
     * Which can be again simplified to
     *
     * 4/(n*(n-1)) * (n * sum d_i^2 - (sum d_i)^2)
     *
     * After substituting d_i back to (y_i - q_i) this results in the function that we actually implement.
     *
     */
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

	// TODO: Fill in correct calculations
    
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