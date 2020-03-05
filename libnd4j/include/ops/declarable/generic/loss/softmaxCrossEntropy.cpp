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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 25.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_softmax_cross_entropy_loss)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops  {


//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(softmax_cross_entropy_loss, 3, 1, false, 1, 1) {

  	auto logits  = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto labels  = INPUT_VARIABLE(2);
    auto output  = OUTPUT_VARIABLE(0);

    int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    double labelsSmoothing = T_ARG(0);

    // input validation
    REQUIRE_TRUE(labels->isSameShape(logits), 0, "SOFTMAX_CROSS_ENTROPY_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
	// only 4 possible reduction modes exist
    REQUIRE_TRUE(reductionMode==0 || reductionMode==1 || reductionMode==2 || reductionMode==3, 0, "SOFTMAX_CROSS_ENTROPY_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!", reductionMode);
	// smoothing is possible for rank of logits/labels > 1
    REQUIRE_TRUE(labels->rankOf() > 1 || (labels->rankOf() == 1 && labelsSmoothing == 0.), 0, "SOFTMAX_CROSS_ENTROPY_LOSS OP: smoothing is not possible when rank of labels/ logits = 1 !");

    if(!output->isScalar()) {
    	// weights array can be single scalar or has the same shape as output, and must be broadcastable to output shape
    	REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == output->rankOf(), 0, "SOFTMAX_CROSS_ENTROPY_LOSS OP: weights array should be scalar or have the same rank as output array, but got %i and %i correspondingly!", weights->rankOf(), output->rankOf());
    	// check whether broadcast operation is possible for weights array
	    REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *output), 0, "SOFTMAX_CROSS_ENTROPY_LOSS OP: shapes of weights and output arrays should be broadcastable, but got weights = %s and output = %s instead!", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
    }

	// If label_smoothing is nonzero, smooth the labels towards 1/num_classes: new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes
	// num_classes = labels->sizeAt(1)
	NDArray* cLabels = new NDArray(labels->cast(weights->dataType()));
	NDArray* newLabels = cLabels;
	if(labelsSmoothing != 0.) {
		newLabels = new NDArray(cLabels);
    	newLabels->assign((1.f - labelsSmoothing) * *cLabels + labelsSmoothing / cLabels->sizeAt(1));
	}

	// main formula: result = - sum_i(lables_i * log(softmax_i)) - sum over last dimension
	// softmax_i = exp(logits_i) / sum_j(exp(logits_j))
	// so result = sum_i( lables_i * (log(sum_j(exp(logits_j))) - logits_i) )
	// for numerical stability we use shifted logits (one can approve this using simple math):
	// softmax_i = exp(logits_i - maxLogit) / sum_j(exp(logits_j - maxLogit))
	// maxLogit is max among logits_i


	std::vector<int> dimensions = {-1};
	NDArray shiftedLogits = *logits - logits->reduceAlongDimension(reduce::Max, dimensions, true);
	NDArray logSumExp = shiftedLogits.transform(transform::Exp).reduceAlongDimension(reduce::Sum, dimensions, true).transform(transform::Log);
	NDArray E = (*newLabels * (logSumExp - shiftedLogits)).reduceAlongDimension(reduce::Sum, dimensions);

	// perform weights broadcasting/tile to E if it is necessary
	auto weightsBroad = weights;
	if(!weights->isScalar() && !weights->isSameShape(&E)) {
		if(E.rankOf() == 1 && weights->isVector() && weights->rankOf() > 1)
    		weightsBroad = new NDArray(weights->reshape(weights->ordering(), {weights->lengthOf()}));
    	else
			weightsBroad = new NDArray(weights->tileToShape(E.getShapeInfo()));
	}

    // multiply E on weights
    E *= *weightsBroad;

	switch (reductionMode) {
		case 0:												// 0 - "none", un-reduced weighted losses with the same shape as labels.
			output->assign(&E);
			break;

		case 1: {											// 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
			E.reduceNumber(reduce::Sum, *output);
			break;
		}
		case 2: {											// 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of all elements of weightsBroad array
			double sum;
			if (weights->isScalar())
				sum = weights->e<double>(0) * E.lengthOf();
			else
				sum = weightsBroad->reduceNumber(reduce::Sum).e<double>(0);

			if (sum == 0.)
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
				*output = 0.;
			else
				output->assign(E.reduceNumber(reduce::Sum) / double(numOfNonZeroWeights));

			break;
		}
	}

    if(weightsBroad != weights)
    	delete weightsBroad;

    if(newLabels != cLabels)
    	delete newLabels;

	delete cLabels;

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(softmax_cross_entropy_loss) {

	getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS})
					->setAllowedInputTypes(1, {ALL_FLOATS})
					->setAllowedInputTypes(2, {ALL_FLOATS, ALL_INTS})
					->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(softmax_cross_entropy_loss) {

	auto logitsShapeInfo  = inputShape->at(0);
	auto weightsShapeInfo = inputShape->at(1);
    auto labelsShapeInfo  = inputShape->at(2);

	// labels and logits must have the same shapes
    REQUIRE_TRUE(shape::shapeEquals(logitsShapeInfo, labelsShapeInfo), 0, "SOFTMAX_CROSS_ENTROPY_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly!", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());

	DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));
	Nd4jLong* outShapeInfo = nullptr;

    if(INT_ARG(0) != 0) 			// in this case output is scalar
    	outShapeInfo = ConstantShapeHelper::getInstance()->scalarShapeInfo(outType);
    else { 							// in this case output has the shape as labels and logits minus last dimension
    	std::vector<int> dimensions = {-1};
    	outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(logitsShapeInfo), dimensions, logitsShapeInfo, false, true, block.getWorkspace());

		// weights array can be single scalar or has the same rank as output, and must be broadcastable to output
    	REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(outShapeInfo), 0, "SOFTMAX_CROSS_ENTROPY_LOSS OP: weights array should be scalar or have the same rank as output array, but got %i and %i correspondingly!", shape::rank(weightsShapeInfo), shape::rank(outShapeInfo));
    	// check whether broadcast operation is possible for weights array
	    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, outShapeInfo), 0, "SOFTMAX_CROSS_ENTROPY_LOSS OP: shapes of weights and output arrays should be broadcastable, but got weights = %s and output = %s instead!", ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(outShapeInfo).c_str());
    }

    return SHAPELIST(outShapeInfo);
}









//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(softmax_cross_entropy_loss_grad, 3, 3, false, 1, 1) {

	auto logits  = INPUT_VARIABLE(0);
    auto weights = INPUT_VARIABLE(1);
    auto labels  = INPUT_VARIABLE(2);

    auto dLdp = OUTPUT_VARIABLE(0);		// dL/dlogits
    auto dLdw = OUTPUT_VARIABLE(1);		// dL/dweights
    auto dLdl = OUTPUT_VARIABLE(2);		// dL/dlabels

    auto labelsSmoothing = T_ARG(0);

    int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
    // take into account Alex's proposition to treat "none" the same as "weighted_sum" mode when calculating gradients
    if(reductionMode == 0)
    	reductionMode = 1;

    std::vector<int> dimensions = {-1};

    // input validation
    REQUIRE_TRUE(labels->isSameShape(logits), 0, "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
	// only 4 possible reduction modes exist
    REQUIRE_TRUE(reductionMode==0 || reductionMode==1 || reductionMode==2 || reductionMode==3, 0, "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!", reductionMode);
   	auto lossShapeInfo = ShapeUtils::evalReduceShapeInfo(logits->ordering(), dimensions, logits->getShapeInfo(), false, false, block.getWorkspace());
   	// weights array can be single scalar or has the same shape as loss, and must be broadcastable to loss shape
   	REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == shape::rank(lossShapeInfo), 0, "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: weights array should be scalar or have the same rank as loss array, but got %i and %i correspondingly!", weights->rankOf(), shape::rank(lossShapeInfo));
   	// check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(weights->getShapeInfo(), lossShapeInfo), 0, "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: shapes of weights and loss arrays should be broadcastable, but got weights = %s and loss = %s instead!", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(lossShapeInfo).c_str());
    // smoothing is possible for rank of logits/labels > 1
    REQUIRE_TRUE(labels->rankOf() > 1 || (labels->rankOf() == 1 && labelsSmoothing == 0.), 0, "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: smoothing is not possible when rank of labels/ logits = 1 !");

	// If label_smoothing is nonzero, smooth the labels towards 1/num_classes: new_onehot_labels = onehot_labels * (1 - label_smoothing) + label_smoothing / num_classes
	// num_classes = labels->sizeAt(1)
	NDArray* cLabels = new NDArray(labels->cast(weights->dataType()));
	NDArray* newLabels = cLabels;
	if(labelsSmoothing != 0.) {
		newLabels = new NDArray(labels->getShapeInfo(), dLdl->dataType(), false, block.launchContext());
    	newLabels->assign((1.f - labelsSmoothing) * *cLabels + labelsSmoothing / cLabels->sizeAt(1));
	}

	NDArray softmax = (*logits - logits->reduceAlongDimension(reduce::Max, dimensions, true)).transform(transform::Exp);
	softmax /= softmax.reduceAlongDimension(reduce::Sum, dimensions, true);

	// dEdp = softmax * sum_i(lables_i) - labels
	dLdp->assign(softmax * newLabels->reduceAlongDimension(reduce::Sum, dimensions, true) - *newLabels);

	// dEdl = -log(softmax)
	dLdl->assign(-softmax.transform(transform::Log)* (1.f - labelsSmoothing));

	NDArray shiftedLogits = *logits - logits->reduceAlongDimension(reduce::Max, dimensions, true);
    NDArray logSumExp = shiftedLogits.transform(transform::Exp).reduceAlongDimension(reduce::Sum, dimensions, true).transform(transform::Log);
    NDArray E = (*newLabels * (logSumExp - shiftedLogits)).reduceAlongDimension(reduce::Sum, dimensions);

	// perform weights broadcasting/tile to E if it is necessary
	auto weightsBroad = weights;
	if(!weights->isScalar() && !weights->isSameShape(&E))
			weightsBroad = new NDArray(weights->tileToShape(E.getShapeInfo()));

	dimensions = ShapeUtils::evalDimsToExclude(dLdp->rankOf(), dimensions);

	switch (reductionMode) {
		case 1: {											// 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

			if(weights->isScalar() || weights->lengthOf() == 1) {
				dLdw->assign(E.reduceNumber(reduce::Sum));
				*dLdp *= *weights;
				*dLdl *= *weights;
			}
			else {
				dLdp->applyBroadcast(sd::broadcast::Multiply, dimensions, *weightsBroad, *dLdp);
				dLdl->applyBroadcast(sd::broadcast::Multiply, dimensions, *weightsBroad, *dLdl);

				if(weights != weightsBroad) {
					std::vector<int> axesToReduceAlong = ShapeUtils::evalBroadcastBackwardAxis(weights->getShapeInfo(), weightsBroad->getShapeInfo());
					E.reduceAlongDimension(reduce::Sum, *dLdw, axesToReduceAlong, true, false, false);
				}
				else
					dLdw->assign(E);
			}

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

				if(weights->isScalar() || weights->lengthOf() == 1) {
					NDArray temp = *weights / sum;
					*dLdp *= temp;
					*dLdl *= temp;
					*dLdw = 0.;
				}
				else {

					NDArray temp = *weightsBroad / sum;
					dLdp->applyBroadcast(sd::broadcast::Multiply, dimensions, temp, *dLdp);
					dLdl->applyBroadcast(sd::broadcast::Multiply, dimensions, temp, *dLdl);

					if(weights != weightsBroad) {
						std::vector<int> axesToReduceAlong = ShapeUtils::evalBroadcastBackwardAxis(weights->getShapeInfo(), weightsBroad->getShapeInfo());
						((E * sum - (E * *weightsBroad).reduceNumber(reduce::Sum)) / (sum*sum)).reduceAlongDimension(reduce::Sum, *dLdw, axesToReduceAlong, true, false, false);
					}
					else
						dLdw->assign((E * sum - (E * *weightsBroad).reduceNumber(reduce::Sum)) / (sum*sum));
				}
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

				if(weights->isScalar() || weights->lengthOf() == 1) {
					NDArray temp = *weights / numOfNonZeroWeights;
					*dLdp *= temp;
					*dLdl *= temp;
					dLdw->assign(E.reduceNumber(reduce::Sum) / numOfNonZeroWeights);
				}
				else {
					NDArray temp = *weightsBroad / numOfNonZeroWeights;
					dLdp->applyBroadcast(sd::broadcast::Multiply, dimensions, temp, *dLdp);
					dLdl->applyBroadcast(sd::broadcast::Multiply, dimensions, temp, *dLdl);

					if(weights != weightsBroad) {
						std::vector<int> axesToReduceAlong = ShapeUtils::evalBroadcastBackwardAxis(weights->getShapeInfo(), weightsBroad->getShapeInfo());
						E.reduceAlongDimension(reduce::Sum, *dLdw, axesToReduceAlong, true, false, false);
						*dLdw /= numOfNonZeroWeights;
					}
					else
						dLdw->assign(E / numOfNonZeroWeights);
				}
			}
			break;
		}
	}

    if(weightsBroad != weights)
    	delete weightsBroad;

    if(newLabels != cLabels)
    	delete newLabels;

    delete cLabels;

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(softmax_cross_entropy_loss_grad) {

	getOpDescriptor()->setAllowedInputTypes(0, {ALL_FLOATS})
					 ->setAllowedInputTypes(1, {ALL_FLOATS})
					 ->setAllowedInputTypes(2, {ALL_FLOATS, ALL_INTS})
					 ->setAllowedInputTypes(3, {ALL_FLOATS})
					 ->setAllowedInputTypes(4, {ALL_FLOATS})
					 ->setAllowedInputTypes(5, {ALL_FLOATS})
					 ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(softmax_cross_entropy_loss_grad) {

	auto logitsShapeInfo  = inputShape->at(0);
	auto weightsShapeInfo = inputShape->at(1);
    auto labelsShapeInfo  = inputShape->at(2);

    std::vector<int> dimensions = {-1};

	// labels and logits must have the same shapes
    REQUIRE_TRUE(shape::shapeEquals(logitsShapeInfo, labelsShapeInfo), 0, "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: labels and logits arrays must have the same shapes, but got %s and %s correspondingly!", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());
	auto lossShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(logitsShapeInfo), dimensions, logitsShapeInfo, false, false, block.getWorkspace());
	// weights array can be single scalar or has the same rank as loss, and must be broadcastable to loss
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(lossShapeInfo), 0, "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: weights array should be scalar or have the same rank as loss array, but got %i and %i correspondingly!", shape::rank(weightsShapeInfo), shape::rank(lossShapeInfo));
    // check whether broadcast operation is possible for weights array
	REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, lossShapeInfo), 0, "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: shapes of weights and loss arrays should be broadcastable, but got weights = %s and loss = %s instead!", ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(lossShapeInfo).c_str());

    auto outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));

    auto dLdpShapeInfo = ConstantShapeHelper::getInstance()->createShapeInfo(ShapeDescriptor(outType, shape::order(logitsShapeInfo), shape::shapeOf(logitsShapeInfo), shape::rank(logitsShapeInfo)));
    auto dLdwShapeInfo = ConstantShapeHelper::getInstance()->createShapeInfo(ShapeDescriptor(outType, shape::order(weightsShapeInfo), shape::shapeOf(weightsShapeInfo), shape::rank(weightsShapeInfo)));
    auto dLdlShapeInfo = ConstantShapeHelper::getInstance()->createShapeInfo(ShapeDescriptor(outType, shape::order(labelsShapeInfo), shape::shapeOf(labelsShapeInfo), shape::rank(labelsShapeInfo)));

    return SHAPELIST(dLdpShapeInfo, dLdwShapeInfo, dLdlShapeInfo);
}


}
}

#endif