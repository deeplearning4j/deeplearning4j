#pragma clang diagnostic push
#pragma ide diagnostic ignored "cert-err58-cpp"

/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
    namespace ops {


//////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(mean_pairwssqerr_loss, 3, 1, false, 0, 1) {
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
            auto weights = INPUT_VARIABLE(1);
            auto labels = INPUT_VARIABLE(2);

            auto output = OUTPUT_VARIABLE(0);

            int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"


            // input validation
            REQUIRE_TRUE(labels->isSameShape(predictions), 0,
                         "MEAN_PAIRWSSQERR_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
                         ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
                      // only 4 possible reduction modes exist
            REQUIRE_TRUE(reductionMode==0 || reductionMode==1 || reductionMode==2 || reductionMode==3, 0, "MEAN_PAIRWSSQERR_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!", reductionMode);

            if (labels->rankOf() == 1) { // If labels and predictions are of rank 1, it means that all data entries are 0-tensor (scalar) so that the result of becomes always zero.
                *output = 0.;
                return Status::OK();
            }

            std::vector<int> reductionIdx = ShapeUtils::evalDimsToExclude(labels->rankOf(), {0});

            auto n = double(labels->sizeAt(1));
            auto diffs = *predictions - *labels;

            auto sumOfSquares = (diffs * diffs).reduceAlongDimension(reduce::Sum, reductionIdx, true);

            auto squareOfSum  = diffs.reduceAlongDimension(reduce::Sum, reductionIdx, true);
            squareOfSum.applyScalar(scalar::Pow, 2, squareOfSum);


            auto E = ((sumOfSquares * n) - squareOfSum) * (4/(n*(n-1)));

            // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
            REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == E.rankOf(), 0, "MEAN_PAIRWSSQERR_LOSS_GRAD OP: weights array should be scalar or have the same rank as results array, but got %i and %i correspondingly!", weights->rankOf(), E.rankOf());
            // check whether broadcast operation is possible for weights array
            REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, E), 0, "MEAN_PAIRWSSQERR_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and results = %s instead!", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(&E).c_str());

            // perform weights broadcasting/tile to labels if needed
            auto weightsBroad = weights;
            if(!weights->isScalar() && !weights->isSameShape(E))
                weightsBroad = new NDArray(weights->tileToShape(E.getShapeInfo()));

            E *= *weightsBroad;

            switch (reductionMode) {
                case 0:												// 0 - "none", un-reduced weighted losses with the same shape as labels.
                    output->assign(E);
                    break;

                case 1: {											// 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
                    E.reduceNumber(reduce::Sum, *output);
                    break;
                }
                case 2: {											// 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of all elements of weightsBroad array
                    NDArray sum;
                    if (weights->isScalar())
                        sum = (*weights) * E.lengthOf();
                    else
                        sum = weightsBroad->reduceNumber(reduce::Sum);

                    if (sum.e<double>(0) == 0.)
                        (*output) = 0.;
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

            if (weightsBroad != weights)
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
            auto weightsShapeInfo = inputShape->at(1);
            auto labelsShapeInfo = inputShape->at(2);

            REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
                         "MEAN_PAIRWSSQERR_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
                         ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
                         ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
            DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));
            Nd4jLong* outShapeInfo = nullptr;

            if(INT_ARG(0) != 0) 			// in this case output is scalar
                outShapeInfo = ConstantShapeHelper::getInstance()->scalarShapeInfo(outType);
            else { 							// in this case output has the shape as labels and logits minus last dimension
                std::vector<int> dimensions = {-1};
                outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(predictionsShapeInfo), dimensions, predictionsShapeInfo, false, true, block.getWorkspace());

                // weights array can be single scalar or has the same rank as output, and must be broadcastable to output
                REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(outShapeInfo), 0, "MEAN_PAIRWSSQERR_LOSS OP: weights array should be scalar or have the same rank as output array, but got %i and %i correspondingly!", shape::rank(weightsShapeInfo), shape::rank(outShapeInfo));
                // check whether broadcast operation is possible for weights array
                REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, outShapeInfo), 0, "MEAN_PAIRWSSQERR_LOSS OP: shapes of weights and output arrays should be broadcastable, but got weights = %s and output = %s instead!", ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(outShapeInfo).c_str());
            }

            return SHAPELIST(outShapeInfo);
        }


        //////////////////////////////////////////////////////////////////////////
        CUSTOM_OP_IMPL(mean_pairwssqerr_loss_grad, 3, 3, false, 0, 1) {

            auto predictions = INPUT_VARIABLE(0);
            auto weights 	 = INPUT_VARIABLE(1);
            auto labels  	 = INPUT_VARIABLE(2);

            auto dLdp = OUTPUT_VARIABLE(0);		// dL/dpredictions
            auto dLdw = OUTPUT_VARIABLE(1);		// dL/dweights
            auto dLdl = OUTPUT_VARIABLE(2);		// dL/dlabels


            int reductionMode = INT_ARG(0);			// 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
            // take into account Alex's proposition to treat "none" the same as "weighted_sum" mode when calculating gradients
            if(reductionMode == 0)
                reductionMode = 1;

            // inputs validation
            REQUIRE_TRUE(labels->isSameShape(predictions), 0, "MEAN_PAIRWSSQERR_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
            // only 4 possible reduction modes exist
            REQUIRE_TRUE(reductionMode==0 || reductionMode==1 || reductionMode==2 || reductionMode==3, 0, "MEAN_PAIRWSSQERR_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!", reductionMode);

            auto n = double(labels->sizeAt(1));
            auto diffs = *predictions - *labels;

            std::vector<int> reductionIdx = ShapeUtils::evalDimsToExclude(labels->rankOf(), {0});
            auto sumOfSquares = (diffs * diffs).reduceAlongDimension(reduce::Sum, reductionIdx, true);

            auto squareOfSum  = diffs.reduceAlongDimension(reduce::Sum, reductionIdx, true);
            squareOfSum.applyScalar(scalar::Pow, 2, squareOfSum);

            auto E = ((sumOfSquares * n) - squareOfSum) * (4/(n*(n-1)));

            auto sumPred = predictions->reduceAlongDimension(reduce::Sum, reductionIdx, true);
            auto sumLabel = labels->reduceAlongDimension(reduce::Sum, reductionIdx, true);

            dLdp->assign(((diffs * n) - sumPred + sumLabel)*(8/(n*(n-1))));


            // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
            REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == E.rankOf(), 0, "MEAN_PAIRWSSQERR_LOSS_GRAD OP: weights array should be scalar or have the same rank as results array, but got %i and %i correspondingly!", weights->rankOf(), E.rankOf());
            // check whether broadcast operation is possible for weights array
            REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, E), 0, "MEAN_PAIRWSSQERR_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and results = %s instead!", ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(&E).c_str());

            // perform weights broadcasting/tile to labels if needed
            auto weightsBroad = weights;
            if(!weights->isScalar() && !weights->isSameShape(E))
                weightsBroad = new NDArray(weights->tileToShape(E.getShapeInfo()));

            switch (reductionMode) {

                case 1: {											// 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

                    *dLdp *= *weightsBroad;

                    if(weights->isScalar())
                        dLdw->assign(E.reduceNumber(reduce::Sum));
                    else if(weights != weightsBroad) {
                        std::vector<int> axesToReduceAlong = ShapeUtils::evalBroadcastBackwardAxis(weights->getShapeInfo(), weightsBroad->getShapeInfo());
                        E.reduceAlongDimension(reduce::Sum, *dLdw, axesToReduceAlong, true, false, false);
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
                        *dLdw = 0.;
                    }
                    else {

                        *dLdp *= *weightsBroad / sum;

                        if(weights->isScalar())
                            *dLdw = 0.;
                        else if(weights != weightsBroad) {
                            std::vector<int> axesToReduceAlong = ShapeUtils::evalBroadcastBackwardAxis(weights->getShapeInfo(), weightsBroad->getShapeInfo());
                            ((E * sum - (E * *weightsBroad).reduceNumber(reduce::Sum)) / (sum*sum)).reduceAlongDimension(reduce::Sum, *dLdw, axesToReduceAlong, true, false, false);
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
                        *dLdw = 0.;
                    }
                    else {
                        auto numOfNonZeroWeightsScalar = NDArrayFactory::create(dLdw->dataType(), numOfNonZeroWeights, block.launchContext());

                        if(weights->isScalar())
                            dLdw->assign(E.reduceNumber(reduce::Sum) / double(numOfNonZeroWeights));
                        else if(weights != weightsBroad) {
                            std::vector<int> axesToReduceAlong = ShapeUtils::evalBroadcastBackwardAxis(weights->getShapeInfo(), weightsBroad->getShapeInfo());
                            E.reduceAlongDimension(reduce::Sum, *dLdw, axesToReduceAlong, true, false, false);
                            *dLdw /= numOfNonZeroWeightsScalar;
                        }
                        else
                            dLdw->assign(E / numOfNonZeroWeightsScalar);

                        NDArray temp = *weightsBroad / numOfNonZeroWeightsScalar;
                        *dLdp *= temp;
                    }
                    break;
                }
            }

            dLdl->assign(-*dLdp);

            if(weightsBroad != weights)
                delete weightsBroad;

            return Status::OK();
        }

        DECLARE_TYPES(mean_pairwssqerr_loss_grad) {
            getOpDescriptor()->setAllowedInputTypes(nd4j::DataType::ANY)->setAllowedOutputTypes({ALL_FLOATS});
        }

        DECLARE_SHAPE_FN(mean_pairwssqerr_loss_grad) {

            auto predictionsShapeInfo = inputShape->at(0);
            auto weightsShapeInfo 	  = inputShape->at(1);
            auto labelsShapeInfo  	  = inputShape->at(2);

            // labels and predictions must have the same shapes
            REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0, "MEAN_PAIRWSSQERR_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !", ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
            // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
            REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0, "MEAN_PAIRWSSQERR_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i and %i correspondingly!", shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
            // check whether broadcast operation is possible for weights array
            REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0, "MEAN_PAIRWSSQERR_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s instead!", ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

            DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));

            Nd4jLong *dLdpShapeInfo = ShapeBuilders::copyShapeInfoAndType(predictionsShapeInfo, outType, false, block.getWorkspace());
            Nd4jLong *dLdwShapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, outType, false, block.getWorkspace());
            Nd4jLong *dLdlShapeInfo = ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());

            return SHAPELIST(dLdpShapeInfo, dLdwShapeInfo, dLdlShapeInfo);
        }
    }
}

#endif
#pragma clang diagnostic pop
