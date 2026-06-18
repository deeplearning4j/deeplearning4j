/* ******************************************************************************
*
*
* This program and the accompanying materials are made available under the
* terms of the Apache License, Version 2.0 which is available at
* https://www.apache.org/licenses/LICENSE-2.0.
*
*  See the NOTICE file distributed with this work for additional
*  information regarding copyright ownership.
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
* WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
* License for the specific language governing permissions and limitations
* under the License.
*
* SPDX-License-Identifier: Apache-2.0
******************************************************************************/

//
//  @author raver119@gmail.com
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_log_poisson_loss)

#include <array/NDArrayFactory.h>
#include <ops/declarable/headers/loss.h>

namespace sd {
namespace ops {
CUSTOM_OP_IMPL(log_poisson_loss, 3, 1, true, 0, 1) {
 auto log_predictions = INPUT_VARIABLE(0);
 auto weights = INPUT_VARIABLE(1);
 auto labels = INPUT_VARIABLE(2);

 auto output = OUTPUT_VARIABLE(0);

 int reductionMode =
     INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"

 bool computeFullLoss = false;
 if (block.numI() > 1) computeFullLoss = INT_ARG(1) != 0;

 // inputs validation
 REQUIRE_TRUE(labels->isSameShape(log_predictions), 0,
              "LOG_POISSON_LOSS OP: labels and log_predictions arrays must have the same shapes, but got %s and %s "
              "correspondingly !",
              ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(log_predictions).c_str());
 // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
 REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
              "LOG_POISSON_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i "
              "and %i correspondingly!",
              weights->rankOf(), labels->rankOf());
 // check whether broadcast operation is possible for weights array
 REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
              "LOG_POISSON_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s "
              "and labels = %s instead!",
              ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
 // only 4 possible reduction modes exist
 REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
              "LOG_POISSON_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got "
              "%i instead!",
              reductionMode);

 // perform weights broadcasting/tile to labels if needed
 auto weightsBroad = weights;
 if (!weights->isScalar() && !weights->isSameShape(log_predictions))
   weightsBroad = new NDArray(weights->tileToShape(log_predictions->shapeInfo()));

 NDArray E(labels->shapeInfo(), block.getWorkspace());
 if (computeFullLoss)
   labels->applyPairwiseTransform(pairwise::LogPoissonLossFull, log_predictions, &E);
 else
   labels->applyPairwiseTransform(pairwise::LogPoissonLoss, log_predictions, &E);

 // multiply E on weights
 NDArray EWeighted(E.shapeInfo(), false, block.launchContext());
 E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &EWeighted, false);
 E.assign(&EWeighted);

 switch (reductionMode) {
   case 0: {  // 0 - "none", un-reduced weighted losses with the same shape as labels.
     output->assign(&E);
     break;
   }
   case 1: {  // 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
     E.reduceNumber(reduce::Sum, output);
     break;
   }
   case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
     // all elements of weightsBroad array
     NDArray sum(output->dataType(), block.launchContext());
     if (weights->isScalar()) {
       double wVal = weights->e<double>(0);
       double lenVal = static_cast<double>(E.lengthOf());
       double sumVal = wVal * lenVal;
       sum.assign(sumVal);
     } else {
       weightsBroad->reduceNumber(reduce::Sum, &sum);
     }

     if (sum.e<double>(0) == 0.) {
       double zero = 0.0;
       output->assign(zero);
     } else {
       NDArray sumE(output->dataType(), block.launchContext());
       E.reduceNumber(reduce::Sum, &sumE);
       sumE.applyPairwiseTransform(pairwise::Divide, &sum, output);
     }
     break;
   }
   case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
     // array divided by number of non-zero weights
     LongType numOfNonZeroWeights = 0;
     if (weights->isScalar()) {
       if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E.lengthOf();
     } else {
       NDArray countResult(DataType::INT64, block.launchContext());
       weightsBroad->reduceNumber(reduce::CountNonZero, &countResult);
       numOfNonZeroWeights = countResult.e<LongType>(0);
     }

     if (numOfNonZeroWeights == 0) {
       double zero = 0.0;
       output->assign(zero);
     } else {
       NDArray sumE(output->dataType(), block.launchContext());
       E.reduceNumber(reduce::Sum, &sumE);
       sumE.applyScalar(scalar::Divide, static_cast<double>(numOfNonZeroWeights), output);
     }
     break;
   }
 }

 if (weightsBroad != weights) delete weightsBroad;

 return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(log_poisson_loss) {
 getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(log_poisson_loss) {
 auto predictionsShapeInfo = inputShape->at(0);
 auto weightsShapeInfo = inputShape->at(1);
 auto labelsShapeInfo = inputShape->at(2);

 // labels and predictions must have the same shapes
 REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
              "LOG_POISSON_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s "
              "correspondingly !",
              ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
              ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
 // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
 REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
              "LOG_POISSON_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i "
              "and %i correspondingly!",
              shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
 // check whether broadcast operation is possible for weights array
 REQUIRE_TRUE(
     shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
     "LOG_POISSON_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and "
     "labels = %s instead!",
     ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

 DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));
 LongType* outShapeInfo = nullptr;

 if (INT_ARG(0) != 0)  // in this case output is scalar
   outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
 else {  // in this case output has the same shape as labels and predictions
   outShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(outType, shape::order(labelsShapeInfo),
                                                                       shape::rank(labelsShapeInfo),
                                                                       shape::shapeOf(labelsShapeInfo))->primary();
 }
 return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(log_poisson_loss_grad, 3, 3, false, 0, 1) {
 auto log_predictions = INPUT_VARIABLE(0);
 auto weights = INPUT_VARIABLE(1);
 auto labels = INPUT_VARIABLE(2);

 auto dLdp = OUTPUT_VARIABLE(0);  // dL/dpredictions
 auto dLdw = OUTPUT_VARIABLE(1);  // dL/dweights
 auto dLdl = OUTPUT_VARIABLE(2);  // dL/dlabels

 int reductionMode =
     INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
 // take into account Alex's proposition to treat "none" the same as "weighted_sum" mode when calculating gradients
 if (reductionMode == 0) reductionMode = 1;

 bool computeFullLoss = false;
 if (block.numI() > 1) computeFullLoss = INT_ARG(1) != 0;

 // inputs validation
 REQUIRE_TRUE(labels->isSameShape(log_predictions), 0,
              "LOG_POISSON_LOSS_GRAD OP: labels and log_predictions arrays must have the same shapes, but got %s and "
              "%s correspondingly !",
              ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(log_predictions).c_str());
 // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
 REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
              "LOG_POISSON_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but "
              "got %i and %i correspondingly!",
              weights->rankOf(), labels->rankOf());
 // check whether broadcast operation is possible for weights array
 REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
              "LOG_POISSON_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights "
              "= %s and labels = %s instead!",
              ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
 // only 4 possible reduction modes exist
 REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
              "LOG_POISSON_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but "
              "got %i instead!",
              reductionMode);

 // perform weights broadcasting/tile to labels if needed
 auto weightsBroad = weights;
 if (!weights->isScalar() && !weights->isSameShape(log_predictions))
   weightsBroad = new NDArray(weights->tileToShape(log_predictions->shapeInfo()));

 NDArray E(labels->shapeInfo(), block.getWorkspace());
 if (computeFullLoss) {
   labels->applyPairwiseTransform(pairwise::LogPoissonLossFull, log_predictions, &E);

   NDArray rDiv(labels->shapeInfo(), block.getWorkspace());
   labels->applyScalar(scalar::ReverseDivide, 0.5f, &rDiv);

   // For dLdl: dLdl = rDiv + log(labels) - log_predictions
   // i.e. 0.5/labels + log(labels) - log_predictions
   NDArray logLabels(labels->shapeInfo(), block.getWorkspace());
   labels->applyTransform(transform::Log, &logLabels);

   NDArray rDivPlusLogLabels(labels->shapeInfo(), block.getWorkspace());
   rDiv.applyPairwiseTransform(pairwise::Add, &logLabels, &rDivPlusLogLabels);

   // subtract log_predictions: dLdl = rDivPlusLogLabels - log_predictions
   rDivPlusLogLabels.applyPairwiseTransform(pairwise::Subtract, log_predictions, dLdl);
 } else {
   labels->applyPairwiseTransform(pairwise::LogPoissonLoss, log_predictions, &E);

   // For dLdl - second case: dLdl = -log_predictions
   log_predictions->applyTransform(transform::Neg, dLdl);
 }

 // For dLdp: dLdp = exp(log_predictions) - labels
 NDArray expLogPred(labels->shapeInfo(), block.getWorkspace());
 log_predictions->applyTransform(transform::Exp, &expLogPred);
 expLogPred.applyPairwiseTransform(pairwise::Subtract, labels, dLdp);

 switch (reductionMode) {
   case 1: {  // 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

     {
       NDArray dLdpW(dLdp->shapeInfo(), false, block.launchContext());
       dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdpW, false);
       dLdp->assign(&dLdpW);
       NDArray dLdlW(dLdl->shapeInfo(), false, block.launchContext());
       dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &dLdlW, false);
       dLdl->assign(&dLdlW);
     }

     if (weights->isScalar()) {
       NDArray sumE(dLdw->dataType(), block.launchContext());
       E.reduceNumber(reduce::Sum, &sumE);
       dLdw->assign(&sumE);
     } else if (weights != weightsBroad) {
       std::vector<LongType> axesToReduceAlong =
           ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
       E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
     } else
       dLdw->assign(&E);
     break;
   }
   case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
     // all elements of weightsBroad array

     NDArray sum(dLdw->dataType(), block.launchContext());
     if (weights->isScalar()) {
       double wVal = weights->e<double>(0);
       double lenVal = static_cast<double>(E.lengthOf());
       double sumVal = wVal * lenVal;
       sum.assign(sumVal);
     } else {
       weightsBroad->reduceNumber(reduce::Sum, &sum);
     }

     if (sum.e<double>(0) == 0.) {
       double zero = 0.0;
       dLdp->assign(zero);
       dLdl->assign(zero);
       dLdw->assign(zero);
     } else {
       // dLdp *= weightsBroad / sum; dLdl *= weightsBroad / sum
       NDArray weightsBroadDivSum(weightsBroad->shapeInfo(), false, block.launchContext());
       weightsBroad->applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sum, &weightsBroadDivSum, false);
       NDArray dLdpR(dLdp->shapeInfo(), false, block.launchContext());
       dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsBroadDivSum, &dLdpR, false);
       dLdp->assign(&dLdpR);
       NDArray dLdlR(dLdl->shapeInfo(), false, block.launchContext());
       dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &weightsBroadDivSum, &dLdlR, false);
       dLdl->assign(&dLdlR);

       if (weights->isScalar()) {
         double zero = 0.0;
         dLdw->assign(zero);
       } else if (weights != weightsBroad) {
         std::vector<LongType> axesToReduceAlong =
             ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());

         // numerator = E * sum - E * weightsBroad_sum_reduced
         // = E * sum - (sum of E*weightsBroad)
         // dLdw = reduce(numerator / sum^2, axes)
         NDArray eMulWeightsBroad(labels->shapeInfo(), false, block.launchContext());
         E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &eMulWeightsBroad, false);
         NDArray sumReduced(dLdw->dataType(), block.launchContext());
         eMulWeightsBroad.reduceNumber(reduce::Sum, &sumReduced);

         NDArray eMulSum(labels->shapeInfo(), false, block.launchContext());
         E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &sum, &eMulSum, false);

         NDArray numerator(labels->shapeInfo(), false, block.launchContext());
         eMulSum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &sumReduced, &numerator, false);

         NDArray sumSquared(dLdw->dataType(), block.launchContext());
         sum.applyPairwiseTransform(pairwise::Multiply, &sum, &sumSquared);

         NDArray result(labels->shapeInfo(), false, block.launchContext());
         numerator.applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sumSquared, &result, false);

         result.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
       } else {
         NDArray eMulWeightsBroad(labels->shapeInfo(), false, block.launchContext());
         E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), weightsBroad, &eMulWeightsBroad, false);
         NDArray sumReduced(dLdw->dataType(), block.launchContext());
         eMulWeightsBroad.reduceNumber(reduce::Sum, &sumReduced);

         NDArray eMulSum(labels->shapeInfo(), false, block.launchContext());
         E.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &sum, &eMulSum, false);

         NDArray numerator(labels->shapeInfo(), false, block.launchContext());
         eMulSum.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &sumReduced, &numerator, false);

         NDArray sumSquared(dLdw->dataType(), block.launchContext());
         sum.applyPairwiseTransform(pairwise::Multiply, &sum, &sumSquared);

         NDArray result(labels->shapeInfo(), false, block.launchContext());
         numerator.applyTrueBroadcast(BroadcastOpsTuple::Divide(), &sumSquared, &result, false);

         dLdw->assign(&result);
       }
     }
     break;
   }
   case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
     // array divided by number of non-zero weights

     LongType numOfNonZeroWeights = 0;
     if (weights->isScalar()) {
       if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E.lengthOf();
     } else {
       NDArray countResult(DataType::INT64, block.launchContext());
       weightsBroad->reduceNumber(reduce::CountNonZero, &countResult);
       numOfNonZeroWeights = countResult.e<LongType>(0);
     }

     if (numOfNonZeroWeights == 0) {
       double zero = 0.0;
       dLdp->assign(zero);
       dLdl->assign(zero);
       dLdw->assign(zero);
     } else {
       NDArray numOfNonZeroWeightsScalar(dLdw->dataType(), block.launchContext());
       double numNonZeroD = static_cast<double>(numOfNonZeroWeights);
       numOfNonZeroWeightsScalar.assign(numNonZeroD);

       if (weights->isScalar()) {
         NDArray sumE(dLdw->dataType(), block.launchContext());
         E.reduceNumber(reduce::Sum, &sumE);
         // sumE and numOfNonZeroWeightsScalar are both scalars
         sumE.applyPairwiseTransform(pairwise::Divide, &numOfNonZeroWeightsScalar, dLdw);
       } else if (weights != weightsBroad) {
         std::vector<LongType> axesToReduceAlong =
             ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
         E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
         NDArray dLdwResult(dLdw->shapeInfo(), false, block.launchContext());
         // dLdw and numOfNonZeroWeightsScalar — dLdw may be reduced shape, numOfNZW is scalar
         dLdw->applyTrueBroadcast(BroadcastOpsTuple::Divide(), &numOfNonZeroWeightsScalar, &dLdwResult, false);
         dLdw->assign(&dLdwResult);
       } else {
         // E and dLdw are both labels-shaped, numOfNonZeroWeightsScalar is scalar
         NDArray dLdwResult(dLdw->shapeInfo(), false, block.launchContext());
         E.applyTrueBroadcast(BroadcastOpsTuple::Divide(), &numOfNonZeroWeightsScalar, &dLdwResult, false);
         dLdw->assign(&dLdwResult);
       }

       NDArray temp(weightsBroad->shapeInfo(), false, block.launchContext());
       weightsBroad->applyTrueBroadcast(BroadcastOpsTuple::Divide(), &numOfNonZeroWeightsScalar, &temp, false);
       NDArray dLdpResult(dLdp->shapeInfo(), false, block.launchContext());
       dLdp->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdpResult, false);
       dLdp->assign(&dLdpResult);
       NDArray dLdlResult(dLdl->shapeInfo(), false, block.launchContext());
       dLdl->applyTrueBroadcast(BroadcastOpsTuple::Multiply(), &temp, &dLdlResult, false);
       dLdl->assign(&dLdlResult);
     }
     break;
   }
 }

 if (weightsBroad != weights) delete weightsBroad;

 return Status::OK;
}

DECLARE_TYPES(log_poisson_loss_grad) {
 getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

DECLARE_SHAPE_FN(log_poisson_loss_grad) {
 auto predictionsShapeInfo = inputShape->at(0);
 auto weightsShapeInfo = inputShape->at(1);
 auto labelsShapeInfo = inputShape->at(2);

 // labels and predictions must have the same shapes
 REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
              "LOG_POISSON_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s "
              "correspondingly !",
              ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
              ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
 // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
 REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
              "LOG_POISSON_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but "
              "got %i and %i correspondingly!",
              shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
 // check whether broadcast operation is possible for weights array
 REQUIRE_TRUE(
     shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
     "LOG_POISSON_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and "
     "labels = %s instead!",
     ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

 DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));

 auto dLdpShapeInfo = ShapeBuilders::copyShapeInfoAndType(predictionsShapeInfo, outType, false, block.getWorkspace());
 auto dLdwShapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, outType, false, block.getWorkspace());
 auto dLdlShapeInfo = ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());

 return SHAPELIST(dLdpShapeInfo, dLdwShapeInfo, dLdlShapeInfo);
}
}  // namespace ops
}  // namespace sd

#endif
