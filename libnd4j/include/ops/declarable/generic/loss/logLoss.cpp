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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 23.11.2017
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_log_loss)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(log_loss, 3, 1, false, 1, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  // FIXME: double?
  double epsilon = T_ARG(0);

  // input validation
  REQUIRE_TRUE(
      labels->isSameShape(predictions), 0,
      "LOG_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "LOG_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i "
               "correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "LOG_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and "
               "labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(
      reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
      "LOG_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!",
      reductionMode);

  // perform weights broadcasting/tile to predictions if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(predictions))
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));

  // E = -labels * log(predictions + epsilon) - (1 - labels) * log(1 + epsilon - predictions)
  // Break this into steps:
  NDArray* predPlusEps = (*predictions) + epsilon;
  NDArray* logPredPlusEps = predPlusEps->transform(transform::Log);
  delete predPlusEps;
  
  NDArray negLabels = -(*labels);  // unary negation returns value
  NDArray* term1 = negLabels * (*logPredPlusEps);
  delete logPredPlusEps;
  
  NDArray* oneMinusLabels = 1. - (*labels);
  NDArray* onePlusEpsMinusPred = (1. + epsilon) - (*predictions);
  NDArray* logOnePlusEpsMinusPred = onePlusEpsMinusPred->transform(transform::Log);
  delete onePlusEpsMinusPred;
  
  NDArray* term2 = (*oneMinusLabels) * (*logOnePlusEpsMinusPred);
  delete oneMinusLabels;
  delete logOnePlusEpsMinusPred;
  
  NDArray* E_ptr = (*term1) - (*term2);
  delete term1;
  delete term2;
  
  NDArray E = *E_ptr;
  delete E_ptr;

  // multiply E on weights
  E *= *weightsBroad;

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
      double sum;
      if (weights->isScalar()) {
        sum = weights->e<double>(0) * E.lengthOf();
      } else {
        NDArray* sumPtr = weightsBroad->reduceNumber(reduce::Sum);
        sum = sumPtr->e<double>(0);
        delete sumPtr;
      }

      if (sum == 0.)
        *output = 0.;
      else {
        NDArray* eSum = E.reduceNumber(reduce::Sum);
        NDArray* result = (*eSum) / sum;
        delete eSum;
        output->assign(result);
        delete result;
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
      // array divided by number of non-zero weights
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E.lengthOf();
      } else {
        NDArray* countNonZero = weightsBroad->reduceNumber(reduce::CountNonZero);
        numOfNonZeroWeights = countNonZero->e<LongType>(0);
        delete countNonZero;
      }

      if (numOfNonZeroWeights == 0)
        (*output) = 0.;
      else {
        NDArray* eSum = E.reduceNumber(reduce::Sum);
        NDArray* result = (*eSum) / double(numOfNonZeroWeights);
        delete eSum;
        output->assign(result);
        delete result;
      }
      break;
    }
  }

  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(log_loss) { getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS}); }

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(log_loss) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(
      shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
      "LOG_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "LOG_LOSS OP: weights array should be scalar or have the same rank as labels array, but got %i and %i "
               "correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "LOG_LOSS OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels = %s "
      "instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));
  LongType* outShapeInfo = nullptr;

  if (INT_ARG(0) != 0)  // in this case output is scalar
    outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
  else {  // in this case output has the same shape as labels and predictions
    outShapeInfo = ConstantShapeHelper::getInstance()
                       .bufferForShapeInfo(outType, shape::order(labelsShapeInfo), shape::rank(labelsShapeInfo),
                                           shape::shapeOf(labelsShapeInfo))
                       ->primary();
  }
  return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(log_loss_grad, 3, 3, false, 1, 1) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto dLdp = OUTPUT_VARIABLE(0);  // dL/dpredictions
  auto dLdw = OUTPUT_VARIABLE(1);  // dL/dweights
  auto dLdl = OUTPUT_VARIABLE(2);  // dL/dlabels

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  // take into account Alex's proposition to treat "none" the same as "weighted_sum" mode when calculating gradients
  if (reductionMode == 0) reductionMode = 1;

  // FIXME: double?
  double epsilon = T_ARG(0);

  // input validation
  REQUIRE_TRUE(
      labels->isSameShape(predictions), 0,
      "LOG_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == labels->rankOf(), 0,
               "LOG_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i and "
               "%i correspondingly!",
               weights->rankOf(), labels->rankOf());
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *labels), 0,
               "LOG_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s "
               "and labels = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(
      reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
      "LOG_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but got %i instead!",
      reductionMode);

  // perform weights broadcasting/tile to labels if needed
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(predictions))
    weightsBroad = new NDArray(weights->tileToShape(predictions->shapeInfo()));

  NDArray* predictPlusEps_ptr = (*predictions) + epsilon;
  NDArray predictPlusEps = *predictPlusEps_ptr;
  delete predictPlusEps_ptr;
  
  NDArray* oneMinusLabels_ptr = 1. - (*labels);
  NDArray oneMinusLabels = *oneMinusLabels_ptr;
  delete oneMinusLabels_ptr;
  
  NDArray* onePlusEpsMinusPredict_ptr = (1. + epsilon) - (*predictions);
  NDArray onePlusEpsMinusPredict = *onePlusEpsMinusPredict_ptr;
  delete onePlusEpsMinusPredict_ptr;

  // dE_i/dp_i = (1-y_i)/(1-p_i+eps) - y_i/(p_i+eps)
  NDArray* oneMinusDiv = oneMinusLabels / onePlusEpsMinusPredict;
  NDArray* labelsDiv = (*labels) / predictPlusEps;
  NDArray* dEdp = (*oneMinusDiv) - (*labelsDiv);
  delete oneMinusDiv;
  delete labelsDiv;
  dLdp->assign(dEdp);
  delete dEdp;
  
  // dE_i/dy_i = log((1+2eps)/(p_i+eps) - 1)
  double onePlus2Eps = 1. + 2. * epsilon;
  NDArray* ratio = onePlus2Eps / predictPlusEps;
  NDArray* ratioMinus1 = (*ratio) - 1.;
  delete ratio;
  ratioMinus1->applyTransform(transform::Log, dLdl);
  delete ratioMinus1;

  // Compute E for gradient calculations
  NDArray* logPredPlusEps = predictPlusEps.transform(transform::Log);
  NDArray* logOnePlusEpsMinusPred = onePlusEpsMinusPredict.transform(transform::Log);
  
  NDArray negLabels = -(*labels);  // unary negation returns value
  NDArray* term1 = negLabels * (*logPredPlusEps);
  delete logPredPlusEps;
  
  NDArray* term2 = oneMinusLabels * (*logOnePlusEpsMinusPred);
  delete logOnePlusEpsMinusPred;
  
  NDArray* E_ptr = (*term1) - (*term2);
  delete term1;
  delete term2;
  
  NDArray E = *E_ptr;
  delete E_ptr;

  // process 3 possible reduction modes below
  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

      *dLdp *= *weightsBroad;
      *dLdl *= *weightsBroad;

      if (weights->isScalar()) {
        NDArray* eSum = E.reduceNumber(reduce::Sum);
        dLdw->assign(eSum);
        delete eSum;
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

      double sum;
      if (weights->isScalar()) {
        sum = weights->e<double>(0) * E.lengthOf();
      } else {
        NDArray* sumPtr = weightsBroad->reduceNumber(reduce::Sum);
        sum = sumPtr->e<double>(0);
        delete sumPtr;
      }

      if (sum == 0.) {
        *dLdp = 0.;
        *dLdl = 0.;
        *dLdw = 0.;
      } else {
        NDArray* weightsDivSum = (*weightsBroad) / sum;
        NDArray temp = *weightsDivSum;
        delete weightsDivSum;
        
        *dLdp *= temp;
        *dLdl *= temp;

        if (weights->isScalar())
          *dLdw = 0.;
        else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          
          // Compute (E * sum - (E * weightsBroad).reduceNumber(Sum)) / (sum * sum)
          NDArray* ETimesSum = E * sum;
          NDArray* ETimesWeights = E * (*weightsBroad);
          NDArray* ETimesWeightsSum = ETimesWeights->reduceNumber(reduce::Sum);
          delete ETimesWeights;
          
          NDArray* numerator = (*ETimesSum) - (*ETimesWeightsSum);
          delete ETimesSum;
          delete ETimesWeightsSum;
          
          double sumSquared = sum * sum;
          NDArray* result = (*numerator) / sumSquared;
          delete numerator;
          
          result->reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          delete result;
        } else {
          // Compute (E * sum - (E * weightsBroad).reduceNumber(Sum)) / (sum * sum)
          NDArray* ETimesSum = E * sum;
          NDArray* ETimesWeights = E * (*weightsBroad);
          NDArray* ETimesWeightsSum = ETimesWeights->reduceNumber(reduce::Sum);
          delete ETimesWeights;
          
          NDArray* numerator = (*ETimesSum) - (*ETimesWeightsSum);
          delete ETimesSum;
          delete ETimesWeightsSum;
          
          double sumSquared = sum * sum;
          NDArray* result = (*numerator) / sumSquared;
          delete numerator;
          
          dLdw->assign(result);
          delete result;
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
        NDArray* countNonZero = weightsBroad->reduceNumber(reduce::CountNonZero);
        numOfNonZeroWeights = countNonZero->e<LongType>(0);
        delete countNonZero;
      }

      if (numOfNonZeroWeights == 0) {
        *dLdp = 0.;
        *dLdl = 0.;
        *dLdw = 0.;
      } else {
        auto* numOfNonZeroWeightsScalar =
            NDArrayFactory::create(dLdw->dataType(), numOfNonZeroWeights, block.launchContext());
        
        if (weights->isScalar()) {
          NDArray* eSum = E.reduceNumber(reduce::Sum);
          NDArray* result = (*eSum) / numOfNonZeroWeights;
          delete eSum;
          dLdw->assign(result);
          delete result;
        } else if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          *dLdw /= *numOfNonZeroWeightsScalar;
        } else {
          NDArray* EDivNum = E / (*numOfNonZeroWeightsScalar);
          dLdw->assign(EDivNum);
          delete EDivNum;

          NDArray* weightsDivNum = (*weightsBroad) / (*numOfNonZeroWeightsScalar);
          NDArray temp = *weightsDivNum;
          delete weightsDivNum;
          
          *dLdp *= temp;
          *dLdl *= temp;
        }
        
        delete numOfNonZeroWeightsScalar;
      }
      break;
    }
  }

  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(log_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(log_loss_grad) {
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(
      shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
      "LOG_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and %s correspondingly !",
      ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // weights array can be single scalar or has the same rank as labels, and must be broadcastable to labels
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(labelsShapeInfo), 0,
               "LOG_LOSS_GRAD OP: weights array should be scalar or have the same rank as labels array, but got %i and "
               "%i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(labelsShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(
      shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, labelsShapeInfo), 0,
      "LOG_LOSS_GRAD OP: shapes of weights and labels arrays should be broadcastable, but got weights = %s and labels "
      "= %s instead!",
      ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(labelsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));

  auto dLdpShapeInfo = ShapeBuilders::copyShapeInfoAndType(predictionsShapeInfo, outType, false, block.getWorkspace());
  auto dLdwShapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, outType, false, block.getWorkspace());
  auto dLdlShapeInfo = ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());

  return SHAPELIST(CONSTANT(dLdpShapeInfo), CONSTANT(dLdwShapeInfo), CONSTANT(dLdlShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
