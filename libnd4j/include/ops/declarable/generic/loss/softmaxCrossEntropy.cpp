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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 25.11.2017.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_softmax_cross_entropy_loss)

#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(softmax_cross_entropy_loss, 3, 1, false, 1, 1) {
  auto logits = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  double labelsSmoothing = T_ARG(0);

  // input validation
  REQUIRE_TRUE(labels->isSameShape(logits), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, "
               "but got %i instead!",
               reductionMode);
  // smoothing is possible for rank of logits/labels > 1
  REQUIRE_TRUE(labels->rankOf() > 1 || (labels->rankOf() == 1 && labelsSmoothing == 0.), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS OP: smoothing is not possible when rank of labels/ logits = 1 !");

  if (!output->isScalar()) {
    // weights array can be single scalar or has the same shape as output, and must be broadcastable to output shape
    REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == output->rankOf(), 0,
                 "SOFTMAX_CROSS_ENTROPY_LOSS OP: weights array should be scalar or have the same rank as output array, "
                 "but got %i and %i correspondingly!",
                 weights->rankOf(), output->rankOf());
    // check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *output), 0,
                 "SOFTMAX_CROSS_ENTROPY_LOSS OP: shapes of weights and output arrays should be broadcastable, but got "
                 "weights = %s and output = %s instead!",
                 ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  }

  // If label_smoothing is nonzero, smooth the labels towards 1/num_classes: new_onehot_labels = onehot_labels * (1 -
  // label_smoothing) + label_smoothing / num_classes num_classes = labels->sizeAt(1)
  NDArray* cLabels = new NDArray(labels->cast(weights->dataType()));
  NDArray* newLabels = cLabels;
  if (labelsSmoothing != 0.) {
    newLabels = new NDArray(cLabels);
    NDArray* term1 = (1.f - labelsSmoothing) * (*cLabels);
    NDArray* term2 = (*term1) + (labelsSmoothing / cLabels->sizeAt(1));
    delete term1;
    newLabels->assign(term2);
    delete term2;
  }

  // main formula: result = - sum_i(lables_i * log(softmax_i)) - sum over last dimension
  // softmax_i = exp(logits_i) / sum_j(exp(logits_j))
  // so result = sum_i( lables_i * (log(sum_j(exp(logits_j))) - logits_i) )
  // for numerical stability we use shifted logits (one can approve this using simple math):
  // softmax_i = exp(logits_i - maxLogit) / sum_j(exp(logits_j - maxLogit))
  // maxLogit is max among logits_i

  std::vector<LongType> dimensions = {-1};
  NDArray* maxLogits = logits->reduceAlongDimension(reduce::Max, &dimensions, true);
  NDArray* shiftedLogits_ptr = (*logits) - (*maxLogits);
  delete maxLogits;
  NDArray shiftedLogits = *shiftedLogits_ptr;
  delete shiftedLogits_ptr;
  
  NDArray* expShifted = shiftedLogits.transform(transform::Exp);
  NDArray* sumExp = expShifted->reduceAlongDimension(reduce::Sum, &dimensions, true);
  delete expShifted;
  NDArray* logSumExp_ptr = sumExp->transform(transform::Log);
  delete sumExp;
  NDArray logSumExp = *logSumExp_ptr;
  delete logSumExp_ptr;
  
  // E = (newLabels * (logSumExp - shiftedLogits)).reduceAlongDimension(Sum)
  NDArray* diff = logSumExp - shiftedLogits;
  NDArray* product = (*newLabels) * (*diff);
  delete diff;
  NDArray* E_ptr = product->reduceAlongDimension(reduce::Sum, &dimensions);
  delete product;
  NDArray E = *E_ptr;
  delete E_ptr;

  // perform weights broadcasting/tile to E if it is necessary
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(&E)) {
    std::vector<LongType> weightsShape = {weights->lengthOf()};
    if (E.rankOf() == 1 && weights->isVector() && weights->rankOf() > 1)
      weightsBroad = weights->reshape(weights->ordering(), weightsShape);
    else
      weightsBroad = new NDArray(weights->tileToShape(E.shapeInfo()));
  }

  // multiply E on weights
  E *= *weightsBroad;

  switch (reductionMode) {
    case 0:  // 0 - "none", un-reduced weighted losses with the same shape as labels.
      output->assign(&E);
      break;

    case 1: {  // 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
      E.reduceNumber(reduce::Sum, output);
      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
      // all elements of weightsBroad array
      double sum;
      if (weights->isScalar())
        sum = weights->e<double>(0) * E.lengthOf();
      else {
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
        *output = 0.;
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

  if (newLabels != cLabels) delete newLabels;

  delete cLabels;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(softmax_cross_entropy_loss) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS, ALL_INTS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(softmax_cross_entropy_loss) {
  auto logitsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  // labels and logits must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(logitsShapeInfo, labelsShapeInfo), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS OP: labels and logits arrays must have the same shapes, but got %s and %s "
               "correspondingly!",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));
  LongType* outShapeInfo = nullptr;

  if (INT_ARG(0) != 0)  // in this case output is scalar
    outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
  else {  // in this case output has the shape as labels and logits minus last dimension
    std::vector<LongType> dimensions = {-1};
    outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(logitsShapeInfo), &dimensions, logitsShapeInfo, false,
                                                   true, block.getWorkspace());

    // weights array can be single scalar or has the same rank as output, and must be broadcastable to output
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(outShapeInfo), 0,
                 "SOFTMAX_CROSS_ENTROPY_LOSS OP: weights array should be scalar or have the same rank as output array, "
                 "but got %i and %i correspondingly!",
                 shape::rank(weightsShapeInfo), shape::rank(outShapeInfo));
    // check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(
        shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, outShapeInfo), 0,
        "SOFTMAX_CROSS_ENTROPY_LOSS OP: shapes of weights and output arrays should be broadcastable, but got weights = "
        "%s and output = %s instead!",
        ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(outShapeInfo).c_str());
  }

  return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(softmax_cross_entropy_loss_grad, 3, 3, false, 1, 1) {
  auto logits = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto dLdp = OUTPUT_VARIABLE(0);  // dL/dlogits
  auto dLdw = OUTPUT_VARIABLE(1);  // dL/dweights
  auto dLdl = OUTPUT_VARIABLE(2);  // dL/dlabels

  auto labelsSmoothing = T_ARG(0);

  int reductionMode =
      INT_ARG(0);  // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  // take into account Alex's proposition to treat "none" the same as "weighted_sum" mode when calculating gradients
  if (reductionMode == 0) reductionMode = 1;

  std::vector<LongType> *dimensions =  new std::vector<LongType>({-1});

  // input validation
  REQUIRE_TRUE(labels->isSameShape(logits), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: labels and logits arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, "
               "2, 3, but got %i instead!",
               reductionMode);
  auto lossShapeInfo = ShapeUtils::evalReduceShapeInfo(logits->ordering(), dimensions, logits->shapeInfo(), false,
                                                       false, block.getWorkspace());
  // weights array can be single scalar or has the same shape as loss, and must be broadcastable to loss shape
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == shape::rank(lossShapeInfo), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: weights array should be scalar or have the same rank as loss "
               "array, but got %i and %i correspondingly!",
               weights->rankOf(), shape::rank(lossShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(weights->shapeInfo(), lossShapeInfo), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: shapes of weights and loss arrays should be broadcastable, but got "
               "weights = %s and loss = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(lossShapeInfo).c_str());
  // smoothing is possible for rank of logits/labels > 1
  REQUIRE_TRUE(labels->rankOf() > 1 || (labels->rankOf() == 1 && labelsSmoothing == 0.), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: smoothing is not possible when rank of labels/ logits = 1 !");

  // If label_smoothing is nonzero, smooth the labels towards 1/num_classes: new_onehot_labels = onehot_labels * (1 -
  // label_smoothing) + label_smoothing / num_classes num_classes = labels->sizeAt(1)
  NDArray* cLabels = new NDArray(labels->cast(weights->dataType()));
  NDArray* newLabels = cLabels;
  if (labelsSmoothing != 0.) {
    newLabels = new NDArray(labels->shapeInfo(), dLdl->dataType(), false, block.launchContext());
    NDArray* term1 = (1.f - labelsSmoothing) * (*cLabels);
    NDArray* term2 = (*term1) + (labelsSmoothing / cLabels->sizeAt(1));
    delete term1;
    newLabels->assign(term2);
    delete term2;
  }

  // Compute softmax
  NDArray* maxLogits = logits->reduceAlongDimension(reduce::Max, dimensions, true);
  NDArray* shiftedLogits_ptr = (*logits) - (*maxLogits);
  delete maxLogits;
  NDArray* expShifted = shiftedLogits_ptr->transform(transform::Exp);
  delete shiftedLogits_ptr;
  NDArray* sumExp = expShifted->reduceAlongDimension(reduce::Sum, dimensions, true);
  NDArray* softmax_ptr = (*expShifted) / (*sumExp);
  delete expShifted;
  delete sumExp;
  NDArray softmax = *softmax_ptr;
  delete softmax_ptr;

  // dEdp = softmax * sum_i(lables_i) - labels
  NDArray* labelSum = newLabels->reduceAlongDimension(reduce::Sum, dimensions, true);
  NDArray* softmaxTimesLabelSum = softmax * (*labelSum);
  delete labelSum;
  NDArray* dLdpTemp_ptr = (*softmaxTimesLabelSum) - (*newLabels);
  delete softmaxTimesLabelSum;
  dLdp->assign(dLdpTemp_ptr);
  delete dLdpTemp_ptr;
  
  // dEdl = -log(softmax)
  NDArray* logSoftmax = softmax.transform(transform::Log);
  NDArray negLogSoftmax = -(*logSoftmax);  // unary negation returns value
  delete logSoftmax;
  NDArray* dLdlTemp_ptr = negLogSoftmax * (1.f - labelsSmoothing);
  dLdl->assign(dLdlTemp_ptr);
  delete dLdlTemp_ptr;

  // Compute E for gradient calculations
  NDArray* maxLogits2 = logits->reduceAlongDimension(reduce::Max, dimensions, true);
  NDArray* shiftedLogits2_ptr = (*logits) - (*maxLogits2);
  delete maxLogits2;
  NDArray shiftedLogits = *shiftedLogits2_ptr;
  delete shiftedLogits2_ptr;
  
  NDArray* expShifted2 = shiftedLogits.transform(transform::Exp);
  NDArray* sumExp2 = expShifted2->reduceAlongDimension(reduce::Sum, dimensions, true);
  delete expShifted2;
  NDArray* logSumExp_ptr = sumExp2->transform(transform::Log);
  delete sumExp2;
  NDArray logSumExp = *logSumExp_ptr;
  delete logSumExp_ptr;
  
  NDArray* diff = logSumExp - shiftedLogits;
  NDArray* product = (*newLabels) * (*diff);
  delete diff;
  NDArray* E_ptr = product->reduceAlongDimension(reduce::Sum, dimensions);
  delete product;
  NDArray E = *E_ptr;
  delete E_ptr;

  // perform weights broadcasting/tile to E if it is necessary
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(&E))
    weightsBroad = new NDArray(weights->tileToShape(E.shapeInfo()));

  auto excludeDims = ShapeUtils::evalDimsToExclude(dLdp->rankOf(), dimensions->size(), dimensions->data());

  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

      if (weights->isScalar() || weights->lengthOf() == 1) {
        NDArray* eSum = E.reduceNumber(reduce::Sum);
        dLdw->assign(eSum);
        delete eSum;
        *dLdp *= *weights;
        *dLdl *= *weights;
      } else {
        dLdp->applyBroadcast(broadcast::Multiply, excludeDims, weightsBroad, dLdp);
        dLdl->applyBroadcast(broadcast::Multiply, excludeDims, weightsBroad, dLdl);

        if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else
          dLdw->assign(&E);
      }

      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
      // all elements of weightsBroad array
      NDArray* sum_ptr = nullptr;
      if (weights->isScalar())
        sum_ptr = (*weights) * E.lengthOf();
      else
        sum_ptr = weightsBroad->reduceNumber(reduce::Sum);
      
      NDArray sum = *sum_ptr;
      delete sum_ptr;

      if (sum.e<double>(0) == 0.) {
        *dLdp = 0.;
        *dLdl = 0.;
        *dLdw = 0.;
      } else {
        if (weights->isScalar() || weights->lengthOf() == 1) {
          NDArray* temp_ptr = (*weights) / sum;
          NDArray temp = *temp_ptr;
          delete temp_ptr;
          *dLdp *= temp;
          *dLdl *= temp;
          *dLdw = 0.;
        } else {
          NDArray* temp_ptr = (*weightsBroad) / sum;
          NDArray temp = *temp_ptr;
          delete temp_ptr;
          dLdp->applyBroadcast(broadcast::Multiply, dimensions, &temp, dLdp);
          dLdl->applyBroadcast(broadcast::Multiply, dimensions, &temp, dLdl);

          if (weights != weightsBroad) {
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
            
            NDArray* sumSquared = sum * sum;
            NDArray* result = (*numerator) / (*sumSquared);
            delete numerator;
            delete sumSquared;
            
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
            
            NDArray* sumSquared = sum * sum;
            NDArray* result = (*numerator) / (*sumSquared);
            delete numerator;
            delete sumSquared;
            
            dLdw->assign(result);
            delete result;
          }
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
        if (weights->isScalar() || weights->lengthOf() == 1) {
          NDArray* temp_ptr = (*weights) / numOfNonZeroWeights;
          NDArray temp = *temp_ptr;
          delete temp_ptr;
          *dLdp *= temp;
          *dLdl *= temp;
          
          NDArray* eSum = E.reduceNumber(reduce::Sum);
          NDArray* result = (*eSum) / numOfNonZeroWeights;
          delete eSum;
          dLdw->assign(result);
          delete result;
        } else {
          NDArray* temp_ptr = (*weightsBroad) / numOfNonZeroWeights;
          NDArray temp = *temp_ptr;
          delete temp_ptr;
          dLdp->applyBroadcast(broadcast::Multiply, dimensions, &temp, dLdp);
          dLdl->applyBroadcast(broadcast::Multiply, dimensions, &temp, dLdl);

          if (weights != weightsBroad) {
            std::vector<LongType> axesToReduceAlong =
                ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
            E.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
            *dLdw /= numOfNonZeroWeights;
          } else {
            NDArray* eDivNum = E / numOfNonZeroWeights;
            dLdw->assign(eDivNum);
            delete eDivNum;
          }
        }
      }
      break;
    }
  }

  if (weightsBroad != weights) delete weightsBroad;

  if (newLabels != cLabels) delete newLabels;

  delete cLabels;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(softmax_cross_entropy_loss_grad) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS, ALL_INTS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {ALL_FLOATS})
      ->setAllowedInputTypes(5, {ALL_FLOATS})
      ->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(softmax_cross_entropy_loss_grad) {
  auto logitsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  std::vector<LongType> dimensions = {-1};

  // labels and logits must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(logitsShapeInfo, labelsShapeInfo), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: labels and logits arrays must have the same shapes, but got %s and "
               "%s correspondingly!",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(), ShapeUtils::shapeAsString(logitsShapeInfo).c_str());
  auto lossShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(logitsShapeInfo), &dimensions, logitsShapeInfo,
                                                       false, false, block.getWorkspace());
  // weights array can be single scalar or has the same rank as loss, and must be broadcastable to loss
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(lossShapeInfo), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: weights array should be scalar or have the same rank as loss "
               "array, but got %i and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(lossShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, lossShapeInfo),
               0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: shapes of weights and loss arrays should be broadcastable, but got "
               "weights = %s and loss = %s instead!",
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(lossShapeInfo).c_str());

  auto outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(logitsShapeInfo));

  auto dLdpShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(outType, shape::order(logitsShapeInfo),
                                                                             shape::rank(logitsShapeInfo),
                                                                             shape::shapeOf(logitsShapeInfo))->primary();

  auto dLdwShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(outType, shape::order(weightsShapeInfo),
                                                                             shape::rank(weightsShapeInfo),
                                                                             shape::shapeOf(weightsShapeInfo))->primary();

  auto dLdlShapeInfo = ConstantShapeHelper::getInstance().bufferForShapeInfo(outType, shape::order(labelsShapeInfo),
                                                                             shape::rank(labelsShapeInfo),
                                                                             shape::shapeOf(labelsShapeInfo))->primary();
  return SHAPELIST(dLdpShapeInfo, dLdwShapeInfo, dLdlShapeInfo);
}

}  // namespace ops
}  // namespace sd

#endif
