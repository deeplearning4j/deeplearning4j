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

#include <ops/declarable/headers/loss.h>

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

  auto ctx = block.launchContext();

  // If label_smoothing is nonzero, smooth the labels towards 1/num_classes.
  // cLabels is heap-allocated by cast(); newLabels may also be heap-allocated.
  NDArray* cLabels = labels->cast(weights->dataType());
  NDArray* newLabels = cLabels;
  if (labelsSmoothing != 0.) {
    newLabels = new NDArray(cLabels->shapeInfo(), false, ctx);
    // newLabels = (1 - labelsSmoothing) * cLabels + (labelsSmoothing / numClasses)
    cLabels->applyScalar(scalar::Multiply, (1.0 - labelsSmoothing), newLabels);
    double addVal = labelsSmoothing / cLabels->sizeAt(1);
    newLabels->applyScalar(scalar::Add, addVal, newLabels);
  }

  std::vector<LongType> dimensions = {-1};

  // maxLogits: heap (reduceAlongDimension)
  NDArray* maxLogits = logits->reduceAlongDimension(reduce::Max, &dimensions, true);

  // shiftedLogits = logits - maxLogits  (stack)
  // maxLogits has shape [...,1] (keepDims); logits has shape [...,C] → must use broadcast subtract
  NDArray shiftedLogits(logits->shapeInfo(), logits->dataType(), false, ctx);
  logits->applyTrueBroadcast(BroadcastOpsTuple::Subtract(), maxLogits, &shiftedLogits, false);

  // expShifted = exp(shiftedLogits)  (stack)
  NDArray expShifted(shiftedLogits.shapeInfo(), false, ctx);
  shiftedLogits.applyTransform(transform::Exp, &expShifted);

  // sumExp: heap (reduceAlongDimension)
  NDArray* sumExp = expShifted.reduceAlongDimension(reduce::Sum, &dimensions, true);

  // logSumExp = log(sumExp)  (stack, same shape as sumExp)
  NDArray logSumExp(sumExp->shapeInfo(), false, ctx);
  sumExp->applyTransform(transform::Log, &logSumExp);

  // diff = logSumExp - shiftedLogits  (stack, same shape as logits)
  // logSumExp has shape [...,1]; shiftedLogits has shape [...,C] → must use broadcast subtract
  NDArray diff(logits->shapeInfo(), logits->dataType(), false, ctx);
  logSumExp.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &shiftedLogits, &diff, false);

  // product = newLabels * diff  (stack)
  NDArray product(logits->shapeInfo(), false, ctx);
  newLabels->applyPairwiseTransform(pairwise::Multiply, &diff, &product);

  // E: heap (reduceAlongDimension, no keepDims)
  NDArray* E = product.reduceAlongDimension(reduce::Sum, &dimensions);

  // perform weights broadcasting/tile to E if it is necessary
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(E)) {
    std::vector<LongType> weightsShape = {weights->lengthOf()};
    if (E->rankOf() == 1 && weights->isVector() && weights->rankOf() > 1)
      weightsBroad = weights->reshape(weights->ordering(), weightsShape);
    else
      weightsBroad = new NDArray(weights->tileToShape(E->shapeInfo()));
  }

  // multiply E on weights (original approach — in-place pairwise multiply)
  *E *= *weightsBroad;

  switch (reductionMode) {
    case 0:  // 0 - "none", un-reduced weighted losses with the same shape as labels.
      output->assign(E);
      break;

    case 1: {  // 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
      E->reduceNumber(reduce::Sum, output);
      break;
    }
    case 2: {  // 2 - "weighted_mean"
      double sum;
      if (weights->isScalar())
        sum = weights->e<double>(0) * E->lengthOf();
      else {
        NDArray* sumPtr = weightsBroad->reduceNumber(reduce::Sum);
        sum = sumPtr->e<double>(0);
        delete sumPtr;
      }

      if (sum == 0.) {
        *output = 0.;
      } else {
        NDArray* eSum = E->reduceNumber(reduce::Sum);
        NDArray* result = (*eSum) / sum;
        output->assign(result);
        delete eSum;
        delete result;
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights"
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E->lengthOf();
      } else {
        NDArray* countNonZero = weightsBroad->reduceNumber(reduce::CountNonZero);
        numOfNonZeroWeights = countNonZero->e<LongType>(0);
        delete countNonZero;
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        output->assign(zero);
      } else {
        NDArray* eSum = E->reduceNumber(reduce::Sum);
        double eSumVal = eSum->e<double>(0);
        delete eSum;
        double result = eSumVal / double(numOfNonZeroWeights);
        output->assign(result);
      }
      break;
    }
  }

  // Clean up heap-allocated intermediates
  delete maxLogits;
  delete sumExp;
  delete E;
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

  std::vector<LongType> *dimensions = new std::vector<LongType>({-1});

  // input validation
  REQUIRE_TRUE(labels->isSameShape(logits), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: labels and logits arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(logits).c_str());
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, "
               "2, 3, but got %i instead!",
               reductionMode);
  auto lossShapeInfo = ShapeUtils::evalReduceShapeInfo(logits->ordering(), dimensions, logits->shapeInfo(), false,
                                                       false, block.getWorkspace());
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == shape::rank(lossShapeInfo), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: weights array should be scalar or have the same rank as loss "
               "array, but got %i and %i correspondingly!",
               weights->rankOf(), shape::rank(lossShapeInfo));
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(weights->shapeInfo(), lossShapeInfo), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: shapes of weights and loss arrays should be broadcastable, but got "
               "weights = %s and loss = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(lossShapeInfo).c_str());
  REQUIRE_TRUE(labels->rankOf() > 1 || (labels->rankOf() == 1 && labelsSmoothing == 0.), 0,
               "SOFTMAX_CROSS_ENTROPY_LOSS_GRAD OP: smoothing is not possible when rank of labels/ logits = 1 !");

  auto ctx = block.launchContext();

  // If label_smoothing is nonzero, smooth the labels towards 1/num_classes.
  NDArray* cLabels = labels->cast(weights->dataType());
  NDArray* newLabels = cLabels;
  if (labelsSmoothing != 0.) {
    newLabels = new NDArray(cLabels->shapeInfo(), false, ctx);
    // newLabels = (1 - labelsSmoothing) * cLabels + (labelsSmoothing / numClasses)
    cLabels->applyScalar(scalar::Multiply, (1.0 - labelsSmoothing), newLabels);
    double addVal = labelsSmoothing / cLabels->sizeAt(1);
    newLabels->applyScalar(scalar::Add, addVal, newLabels);
  }

  // Compute softmax
  // maxLogits: heap
  NDArray* maxLogits = logits->reduceAlongDimension(reduce::Max, dimensions, true);

  // shiftedLogits = logits - maxLogits  (stack)
  // maxLogits has shape [...,1] (keepDims); logits has shape [...,C] → must use broadcast subtract
  NDArray shiftedLogits(logits->shapeInfo(), logits->dataType(), false, ctx);
  logits->applyTrueBroadcast(BroadcastOpsTuple::Subtract(), maxLogits, &shiftedLogits, false);

  // expShifted = exp(shiftedLogits)  (stack)
  NDArray expShifted(shiftedLogits.shapeInfo(), false, ctx);
  shiftedLogits.applyTransform(transform::Exp, &expShifted);

  // sumExp: heap
  NDArray* sumExp = expShifted.reduceAlongDimension(reduce::Sum, dimensions, true);

  // softmax = expShifted / sumExp  (stack)
  // sumExp has shape [...,1] (keepDims); expShifted has shape [...,C] → must use broadcast divide
  NDArray softmax(expShifted.shapeInfo(), expShifted.dataType(), false, ctx);
  expShifted.applyTrueBroadcast(BroadcastOpsTuple::Divide(), sumExp, &softmax, false);

  // dEdp = softmax * sum_i(labels_i) - newLabels
  // labelSum: heap
  NDArray* labelSum = newLabels->reduceAlongDimension(reduce::Sum, dimensions, true);

  // softmaxTimesLabelSum = softmax * labelSum  (stack)
  // labelSum has shape [...,1] (keepDims); softmax has shape [...,C] → must use broadcast multiply
  NDArray softmaxTimesLabelSum(softmax.shapeInfo(), softmax.dataType(), false, ctx);
  softmax.applyTrueBroadcast(BroadcastOpsTuple::Multiply(), labelSum, &softmaxTimesLabelSum, false);

  // dLdp = softmaxTimesLabelSum - newLabels  (write directly to output)
  softmaxTimesLabelSum.applyPairwiseTransform(pairwise::Subtract, newLabels, dLdp);

  // dEdl = -(1 - labelsSmoothing) * log(softmax)
  // logSoftmax = log(softmax)  (stack)
  NDArray logSoftmax(softmax.shapeInfo(), false, ctx);
  softmax.applyTransform(transform::Log, &logSoftmax);

  // negLogSoftmax = -logSoftmax  (stack)
  NDArray negLogSoftmax(logSoftmax.shapeInfo(), false, ctx);
  logSoftmax.applyTransform(transform::Neg, &negLogSoftmax);

  // dLdl = negLogSoftmax * (1 - labelsSmoothing)
  negLogSoftmax.applyScalar(scalar::Multiply, (1.0 - labelsSmoothing), dLdl);

  // Recompute E for gradient weight calculations
  // maxLogits2: heap
  NDArray* maxLogits2 = logits->reduceAlongDimension(reduce::Max, dimensions, true);

  // shiftedLogits2 = logits - maxLogits2  (stack)
  // maxLogits2 has shape [...,1] (keepDims); logits has shape [...,C] → must use broadcast subtract
  NDArray shiftedLogits2(logits->shapeInfo(), logits->dataType(), false, ctx);
  logits->applyTrueBroadcast(BroadcastOpsTuple::Subtract(), maxLogits2, &shiftedLogits2, false);

  // expShifted2 = exp(shiftedLogits2)  (stack)
  NDArray expShifted2(shiftedLogits2.shapeInfo(), false, ctx);
  shiftedLogits2.applyTransform(transform::Exp, &expShifted2);

  // sumExp2: heap
  NDArray* sumExp2 = expShifted2.reduceAlongDimension(reduce::Sum, dimensions, true);

  // logSumExp = log(sumExp2)  (stack)
  NDArray logSumExp(sumExp2->shapeInfo(), false, ctx);
  sumExp2->applyTransform(transform::Log, &logSumExp);

  // diff = logSumExp - shiftedLogits2  (stack)
  // logSumExp has shape [...,1]; shiftedLogits2 has shape [...,C] → must use broadcast subtract
  NDArray diff(logits->shapeInfo(), logits->dataType(), false, ctx);
  logSumExp.applyTrueBroadcast(BroadcastOpsTuple::Subtract(), &shiftedLogits2, &diff, false);

  // product = newLabels * diff  (stack)
  NDArray product(logits->shapeInfo(), false, ctx);
  newLabels->applyPairwiseTransform(pairwise::Multiply, &diff, &product);

  // E: heap
  NDArray* E = product.reduceAlongDimension(reduce::Sum, dimensions);

  // perform weights broadcasting/tile to E if it is necessary
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(E))
    weightsBroad = new NDArray(weights->tileToShape(E->shapeInfo()));

  // Also tile weights to dLdp shape for logit/label gradient scaling.
  // dLdp has shape [..., numClasses] while weightsBroad has shape [...] (1 fewer dim).
  // For perExample [N]: reshape to [N,1] then tile to [N,C] for row-wise scaling.
  // For perOutput [N,C]: same shape as dLdp, no tiling needed.
  auto weightsBroadFull = weights;
  if (!weights->isScalar() && !weights->isSameShape(dLdp)) {
    if (weightsBroad->rankOf() < dLdp->rankOf()) {
      // weightsBroad has shape [...] (e.g. [N]); dLdp has shape [..., C] (e.g. [N, C]).
      // Append a trailing 1 to weightsBroad's shape, then tile to dLdp's shape.
      // This gives row-wise scaling: result[i,j] = weightsBroad[i] for all j.
      std::vector<LongType> wColShape(weightsBroad->rankOf() + 1);
      for (int _d = 0; _d < weightsBroad->rankOf(); _d++) wColShape[_d] = weightsBroad->sizeAt(_d);
      wColShape[weightsBroad->rankOf()] = 1;
      NDArray* wbColPtr = weightsBroad->reshape('c', wColShape);
      weightsBroadFull = new NDArray(wbColPtr->tileToShape(dLdp->shapeInfo()));
      delete wbColPtr;
    } else {
      weightsBroadFull = new NDArray(weightsBroad->tileToShape(dLdp->shapeInfo()));
    }
  }

  auto excludeDims = ShapeUtils::evalDimsToExclude(dLdp->rankOf(), dimensions->size(), dimensions->data());

  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum"
      if (weights->isScalar() || weights->lengthOf() == 1) {
        NDArray eSum(E->dataType(), ctx);
        E->reduceNumber(reduce::Sum, &eSum);
        dLdw->assign(&eSum);
        double wVal = weights->e<double>(0);
        dLdp->applyScalar(scalar::Multiply, wVal, dLdp);
        dLdl->applyScalar(scalar::Multiply, wVal, dLdl);
      } else {
        dLdp->applyBroadcast(broadcast::Multiply, excludeDims, weightsBroad, dLdp);
        dLdl->applyBroadcast(broadcast::Multiply, excludeDims, weightsBroad, dLdl);

        if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E->reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else
          dLdw->assign(E);
      }
      break;
    }
    case 2: {  // 2 - "weighted_mean"
      double sumD;
      if (weights->isScalar()) {
        sumD = weights->e<double>(0) * E->lengthOf();
      } else {
        NDArray* sumPtr = weightsBroad->reduceNumber(reduce::Sum);
        sumD = sumPtr->e<double>(0);
        delete sumPtr;
      }

      if (sumD == 0.) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdl->assign(zero);
        dLdw->assign(zero);
      } else {
        if (weights->isScalar() || weights->lengthOf() == 1) {
          // dLdp *= weights / sumD
          double wVal = weights->e<double>(0);
          double wDivSum = wVal / sumD;
          dLdp->applyScalar(scalar::Multiply, wDivSum, dLdp);
          dLdl->applyScalar(scalar::Multiply, wDivSum, dLdl);
          double zero = 0.0;
          dLdw->assign(zero);
        } else {
          // Scale dLdp and dLdl by (w / sumW) row-wise.
          // weightsBroadFull has shape matching dLdp (tiled in the setup above).
          NDArray wbDivSum(dLdp->shapeInfo(), false, ctx);
          weightsBroadFull->applyScalar(scalar::Divide, sumD, &wbDivSum);
          dLdp->applyPairwiseTransform(pairwise::Multiply, &wbDivSum, dLdp);
          dLdl->applyPairwiseTransform(pairwise::Multiply, &wbDivSum, dLdl);

          // dLdw = (E * sumD - E * sum(weightsBroad)) / sumD^2
          // ETimesSum = E * sumD  (stack)
          NDArray ETimesSum(E->shapeInfo(), false, ctx);
          E->applyScalar(scalar::Multiply, sumD, &ETimesSum);

          // ETimesWeights = E * weightsBroad  (stack)
          NDArray ETimesWeights(E->shapeInfo(), false, ctx);
          E->applyPairwiseTransform(pairwise::Multiply, weightsBroad, &ETimesWeights);

          // ETimesWeightsSum: heap via no-target overload (avoids type-check issues)
          NDArray* ETimesWeightsSum = ETimesWeights.reduceNumber(reduce::Sum);
          double ewsVal = ETimesWeightsSum->e<double>(0);
          delete ETimesWeightsSum;

          // numerator = ETimesSum - ewsVal  (stack)
          // ETimesSum - ewsVal = ETimesSum + (-ewsVal)
          NDArray numerator(ETimesSum.shapeInfo(), false, ctx);
          ETimesSum.applyScalar(scalar::Add, -ewsVal, &numerator);

          double sumSquared = sumD * sumD;
          // result = numerator / sumSquared  (stack)
          NDArray result(numerator.shapeInfo(), false, ctx);
          numerator.applyScalar(scalar::Divide, sumSquared, &result);

          if (weights != weightsBroad) {
            std::vector<LongType> axesToReduceAlong =
                ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
            result.reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
          } else {
            dLdw->assign(&result);
          }
        }
      }
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights"
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E->lengthOf();
      } else {
        NDArray* countNonZero = weightsBroad->reduceNumber(reduce::CountNonZero);
        numOfNonZeroWeights = countNonZero->e<LongType>(0);
        delete countNonZero;
      }

      if (numOfNonZeroWeights == 0) {
        double zero = 0.0;
        dLdp->assign(zero);
        dLdl->assign(zero);
        dLdw->assign(zero);
      } else {
        if (weights->isScalar() || weights->lengthOf() == 1) {
          double wVal = weights->e<double>(0);
          double wDivNum = wVal / numOfNonZeroWeights;
          dLdp->applyScalar(scalar::Multiply, wDivNum, dLdp);
          dLdl->applyScalar(scalar::Multiply, wDivNum, dLdl);

          NDArray* eSum = E->reduceNumber(reduce::Sum);
          double eSumVal = eSum->e<double>(0);
          delete eSum;
          double result = eSumVal / double(numOfNonZeroWeights);
          dLdw->assign(result);
        } else {
          // Scale dLdp and dLdl by (w / numNonZeroWeights) row-wise.
          // weightsBroadFull has shape matching dLdp (tiled in the setup above).
          NDArray wbDivNum(dLdp->shapeInfo(), false, ctx);
          weightsBroadFull->applyScalar(scalar::Divide, double(numOfNonZeroWeights), &wbDivNum);
          dLdp->applyPairwiseTransform(pairwise::Multiply, &wbDivNum, dLdp);
          dLdl->applyPairwiseTransform(pairwise::Multiply, &wbDivNum, dLdl);

          if (weights != weightsBroad) {
            std::vector<LongType> axesToReduceAlong =
                ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
            E->reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
            dLdw->applyScalar(scalar::Divide, double(numOfNonZeroWeights), dLdw);
          } else {
            // eDivNum = E / numOfNonZeroWeights  (stack)
            NDArray eDivNum(E->shapeInfo(), false, ctx);
            E->applyScalar(scalar::Divide, double(numOfNonZeroWeights), &eDivNum);
            dLdw->assign(&eDivNum);
          }
        }
      }
      break;
    }
  }

  // Clean up heap-allocated intermediates
  delete maxLogits;
  delete sumExp;
  delete labelSum;
  delete maxLogits2;
  delete sumExp2;
  delete E;
  if (weightsBroad != weights) delete weightsBroad;
  if (weightsBroadFull != weights && weightsBroadFull != weightsBroad) delete weightsBroadFull;
  if (newLabels != cLabels) delete newLabels;
  delete cLabels;
  delete dimensions;
  delete excludeDims;

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
