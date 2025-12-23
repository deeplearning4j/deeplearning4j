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
// @author Yurii Shyrma (iuriish@yahoo.com), created on 22.11.2017
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_cosine_distance_loss)

#include <helpers/ShapeUtils.h>
#include <ops/declarable/CustomOperations.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(cosine_distance_loss, 3, 1, false, 0, 2) {
  auto predictions = INPUT_VARIABLE(0);
  auto weights = INPUT_VARIABLE(1);
  auto labels = INPUT_VARIABLE(2);

  auto output = OUTPUT_VARIABLE(0);

  int reductionMode =
      INT_ARG(0);        // 0 - "none"; 1 - "weighted_sum";  2 - "weighted_mean";  3 - "weighted_sum_by_nonzero_weights"
  int dim = INT_ARG(1);  // axis along which sum will be made
  if (dim < 0) dim += labels->rankOf();

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "COSINE_DISTANCE_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // regard 4 possible reduction modes below
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "COSINE_DISTANCE_LOSS OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, but "
               "got %i instead!",
               reductionMode);
  // input dimension can't be larger than labels/predictions/weights rank
  REQUIRE_TRUE(dim < labels->rankOf(), 0,
               "COSINE_DISTANCE_LOSS OP: input reduction dimension (got %i) must be < labels rank %i!", dim,
               labels->rankOf());

  if (!output->isScalar()) {
    // weights array can be single scalar or has the same shape as output, and must be broadcastable to output shape
    REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == output->rankOf(), 0,
                 "SOFTMAX_CROSS_ENTROPY_LOSS OP: weights array should be scalar or have the same rank as output array, "
                 "but got %i and %i correspondingly!",
                 weights->rankOf(), output->rankOf());
    // check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(*weights, *output), 0,
                 "COSINE_DISTANCE_LOSS OP: shapes of weights and output arrays should be broadcastable, but got "
                 "weights = %s and output = %s instead!",
                 ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(labels).c_str());
  }
  std::vector<LongType> dims;
  dims.push_back(dim);
  
  NDArray* predLabels = (*predictions) * (*labels);
  NDArray* dotProduct = predLabels->reduceAlongDimension(reduce::Sum, &dims, true);
  delete predLabels;
  
  NDArray* E = 1. - (*dotProduct);
  delete dotProduct;

  // perform weights broadcasting/tile to E if it is necessary
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(E))
    weightsBroad = new NDArray(weights->tileToShape(E->shapeInfo()));

  // multiply E on weights
  NDArray* EWeighted = (*E) * (*weightsBroad);

  switch (reductionMode) {
    case 0:  // 0 - "none", un-reduced weighted losses with the same shape as labels.
      output->assign(EWeighted);
      break;

    case 1: {  // 1 - "weighted_sum", output is scalar and equal to sum of all elements of E array
      auto* sumResult = EWeighted->reduceNumber(reduce::Sum);
      output->assign(sumResult);
      delete sumResult;
      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
               // all elements of weightsBroad array
      NDArray* sum;
      if (weights->isScalar()) {
        sum = (*weights) * EWeighted->lengthOf();
      } else {
        sum = weightsBroad->reduceNumber(reduce::Sum);
      }

      if (sum->e<double>(0) == 0.) {
        *output = 0.;
      } else {
        auto* sumE = EWeighted->reduceNumber(reduce::Sum);
        auto* result = (*sumE) / (*sum);
        output->assign(result);
        delete result;
        delete sumE;
      }
      delete sum;
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
               // array divided by number of non-zero weights
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = EWeighted->lengthOf();
      } else {
        auto* countResult = EWeighted->reduceNumber(reduce::CountNonZero);
        numOfNonZeroWeights = countResult->e<LongType>(0);
        delete countResult;
      }

      if (numOfNonZeroWeights == 0) {
        *output = 0.;
      } else {
        auto* sumE = EWeighted->reduceNumber(reduce::Sum);
        auto* result = (*sumE) / double(numOfNonZeroWeights);
        output->assign(result);
        delete result;
        delete sumE;
      }
      break;
    }
  }

  STORE_RESULT(*output);

  delete EWeighted;
  delete E;
  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(cosine_distance_loss) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(cosine_distance_loss) {
  // labels and predictions must have the same shapes
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  int dim = INT_ARG(1);
  if (dim < 0) dim += labelsShapeInfo[0];

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "COSINE_DISTANCE_LOSS OP: labels and predictions arrays must have the same shapes, but got %s and %s "
               "correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  // input dimension can't be larger than labels/predictions/weights rank
  REQUIRE_TRUE(dim < labelsShapeInfo[0], 0,
               "COSINE_DISTANCE_LOSS OP: input reduction dimension (got %i) must be < labels rank %i!", dim,
               labelsShapeInfo[0]);

  DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));

  // evaluate output shapeInfo
  LongType * outShapeInfo = nullptr;
  if (INT_ARG(0) != 0)  // in this case output is scalar
    outShapeInfo = ConstantShapeHelper::getInstance().scalarShapeInfo(outType);
  else {  // in this case output has the same shape as labels reduced  by dim axis

    std::vector<LongType> dimensions = {dim};
    outShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(predictionsShapeInfo), &dimensions, predictionsShapeInfo,
                                                   outType, true, false, block.getWorkspace());

    // weights array can be single scalar or has the same rank as output, and must be broadcastable to output
    REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(outShapeInfo), 0,
                 "COSINE_DISTANCE_LOSS OP: weights array should be scalar or have the same rank as output array, but "
                 "got %i and %i correspondingly!",
                 shape::rank(weightsShapeInfo), shape::rank(outShapeInfo));
    // check whether broadcast operation is possible for weights array
    REQUIRE_TRUE(
        shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, outShapeInfo), 0,
        "COSINE_DISTANCE_LOSS OP: shapes of weights and output arrays should be broadcastable, but got weights = %s "
        "and output = %s instead!",
        ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(outShapeInfo).c_str());
  }

  return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(cosine_distance_loss_grad, 3, 3, false, 0, 2) {
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

  int dim = INT_ARG(1);  // axis along which sum will be made
  if (dim < 0) dim += labels->rankOf();

  std::vector<LongType> dimensions = {dim};

  // input validation
  REQUIRE_TRUE(labels->isSameShape(predictions), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labels).c_str(), ShapeUtils::shapeAsString(predictions).c_str());
  // only 4 possible reduction modes exist
  REQUIRE_TRUE(reductionMode == 0 || reductionMode == 1 || reductionMode == 2 || reductionMode == 3, 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: reduction mode value is not acceptable, possible values are 0, 1, 2, 3, "
               "but got %i instead!",
               reductionMode);
  auto lossShapeInfo = ShapeUtils::evalReduceShapeInfo(predictions->ordering(), &dimensions, predictions->shapeInfo(),
                                                       true, false, block.getWorkspace());
  // weights array can be single scalar or has the same shape as loss, and must be broadcastable to loss shape
  REQUIRE_TRUE(weights->isScalar() || weights->rankOf() == shape::rank(lossShapeInfo), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: weights array should be scalar or have the same rank as loss array, but "
               "got %i and %i correspondingly!",
               weights->rankOf(), shape::rank(lossShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(weights->isScalar() || ShapeUtils::areShapesBroadcastable(weights->shapeInfo(), lossShapeInfo), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: shapes of weights and loss arrays should be broadcastable, but got "
               "weights = %s and loss = %s instead!",
               ShapeUtils::shapeAsString(weights).c_str(), ShapeUtils::shapeAsString(lossShapeInfo).c_str());
  // input dimension can't be larger than labels/predictions/weights rank
  REQUIRE_TRUE(dim < labels->rankOf(), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: input reduction dimension (got %i) must be < labels rank %i!", dim,
               labels->rankOf());

  std::vector<LongType> dims;
  dims.push_back(dim);
  
  NDArray* predLabels = (*predictions) * (*labels);
  NDArray* dotProduct = predLabels->reduceAlongDimension(reduce::Sum, &dims, true);
  delete predLabels;
  
  NDArray* E = 1. - (*dotProduct);
  delete dotProduct;

  // perform weights broadcasting/tile to E if it is necessary
  auto weightsBroad = weights;
  if (!weights->isScalar() && !weights->isSameShape(E))
    weightsBroad = new NDArray(weights->tileToShape(E->shapeInfo()));

  NDArray negLabels = -(*labels);
  NDArray negPreds = -(*predictions);
  dLdp->assign(&negLabels);
  dLdl->assign(&negPreds);

  switch (reductionMode) {
    case 1: {  // 1 - "none" and "weighted_sum", output is scalar and equal to sum of all elements of E array

      NDArray* dLdpWeighted = (*dLdp) * (*weightsBroad);
      dLdp->assign(dLdpWeighted);
      delete dLdpWeighted;
      
      NDArray* dLdlWeighted = (*dLdl) * (*weightsBroad);
      dLdl->assign(dLdlWeighted);
      delete dLdlWeighted;

      if (weights->isScalar() || weights->lengthOf() == 1) {
        auto* sumE = E->reduceNumber(reduce::Sum);
        dLdw->assign(sumE);
        delete sumE;
      } else {
        if (weights != weightsBroad) {
          std::vector<LongType> axesToReduceAlong =
              ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
          E->reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
        } else
          dLdw->assign(E);
      }

      break;
    }
    case 2: {  // 2 - "weighted_mean", output is scalar and equal to sum of all elements of E array divided by sum of
               // all elements of weightsBroad array
      NDArray* sum;
      if (weights->isScalar()) {
        sum = (*weights) * E->lengthOf();
      } else {
        sum = weightsBroad->reduceNumber(reduce::Sum);
      }

      if (sum->e<double>(0) == 0.) {
        *dLdp = 0.;
        *dLdl = 0.;
        *dLdw = 0.;
      } else {
        NDArray* temp = (*weightsBroad) / (*sum);
        NDArray* dLdpResult = (*dLdp) * (*temp);
        dLdp->assign(dLdpResult);
        delete dLdpResult;
        
        NDArray* dLdlResult = (*dLdl) * (*temp);
        dLdl->assign(dLdlResult);
        delete dLdlResult;
        delete temp;

        if (weights->isScalar() || weights->lengthOf() == 1) {
          *dLdw = 0.;
        } else {
          if (weights != weightsBroad) {
            std::vector<LongType> axesToReduceAlong =
                ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
            NDArray* EWeighted = (*E) * (*weightsBroad);
            NDArray* EWeightedSum = EWeighted->reduceNumber(reduce::Sum);
            delete EWeighted;
            NDArray* ESum = (*E) * (*sum);
            NDArray* numerator = (*ESum) - (*EWeightedSum);
            delete ESum;
            delete EWeightedSum;
            NDArray* sumSquared = (*sum) * (*sum);
            NDArray* gradTemp = (*numerator) / (*sumSquared);
            delete numerator;
            delete sumSquared;
            gradTemp->reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
            delete gradTemp;
          } else {
            NDArray* EWeighted = (*E) * (*weightsBroad);
            NDArray* EWeightedSum = EWeighted->reduceNumber(reduce::Sum);
            delete EWeighted;
            NDArray* ESum = (*E) * (*sum);
            NDArray* numerator = (*ESum) - (*EWeightedSum);
            delete ESum;
            delete EWeightedSum;
            NDArray* sumSquared = (*sum) * (*sum);
            NDArray* gradTemp = (*numerator) / (*sumSquared);
            delete numerator;
            delete sumSquared;
            dLdw->assign(gradTemp);
            delete gradTemp;
          }
        }
      }
      delete sum;
      break;
    }
    case 3: {  // 3 - "weighted_sum_by_nonzero_weights", output is scalar and equal to scalar sum of all elements of E
               // array divided by number of non-zero weights
      LongType numOfNonZeroWeights = 0;
      if (weights->isScalar()) {
        if (weights->e<double>(0) != 0.) numOfNonZeroWeights = E->lengthOf();
      } else {
        auto* countResult = weightsBroad->reduceNumber(reduce::CountNonZero);
        numOfNonZeroWeights = countResult->e<LongType>(0);
        delete countResult;
      }

      if (numOfNonZeroWeights == 0) {
        *dLdp = 0.;
        *dLdl = 0.;
        *dLdw = 0.;
      } else {
        NDArray* temp = (*weightsBroad) / numOfNonZeroWeights;
        NDArray* dLdpResult = (*dLdp) * (*temp);
        dLdp->assign(dLdpResult);
        delete dLdpResult;
        
        NDArray* dLdlResult = (*dLdl) * (*temp);
        dLdl->assign(dLdlResult);
        delete dLdlResult;
        delete temp;

        if (weights->isScalar() || weights->lengthOf() == 1) {
          auto* sumE = E->reduceNumber(reduce::Sum);
          auto* result = (*sumE) / numOfNonZeroWeights;
          dLdw->assign(result);
          delete result;
          delete sumE;
        } else {
          if (weights != weightsBroad) {
            std::vector<LongType> axesToReduceAlong =
                ShapeUtils::evalBroadcastBackwardAxis(weights->shapeInfo(), weightsBroad->shapeInfo());
            E->reduceAlongDimension(reduce::Sum, dLdw, &axesToReduceAlong, true);
            NDArray* dLdwResult = (*dLdw) / numOfNonZeroWeights;
            dLdw->assign(dLdwResult);
            delete dLdwResult;
          } else {
            NDArray* result = (*E) / numOfNonZeroWeights;
            dLdw->assign(result);
            delete result;
          }
        }
      }
      break;
    }
  }

  delete E;
  if (weightsBroad != weights) delete weightsBroad;

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
DECLARE_TYPES(cosine_distance_loss_grad) {
  getOpDescriptor()->setAllowedInputTypes(ANY)->setAllowedOutputTypes({ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(cosine_distance_loss_grad) {
  /// labels and predictions must have the same shapes
  auto predictionsShapeInfo = inputShape->at(0);
  auto weightsShapeInfo = inputShape->at(1);
  auto labelsShapeInfo = inputShape->at(2);

  int dim = INT_ARG(1);
  if (dim < 0) dim += labelsShapeInfo[0];

  std::vector<LongType> dimensions = {dim};

  // labels and predictions must have the same shapes
  REQUIRE_TRUE(shape::shapeEquals(labelsShapeInfo, predictionsShapeInfo), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: labels and predictions arrays must have the same shapes, but got %s and "
               "%s correspondingly !",
               ShapeUtils::shapeAsString(labelsShapeInfo).c_str(),
               ShapeUtils::shapeAsString(predictionsShapeInfo).c_str());
  auto lossShapeInfo = ShapeUtils::evalReduceShapeInfo(shape::order(predictionsShapeInfo), &dimensions,
                                                       predictionsShapeInfo, true, false, block.getWorkspace());
  // weights array can be single scalar or has the same rank as loss, and must be broadcastable to loss
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || shape::rank(weightsShapeInfo) == shape::rank(lossShapeInfo), 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: weights array should be scalar or have the same rank as loss array, but "
               "got %i and %i correspondingly!",
               shape::rank(weightsShapeInfo), shape::rank(lossShapeInfo));
  // check whether broadcast operation is possible for weights array
  REQUIRE_TRUE(shape::isScalar(weightsShapeInfo) || ShapeUtils::areShapesBroadcastable(weightsShapeInfo, lossShapeInfo),
               0,
               "COSINE_DISTANCE_LOSS_GRAD OP: shapes of weights and loss arrays should be broadcastable, but got "
               "weights = %s and loss = %s instead!",
               ShapeUtils::shapeAsString(weightsShapeInfo).c_str(), ShapeUtils::shapeAsString(lossShapeInfo).c_str());
  // input dimension can't be larger than labels/predictions/weights rank
  REQUIRE_TRUE(dim < labelsShapeInfo[0], 0,
               "COSINE_DISTANCE_LOSS_GRAD OP: input reduction dimension (got %i) must be < labels rank %i!", dim,
               labelsShapeInfo[0]);

  auto outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(predictionsShapeInfo));

  auto dLdpShapeInfo = ShapeBuilders::copyShapeInfoAndType(predictionsShapeInfo, outType, false, block.getWorkspace());
  auto dLdwShapeInfo = ShapeBuilders::copyShapeInfoAndType(weightsShapeInfo, outType, false, block.getWorkspace());
  auto dLdlShapeInfo = ShapeBuilders::copyShapeInfoAndType(labelsShapeInfo, outType, false, block.getWorkspace());

  return SHAPELIST(CONSTANT(dLdpShapeInfo), CONSTANT(dLdwShapeInfo), CONSTANT(dLdlShapeInfo));
}

}  // namespace ops
}  // namespace sd

#endif
