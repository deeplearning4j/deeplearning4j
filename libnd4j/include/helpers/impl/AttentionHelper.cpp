/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Paul Dubs
// @author Adam Gibson
//

#ifndef LIBND4J_ATTENTIONHELPER_CPP
#define LIBND4J_ATTENTIONHELPER_CPP
#include "../AttentionHelper.h"
#include <indexing/NDIndexUtils.h>
#include <helpers/AttentionHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <array/ResultSet.h>
#include <ops/declarable/helpers/batched_gemm.h>
#if NOT_EXCLUDED(OP_multi_head_dot_product_attention)

namespace sd {

NDArray AttentionHelper::multiHeadProject(NDArray *input, NDArray *projectionMatrix,
                                          LaunchContext *context) {
  auto miniBatchSize = input->sizeAt(0);
  auto seqLength = input->sizeAt(2);
  auto numHeads = projectionMatrix->sizeAt(0);
  auto projectedSize = projectionMatrix->sizeAt(1);

  std::vector<sd::LongType> epsPermVec = {1, 0,2};
  auto inputPerm = input->permute(epsPermVec, false, false);  //[batch, nIn, timeSteps] -> [nIn, batch, timeSteps]
  std::vector<sd::LongType> inputPermShape = {input->sizeAt(1), (miniBatchSize * seqLength)};
  auto inputPrep = inputPerm->reshape('c', inputPermShape);  //[nIn, batch*timeSteps]
  std::vector<sd::LongType> projectionMatrixShape = {numHeads * projectionMatrix->sizeAt(1), projectionMatrix->sizeAt(2)};
  auto projectionPrep = projectionMatrix->reshape(
      'c',
      projectionMatrixShape);  //[nHeads, hS, nIn] -> [nHeads*hS, nIn]

  std::vector<LongType> projectedShape = {numHeads * projectionMatrix->sizeAt(1), (miniBatchSize * seqLength)};
  NDArray projected('c',projectedShape, input->dataType(),
                    context);  //[nHeads*hS, batch*timeSteps]
  ops::matmul mmul;
  mmul.execute({projectionPrep, inputPrep}, {&projected});

  projected.reshapei({numHeads, projectedSize, miniBatchSize, seqLength});
  projected.permutei({2, 0, 1, 3}, false, false);  //[minibatch, numHeads, projectedSize, seqLength]

  return projected;
}


/**
 * @param shape
 * @return
 */
NDArray * AttentionHelper::lowerTriangularMask(std::vector<LongType> *shape) {
  // Optimized: use matrix_band_part directly instead of cumsum + greater_equal
  // matrix_band_part with (-1, 0) keeps the lower triangular part
  ops::matrix_band_part matrixBandPart;
  auto ones = NDArrayFactory::valueOf(*shape, 1, 'c');  // Creates INT32 array filled with 1s
  auto lower = matrixBandPart.evaluate({ones}, {}, {-1, 0});
  auto ret = lower.at(0)->cast(BOOL);
  lower.setNonRemovable();
  delete ones;
  return ret;
}

/**
 * @param query
 * @param value
 * @return
 */
NDArray *AttentionHelper::computeCasualMask(NDArray *query, NDArray *value, bool multiHead) {
  if(multiHead) {
    auto qSeqLength = query->sizeAt(1);
    auto vSeqLength = value != nullptr ? value->sizeAt(1) : qSeqLength;
    ops::matrix_band_part matrixBandPart;
    auto ones = NDArrayFactory::create('c',{1,qSeqLength,vSeqLength}, INT32);
    int assignVal = 1;
    ones->assign(assignVal);
    auto lower = matrixBandPart.evaluate({ones},{},{-1,0});
    auto ret = lower.at(0)->cast(BOOL);
    delete ones;
    return ret;

  } else {
    std::vector<LongType> causalMaskShape2;
    causalMaskShape2.push_back(query->sizeAt(0));
    //4d
    if(query->rankOf() > 3)
      causalMaskShape2.push_back(query->sizeAt(1));

    causalMaskShape2.push_back(query->sizeAt(-2));
    causalMaskShape2.push_back(value->sizeAt(-2));

    auto ret  = lowerTriangularMask(&causalMaskShape2);
    return ret;

  }

}


/**
 * @param query
 * @param value
 * @param attentionMask
 * @param useCausalMask
 * @return
 */
NDArray *AttentionHelper::computeAttentionMask(NDArray *query, NDArray *value, NDArray *queryMask, NDArray *valueMask,
                                               NDArray *attentionMask, bool useCausalMask) {
  auto internalQueryMask = queryMask;
  auto internalValueMask = valueMask;
  NDArray *autoMask = nullptr;
  ops::create_view createView;
  ops::boolean_and booleanAnd;
  auto all = NDIndexUtils::createAll();
  auto newAxis = NDIndexUtils::createNewAxis();

  // Track whether we created casted arrays (need to delete them later)
  bool castedQueryMask = false;
  bool castedValueMask = false;

  // Store ResultSets to keep arrays alive - use setNonRemovable so returned pointers remain valid
  ResultSet queryViewResult;
  ResultSet valueViewResult;
  ResultSet boolAndResult1;
  ResultSet boolAndResult2;
  ResultSet boolAndResult3;

  if (internalQueryMask != nullptr && !internalQueryMask->isEmpty()) {
    if(queryMask->dataType() != BOOL) {
      internalQueryMask = queryMask->cast(BOOL);
      castedQueryMask = true;
    }
    queryViewResult = createView.evaluate({internalQueryMask, all, all, newAxis});
    queryViewResult.setNonRemovable();
    autoMask = queryViewResult.at(0);
  }

  if (valueMask != nullptr && !valueMask->isEmpty()) {
    if(valueMask->dataType() != BOOL) {
      internalValueMask = valueMask->cast(BOOL);
      castedValueMask = true;
    }
    valueViewResult = createView.evaluate({internalValueMask, all, newAxis, all});
    valueViewResult.setNonRemovable();
    auto mask = valueViewResult.at(0);
    if (autoMask == nullptr || autoMask->isEmpty()) {
      autoMask = mask;
    } else {
      boolAndResult1 = booleanAnd.evaluate({autoMask, mask});
      boolAndResult1.setNonRemovable();
      autoMask = boolAndResult1.at(0);
    }
  }

  if (useCausalMask) {
    auto mask = computeCasualMask(query, value, false);
    if (autoMask == nullptr) {
      autoMask = mask;
    } else {
      boolAndResult2 = booleanAnd.evaluate({autoMask, mask});
      boolAndResult2.setNonRemovable();
      autoMask = boolAndResult2.at(0);
    }
  }

  // Always clean up the index objects
  delete all;
  delete newAxis;

  // Clean up casted arrays
  if(castedQueryMask && internalQueryMask != nullptr) {
    delete internalQueryMask;
  }
  if(castedValueMask && internalValueMask != nullptr) {
    delete internalValueMask;
  }

  if (autoMask != nullptr && !autoMask->isEmpty()) {
    if (attentionMask == nullptr || attentionMask->isEmpty()) {
      return autoMask;
    } else {
      boolAndResult3 = booleanAnd.evaluate({attentionMask, autoMask});
      boolAndResult3.setNonRemovable();
      auto ret = boolAndResult3.at(0);
      return ret;
    }
  }

  return autoMask;
}

NDArray * AttentionHelper::mergeMasks(NDArray *x, NDArray *y) {
  if(x == nullptr || x->isEmpty()) {
    return y;
  }

  if (y == nullptr || y->isEmpty()) {
    return x;
  }

  // Optimized: use multiplication instead of boolean_and.evaluate()
  // For boolean/0-1 masks: x AND y = x * y
  // This avoids the evaluate() overhead and ResultSet allocation
  auto result = (*x) * (*y);
  return result;
}

void AttentionHelper::applyAttentionScores(NDArray *scores, NDArray *value, NDArray *scoresMask,
                                           double dropout, int randomSeed, NDArray *applyScoresOut, NDArray *attentionLogits,
                                           NDArray *dropoutMask) {
  ops::softmax softmax;
  ops::dropout dropoutOp;
  ops::matmul matmul;

  int softmaxDim = -1;
  if (scoresMask != nullptr && !scoresMask->isEmpty()) {
    REQUIRE_TRUE(scoresMask->sizeAt(-2) == 1 || scoresMask->sizeAt(-2) == scores->sizeAt(-2),0,
                 "Scores mask must be either broadcastable or equal to scores shape. scores size at -2: was: %i scores size at -2 was: %i",scoresMask->sizeAt(-2),scores->sizeAt(-2));

    REQUIRE_TRUE(scoresMask->sizeAt(-1) == scores->sizeAt(-1),0,
                 "Scores mask must be either broadcastable or equal to scores shape. scores size at -1: was: %i scores size at -1 was: %i",scoresMask->sizeAt(-1),scores->sizeAt(-1));

    // Optimized mask application: instead of boolean_not + multiply + subtract,
    // we use arithmetic: logits -= (1 - mask) * largeVal = logits - largeVal + mask * largeVal
    // This avoids the boolean_not.evaluate() call and reduces allocations from 4 to 2.

    // Cast mask to scores datatype (ensures 0.0/1.0 values)
    NDArray* numericMask;
    bool needsDeleteMask = false;

    if(scoresMask->dataType() == scores->dataType()) {
      // Already correct type - if it's a float type, assume 0/1 values
      // If it was BOOL, it would have been caught by the else branch
      numericMask = scoresMask;
    } else {
      // Cast to scores datatype (BOOL->FLOAT gives 0.0/1.0)
      numericMask = scoresMask->cast(scores->dataType());
      needsDeleteMask = true;
    }

    // Use appropriate large value for masking
    float largeVal = (attentionLogits->dataType() == BFLOAT16) ? 65504.0f : 1.0e9f;

    // Step 1: Subtract largeVal from all positions
    attentionLogits->applyScalar(sd::scalar::Subtract, largeVal, attentionLogits);

    // Step 2: Add back largeVal * mask (restores unmasked positions where mask=1)
    auto restore = largeVal * (*numericMask);
    *attentionLogits += *restore;
    delete restore;

    if(needsDeleteMask) {
      delete numericMask;
    }
  }



  softmax.execute({attentionLogits},{scores},{},{softmaxDim});
  auto weights = scores;

  if (dropout > 0) {
    dropoutOp.execute({weights},{weights,dropoutMask},{dropout},{randomSeed});
  }

  //batch size, tq tv
  //batch size tv dim
  //output: batch size, tq dim
  matmul.execute({weights,value},{applyScoresOut});

}

void AttentionHelper::dotProductAttentionBpHelper(NDArray *query, NDArray *key, NDArray *values,
                                                  double scale,
                                                  NDArray *dLdq, NDArray *dLdk, NDArray *dLdv, NDArray *eps, LongType dropoutSeed, NDArray *qMask, NDArray *vMask, bool useCausalMask, double dropout, bool training,
                                                  NDArray *attentionScoresWeights, NDArray *attentionLogits,
                                                  NDArray *dropoutMask) {
  ops::matmul_bp matMulBp;
  ops::softmax_bp softmaxBp;
  NDArray dldW(attentionScoresWeights->shapeInfo());
  NDArray dldS(attentionScoresWeights->shapeInfo());
  NDArray * mask = nullptr;
  NDArray *causalPointer = nullptr;

  if(useCausalMask) {
    std::vector<LongType> causalMaskShape2;
    causalMaskShape2.push_back(attentionLogits->sizeAt(0));
    //4d
    if(attentionLogits->rankOf() > 3)
      causalMaskShape2.push_back(attentionLogits->sizeAt(1));

    for(int i = attentionLogits->rankOf() - 2; i < attentionLogits->rankOf(); i++) {
      causalMaskShape2.push_back(attentionLogits->sizeAt(i));
    }
    causalPointer = lowerTriangularMask(&causalMaskShape2);
  }

  mask = mergeMasks(vMask,causalPointer);



  matMulBp.execute({attentionScoresWeights,values,eps},{&dldW,dLdv},{},{});
  if(dropout > 0.0 && training) {
    ops::dropout_bp dropoutOp;
    auto inputs = {attentionScoresWeights,dropoutMask,&dldW};
    dropoutOp.execute(inputs,{&dldW},{dropout},{dropoutSeed},{false});
  }

  softmaxBp.execute({attentionLogits,&dldW,attentionScoresWeights},{&dldS},{},{-1},{});

  if(scale != 0.0 && scale != 1.0) {
    // Use applyScalar instead of *= to avoid type mismatch between FLOAT arrays and double scalar
    dldS.applyScalar(sd::scalar::Multiply, scale, &dldS);
  }

  if(mask != nullptr && !mask->isEmpty()) {
    auto maskCast = mask->cast(query->dataType());
    dldS *= *maskCast;
    // Only delete maskCast if it's a different array than mask (i.e., cast created a new array)
    if(maskCast != mask) {
      delete maskCast;
    }
  }

  matMulBp.execute({query,key,&dldS},{dLdq,dLdk},{},{0,1,0});
}




/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
void AttentionHelper::attentionBpHelper(NDArray *query, NDArray *key, NDArray *values, double scale, NDArray *dLdq,
                                        NDArray *dLdk, NDArray *dLdv, NDArray *eps,
                                        LongType dropoutSeed,
                                        NDArray *qMask, NDArray *vMask,
                                        bool useCausalMask, double dropout, bool training, NDArray *attentionScoresOut,
                                        NDArray *attentionScoresWeights,
                                        NDArray *attentionScoresLogits,
                                        NDArray *dropoutMask) {
  dotProductAttentionBpHelper(query, key, values, scale, dLdq, dLdk, dLdv, eps, dropoutSeed, qMask, vMask,
                              useCausalMask, dropout, training, attentionScoresWeights, attentionScoresLogits,
                              dropoutMask);


}

/**
   *
   * @param query
   * @param key
   * @param scoreMode
   * @param scale
   * @return
 */
void AttentionHelper::attentionHelper(NDArray *query, NDArray *key, double scale, NDArray *attentionLogits) {
  ops::matmul matmul3;
  matmul3.execute({query,key},{attentionLogits},{},{0,1});
  if(scale != 0.0 && scale != 1.0) {
    // Use applyScalar instead of *= to avoid type mismatch between FLOAT arrays and double scalar
    attentionLogits->applyScalar(sd::scalar::Multiply, scale, attentionLogits);
  }
  // Note: No clipping needed here - softmax already handles numerical stability by:
  // 1. Subtracting max before exp()
  // 2. Clamping differences to [-88, 88] to prevent overflow
}




/**
 * @param inputs
 * @param mask
 * @param training
 * @param returnAttentionScores
 * @param useCausalMask
 */
void AttentionHelper::doAttentionBp(std::vector<NDArray *> &inputs, std::vector<NDArray *> &masks, bool training,
                                    bool useCausalMask, double dropout, double scale, std::vector<NDArray *> outputs,
                                    LongType dropoutSeed) {
  auto q = inputs[0];
  auto v = inputs[1];
  auto k = inputs[2];
  auto attentionScoresOut = inputs[3];
  auto attentionScoresWeights = inputs[4];
  auto attentionScoresLogits = inputs[5];
  auto eps = inputs[6];

  auto dropoutMask = inputs.size() > 7 ? inputs[7] : inputs[7];

  ops::expand_dims expandDims;
  ops::ones_as onesAs;
  ops::shape_of shapeOf;
  ops::concat concatOp;
  ops::create_view createView;
  auto qMask = masks.size() > 0 ? masks[0] : nullptr;
  auto vMask = masks.size() > 1 ? masks[1] : nullptr;
  auto vmaskInternal = vMask;
  auto qMaskInternal = qMask;

  // Store ResultSets to keep expanded masks alive for the duration of this function
  ResultSet vMaskExpandResult;
  ResultSet qMaskExpandResult;
  NDArray *squeezedVMask = nullptr;  // Track squeezed mask for cleanup

  if(vMask != nullptr && !vMask->isEmpty() && vMask->rankOf() < v->rankOf()) {
    // Insert dimension before the last one: for [batch, Tv] -> [batch, 1, Tv]
    // Using vMask->rankOf() - 1 gives position 1 for rank 2, position 2 for rank 3
    int expandDim = static_cast<int>(vMask->rankOf()) - 1;
    vMaskExpandResult = expandDims.evaluate({vMask},{},{expandDim});
    vmaskInternal = vMaskExpandResult.at(0);
  } else if(vMask != nullptr && !vMask->isEmpty() && vMask->rankOf() > attentionScoresLogits->rankOf()) {
    // Squeeze extra leading dimensions from mask to match attention logits rank
    // e.g., mask [1, 1, 1, 512] with attention logits [12, 512, 512] -> squeeze to [1, 512] or [512]
    auto targetRank = attentionScoresLogits->rankOf();
    std::vector<sd::LongType> newShape;

    // Build new shape by skipping leading 1s until we reach target rank
    int skipDims = static_cast<int>(vMask->rankOf()) - static_cast<int>(targetRank);
    for(int i = 0; i < vMask->rankOf(); i++) {
      if(i < skipDims && vMask->sizeAt(i) == 1) {
        continue;
      }
      newShape.push_back(vMask->sizeAt(i));
    }

    if(newShape.size() < static_cast<size_t>(vMask->rankOf())) {
      squeezedVMask = vMask->reshape('c', newShape);
      vmaskInternal = squeezedVMask;
    }
  }

  if(qMask != nullptr && !qMask->isEmpty()) {
    qMaskExpandResult = expandDims.evaluate({qMaskInternal},{},{-1});
    qMaskInternal = qMaskExpandResult.at(0);
  }


  auto dLdq = outputs[0];
  auto dLdv = outputs[1];
  auto dLdk = outputs[2];
  attentionBpHelper(q, k, v, scale, dLdq, dLdk, dLdv, eps, dropoutSeed, qMaskInternal, vmaskInternal, useCausalMask,
                    dropout, training, attentionScoresOut, attentionScoresWeights, attentionScoresLogits, dropoutMask);

  // Clean up squeezed mask if we created one
  if(squeezedVMask != nullptr) {
    delete squeezedVMask;
  }
}


/**
 * @param inputs
 * @param mask
 * @param training
 * @param returnAttentionScores
 * @param useCausalMask
 */
void AttentionHelper::doAttention(std::vector<NDArray *> &inputs, std::vector<NDArray *> &masks, bool training,
                                  bool useCausalMask, double dropout, double scale, NDArray *attentionScores,
                                  int dropoutSeed, NDArray *applyScoresOut, NDArray *attentionLogits,
                                  NDArray *dropoutMask) {
  auto q = inputs[0];
  auto v = inputs[1];
  auto k = inputs.size() > 2 ? inputs[2]  : v;
  auto concatWeights = inputs.size() > 3 ? inputs[3] : nullptr;

  ops::expand_dims expandDims;
  ops::ones_as onesAs;
  ops::shape_of shapeOf;
  ops::concat concatOp;
  ops::create_view createView;
  auto qMask = masks.size() > 0 ? masks[0] : nullptr;
  auto vMask = masks.size() > 1 ? masks[1] : nullptr;
  auto vmaskInternal = vMask;
  auto qMaskInternal = qMask;

  // Store ResultSets to keep expanded masks alive for the duration of this function
  ResultSet vMaskExpandResult;
  ResultSet qMaskExpandResult;

  NDArray *casualPointer = nullptr;
  NDArray *squeezedVMask = nullptr;  // Track squeezed mask for cleanup
  //inputs: query and value
  //shape: batch_size Tq dim (batch_size Tv dim)
  //note this does not apply softmax yet, we are just computing logits here
  attentionHelper(q, k, scale, attentionLogits);

  if(vMask != nullptr && !vMask->isEmpty() && vMask->rankOf() < v->rankOf()) {
    // Insert dimension before the last one: for [batch, Tv] -> [batch, 1, Tv]
    // Using vMask->rankOf() - 1 gives position 1 for rank 2, position 2 for rank 3
    int expandDim = static_cast<int>(vMask->rankOf()) - 1;
    vMaskExpandResult = expandDims.evaluate({vMask},{},{expandDim});
    vmaskInternal = vMaskExpandResult.at(0);
  } else if(vMask != nullptr && !vMask->isEmpty() && vMask->rankOf() > attentionLogits->rankOf()) {
    // Squeeze extra leading dimensions from mask to match attention logits rank
    // e.g., mask [1, 1, 1, 512] with attention logits [12, 512, 512] -> squeeze to [1, 512] or [512]
    // We squeeze from the front until ranks match or we can't squeeze anymore
    auto targetRank = attentionLogits->rankOf();
    std::vector<sd::LongType> newShape;

    // Build new shape by skipping leading 1s until we reach target rank
    int skipDims = static_cast<int>(vMask->rankOf()) - static_cast<int>(targetRank);
    for(int i = 0; i < vMask->rankOf(); i++) {
      if(i < skipDims && vMask->sizeAt(i) == 1) {
        // Skip this dimension (squeeze it)
        continue;
      }
      newShape.push_back(vMask->sizeAt(i));
    }

    // If we successfully reduced dimensions, reshape
    if(newShape.size() < static_cast<size_t>(vMask->rankOf())) {
      squeezedVMask = vMask->reshape('c', newShape);
      vmaskInternal = squeezedVMask;
    }
  }

  if(useCausalMask) {
    std::vector<LongType> causalMaskShape2;
    causalMaskShape2.push_back(attentionScores->sizeAt(0));
    //4d
    if(attentionScores->rankOf() > 3)
      causalMaskShape2.push_back(attentionScores->sizeAt(1));

    for(int i = attentionScores->rankOf() - 2; i < attentionScores->rankOf(); i++) {
      causalMaskShape2.push_back(attentionScores->sizeAt(i));
    }
    casualPointer = lowerTriangularMask(&causalMaskShape2);
  }

  auto scoresMask = mergeMasks(vmaskInternal,casualPointer);

  //compute actual softmax now
  if(training) {
    applyAttentionScores(attentionScores, v, scoresMask, dropout, dropoutSeed, applyScoresOut, attentionLogits,
                         dropoutMask);
  } else {
    applyAttentionScores(attentionScores, v, scoresMask, 0, dropoutSeed, applyScoresOut, attentionLogits, dropoutMask);
  }
  //inputs: scores:  batch size tq tv value:batch size, tv,dim scoresmask: batch size 1 tv or batch size tq tv
  if(qMask != nullptr && !qMask->isEmpty()) {
    qMaskExpandResult = expandDims.evaluate({qMaskInternal},{},{-1});
    qMaskInternal = qMaskExpandResult.at(0);
    auto casted = qMaskInternal->cast(attentionScores->dataType());
    *attentionScores *= *casted;
    // Clean up casted array if it's different from qMaskInternal
    if(casted != qMaskInternal) {
      delete casted;
    }
  }

  // Clean up squeezed mask if we created one
  if(squeezedVMask != nullptr) {
    delete squeezedVMask;
  }
}


void AttentionHelper::multiHeadProjectBp(NDArray *input, NDArray *projectionMatrix,
                                         NDArray *eps,
                                         NDArray *dLdInput, NDArray *dLdProjectionMatrix, LaunchContext *context) {
  auto miniBatchSize = input->sizeAt(0);
  auto seqLength = input->sizeAt(2);
  auto numHeads = projectionMatrix->sizeAt(0);
  auto projectedSize = projectionMatrix->sizeAt(1);

  std::vector<sd::LongType> epsPermVec = {1, 2, 0, 3};
  auto epsPerm = eps->permute(epsPermVec, false, false);
  std::vector<sd::LongType> epsReshapeVec = {numHeads * projectedSize, miniBatchSize * seqLength};
  auto epsReshaped = epsPerm->reshape('c', epsReshapeVec);

  std::vector<sd::LongType> inputPermVec = {1, 0, 2};
  auto inputPerm = input->permute(inputPermVec, false, false);
  std::vector<sd::LongType> inputPermShape = {input->sizeAt(1), miniBatchSize * seqLength};
  auto inputPrep = inputPerm->reshape('c',inputPermShape,false);
  std::vector<sd::LongType> projectionMatrixShape = {numHeads * projectionMatrix->sizeAt(1), projectionMatrix->sizeAt(2)};
  auto projectionPrep =
      projectionMatrix->reshape('c', projectionMatrixShape);

  ops::matmul_bp mmulBp;
  NDArray dLdProjectionPrep(projectionPrep->shapeInfo(), false, context);
  NDArray dLdInputPrep(inputPrep->shapeInfo(), false, context);
  mmulBp.execute({projectionPrep, inputPrep, epsReshaped}, std::vector<NDArray *>{&dLdProjectionPrep, &dLdInputPrep},
                 {}, {}, {});

  dLdProjectionPrep.reshapei({numHeads, projectionMatrix->sizeAt(1), projectionMatrix->sizeAt(2)});
  dLdProjectionMatrix->assign(&dLdProjectionPrep);

  dLdInputPrep.reshapei({input->sizeAt(1), miniBatchSize, seqLength});
  dLdInputPrep.permutei({1, 0, 2}, false, false);
  dLdInput->assign(&dLdInputPrep);

  delete epsReshaped;
  delete projectionPrep;

}
}  // namespace sd
#endif

#endif
