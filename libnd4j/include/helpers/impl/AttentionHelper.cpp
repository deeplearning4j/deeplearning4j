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
  auto inputPermDup = inputPerm->dup('c');  // force contiguous before reshape
  std::vector<sd::LongType> inputPermShape = {input->sizeAt(1), (miniBatchSize * seqLength)};
  auto inputPrep = inputPermDup->reshape('c', inputPermShape);  //[nIn, batch*timeSteps]
  std::vector<sd::LongType> projectionMatrixShape = {numHeads * projectionMatrix->sizeAt(1), projectionMatrix->sizeAt(2)};
  auto projectionPrep = projectionMatrix->reshape(
      'c',
      projectionMatrixShape);  //[nHeads, hS, nIn] -> [nHeads*hS, nIn]

  std::vector<LongType> projectedShape = {numHeads * projectionMatrix->sizeAt(1), (miniBatchSize * seqLength)};
  NDArray projected('c',projectedShape, input->dataType(),
                    context);  //[nHeads*hS, batch*timeSteps]
  ops::matmul mmul;
  mmul.execute({projectionPrep, inputPrep}, {&projected});

  delete inputPerm;
  delete inputPermDup;
  delete inputPrep;
  delete projectionPrep;

  projected.reshapei({numHeads, projectedSize, miniBatchSize, seqLength});
  projected.permutei({2, 0, 1, 3}, false, false);  //[minibatch, numHeads, projectedSize, seqLength]

  return projected;
}


/**
 * @param shape
 * @return
 */
NDArray * AttentionHelper::lowerTriangularMask(std::vector<LongType> *shape) {
  // Causal mask over the last two dimensions [rows = queries, cols = keys], aligned
  // to the bottom-right like every FlashAttentionHelper path: query i sits at key
  // position i + max(0, cols - rows) and sees keys j <= that position. With
  // rows == cols this is the ordinary lower triangle; a single decode row sees
  // every key. Forward and backward must share this convention.
  auto rank = shape->size();
  auto rows = shape->at(rank - 2);
  auto cols = shape->at(rank - 1);
  const LongType causalOffset = cols > rows ? cols - rows : 0;

  // matrix_band_part keeps all lower diagonals (-1) and causalOffset upper ones.
  ops::matrix_band_part matrixBandPart;
  // Use FLOAT32 because matrix_band_part only supports float types (SD_FLOAT_TYPES)
  auto ones = NDArrayFactory::valueOf(*shape, 1.0f, 'c');
  auto lower = matrixBandPart.evaluate({ones}, {}, {-1, causalOffset});
  auto ret = lower.at(0)->cast(BOOL);
  // The band-part and cast kernels read `ones` and the band-part output asynchronously on CUDA, and both buffers are
  // released through the device allocator's free-stream: flush the exec stream before they go (capture-aware, a no-op
  // on CPU). `lower` frees the band-part output when it leaves scope; it used to be marked non-removable, which leaked
  // that array on every call.
  ret->synchronizeExecStream("lowerTriangularMask: sync before temp free");
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
    // Use FLOAT32 because matrix_band_part only supports float types (SD_FLOAT_TYPES)
    auto ones = NDArrayFactory::create('c',{1,qSeqLength,vSeqLength}, FLOAT32);
    float assignVal = 1.0f;
    ones->assign(assignVal);
    // Bottom-right aligned, as in lowerTriangularMask.
    const LongType causalOffset = vSeqLength > qSeqLength ? vSeqLength - qSeqLength : 0;
    auto lower = matrixBandPart.evaluate({ones},{},{-1,causalOffset});
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

  // Ensure both masks have the same type before multiplication
  // Cast to BOOL since these are logical masks
  NDArray* xBool = (x->dataType() == BOOL) ? x : x->cast(BOOL);
  NDArray* yBool = (y->dataType() == BOOL) ? y : y->cast(BOOL);

  // For boolean masks: x AND y = x * y
  // Using explicit applyTrueBroadcast to avoid operator issues
  NDArray* result = xBool->applyTrueBroadcast(sd::BroadcastOpsTuple::Multiply(), yBool);

  // Clean up casted arrays if we created them
  if (xBool != x) delete xBool;
  if (yBool != y) delete yBool;

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

    // Use appropriate large value for masking
    float largeVal = (attentionLogits->dataType() == BFLOAT16 || attentionLogits->dataType() == HALF) ? 65504.0f : 1.0e9f;

    // Cast mask to scores datatype if needed
    NDArray* numericMask = scoresMask;
    bool needsDeleteMask = false;
    if(scoresMask->dataType() != scores->dataType()) {
      numericMask = scoresMask->cast(scores->dataType());
      needsDeleteMask = true;
    }

    // Apply masking: where mask=0 (masked positions), subtract largeVal to push toward -inf
    // Where mask=1 (keep positions), the subtract and add cancel out
    // Using explicit function calls to avoid operator issues with CUDA memory

    // Apply masking using the add operation
    // maskedVals = mask * largeVal - largeVal (where mask=0 gives -largeVal, mask=1 gives 0)
    // Then attentionLogits = attentionLogits + maskedVals

    // Step 1: Create temporary result array with same shape as attentionLogits
    NDArray tempResult(attentionLogits->shapeInfo(), false, attentionLogits->getContext(), true);

    // Step 2: Compute maskedVals = numericMask * largeVal - largeVal
    // Use broadcast Add: result = attentionLogits + (numericMask * largeVal - largeVal)
    // First compute the mask offset term
    NDArray maskScaled(numericMask->shapeInfo(), false, numericMask->getContext(), true);
    numericMask->applyScalar(sd::scalar::Multiply, largeVal, &maskScaled);
    maskScaled.applyScalar(sd::scalar::Subtract, largeVal, &maskScaled);

    // Step 3: Add to attentionLogits using broadcast into temp, then copy back
    attentionLogits->applyTrueBroadcast(sd::BroadcastOpsTuple::Add(), &maskScaled, &tempResult, false);

    // Step 4: Copy result back to attentionLogits
    attentionLogits->assign(&tempResult);

    // maskScaled/tempResult (stack) and numericMask die at this block's end, but the
    // applyScalar/broadcast/assign kernels above read them ASYNCHRONOUSLY. Their
    // DataBuffers are freed via cudaFreeAsync on the DSP free-stream, so the pool can
    // recycle the blocks while those kernels are still in flight — later ops overwrite
    // them mid-read → garbage/NaN logits (BGE attention NaN root). Flush the exec
    // stream before they die; capture-aware, CPU no-op.
    attentionLogits->synchronizeExecStream("applyAttentionScores mask temps");

    if(needsDeleteMask) {
      delete numericMask;
    }
  }

  softmax.execute({attentionLogits},{scores},{},{softmaxDim});
  auto weights = scores;

  if (dropout > 0) {
    REQUIRE_TRUE(dropoutMask != nullptr, 0,
                 "AttentionHelper::applyAttentionScores: dropout needs the dropout mask output to record the draw");
    REQUIRE_TRUE(dropout <= 1.0, 0,
                 "AttentionHelper::applyAttentionScores: the dropout probability must be in [0, 1], got %f", dropout);
    // `dropout` is the probability of DROPPING a weight (the Keras rate). The dropout op reads its probability argument
    // as the probability of KEEPING an element unless its inverted flag is set, which makes it the drop probability, so
    // the flag is set here. The op runs in place on the weights (it writes every element: a dropped weight becomes 0)
    // and fills the mask output with 1 for kept and 0 for dropped weights.
    if (dropoutOp.execute({weights},{weights,dropoutMask},{dropout},{randomSeed},{true}) != Status::OK) {
      THROW_EXCEPTION("AttentionHelper::applyAttentionScores: the dropout op failed");
    }

    // Inverted dropout, as Keras applies it: the kept weights are scaled by 1 / (1 - dropout), so the expected value
    // of the attention output in training equals the inference output. The mask output then holds the multiplier the
    // weights were scaled by (0 for dropped, 1 / (1 - dropout) for kept weights), which is exactly what the backward
    // multiplies the weight gradient by. A dropout of 1 drops everything and leaves nothing to scale.
    if (dropout < 1.0) {
      const double keepScale = 1.0 / (1.0 - dropout);
      weights->applyScalar(sd::scalar::Multiply, keepScale, weights);
      dropoutMask->applyScalar(sd::scalar::Multiply, keepScale, dropoutMask);
    }
  }

  //batch size, tq tv
  //batch size tv dim
  //output: batch size, tq dim
  matmul.execute({weights,value},{applyScoresOut});

}

// Backward of the forward in doAttention (Keras' Attention layer):
//
//   logits  = scale * Q K^T, plus the additive value/causal mask      (op output 2)
//   weights = softmax(logits)
//   dropped = weights * multiplier                                    (op output 1; multiplier is op output 3, 0 or
//                                                                      1 / (1 - dropout), 1 without dropout)
//   result  = (dropped @ V) * queryMask[..., None]                    (op output 0)
//
// For the gradient E of the result:
//
//   E'        = E * queryMask[..., None]            the query mask multiplies the RESULT, so a masked query's rows
//                                                   receive no gradient (everything below uses E')
//   dV        = dropped^T E'
//   d dropped = E' V^T,   dWeights = d dropped * multiplier
//   dLogits   = weights * (dWeights - rowsum(dWeights * weights))     softmax backward
//   dQ        = scale * (dLogits * valueAndCausalMask) K,   dK = scale * (dLogits * valueAndCausalMask)^T Q
//
// The masks are passed already expanded: qMask [..., Tq, 1] and vMask [..., 1, Tv]. Without dropout the weights tensor
// the forward returned (op output 1) is the softmax output; with dropout it holds the post-dropout weights, so the
// softmax backward gets the pre-dropout weights recomputed from the logits (a dropped weight's softmax value is not
// recoverable from the dropped tensor).
void AttentionHelper::dotProductAttentionBpHelper(NDArray *query, NDArray *key, NDArray *values,
                                                  double scale,
                                                  NDArray *dLdq, NDArray *dLdk, NDArray *dLdv, NDArray *eps, LongType dropoutSeed, NDArray *qMask, NDArray *vMask, bool useCausalMask, double dropout, bool training,
                                                  NDArray *attentionScoresWeights, NDArray *attentionLogits,
                                                  NDArray *dropoutMask) {
  // Dropout needs no seed here: the forward's mask output records every draw.
  (void)dropoutSeed;

  const bool hasQueryMask = qMask != nullptr && !qMask->isEmpty() && qMask->rankOf() > 0;
  const bool useDropout = dropout > 0.0 && training;

  if(hasQueryMask) {
    REQUIRE_TRUE(qMask->sizeAt(-1) == 1 && qMask->sizeAt(-2) == eps->sizeAt(-2), 0,
                 "dot_product_attention_v2_bp: the query mask must be expanded to [..., queries, 1] (queries %lld), got "
                 "last dimensions %lld, %lld",
                 static_cast<long long>(eps->sizeAt(-2)), static_cast<long long>(qMask->sizeAt(-2)),
                 static_cast<long long>(qMask->sizeAt(-1)));
  }
  if(useDropout) {
    REQUIRE_TRUE(dropoutMask != nullptr && !dropoutMask->isEmpty(), 0,
                 "dot_product_attention_v2_bp: dropout %f in training needs the dropout mask output of the forward",
                 dropout);
    REQUIRE_TRUE(attentionLogits != nullptr && !attentionLogits->isEmpty(), 0,
                 "dot_product_attention_v2_bp: dropout in training needs the attention logits output of the forward");
  }

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

  // mergeMasks returns vMask, causalPointer or (when both exist) a new array: only the last is owned here
  mask = mergeMasks(vMask,causalPointer);

  // Temporaries of this call; all are freed together below after the stream fence.
  NDArray *queryMaskCast = nullptr;
  NDArray *maskedEps = nullptr;
  NDArray *preDropoutWeights = nullptr;
  NDArray *valueMaskCast = nullptr;

  // The query mask multiplies the forward's result, so the gradient of the result is masked the same way.
  NDArray *gradOutput = eps;
  if(hasQueryMask) {
    queryMaskCast = qMask->cast(eps->dataType());
    maskedEps = new NDArray(eps->shapeInfo());
    eps->applyTrueBroadcast(sd::BroadcastOpsTuple::Multiply(), queryMaskCast, maskedEps, false);
    gradOutput = maskedEps;
  }

  // The softmax backward differentiates the pre-dropout weights; without dropout those are the weights output itself.
  NDArray *softmaxWeights = attentionScoresWeights;
  if(useDropout) {
    preDropoutWeights = new NDArray(attentionScoresWeights->shapeInfo());
    ops::softmax softmaxOp;
    if(softmaxOp.execute({attentionLogits},{preDropoutWeights},{},{-1},{}) != Status::OK) {
      delete preDropoutWeights;
      delete maskedEps;
      delete queryMaskCast;
      THROW_EXCEPTION("dot_product_attention_v2_bp: softmax of the attention logits failed");
    }
    softmaxWeights = preDropoutWeights;
  }

  // dV = dropped^T E', and d dropped = E' V^T (the weights output is the dropped tensor)
  matMulBp.execute({attentionScoresWeights,values,gradOutput},{&dldW,dLdv},{},{});
  if(useDropout) {
    // dWeights = d dropped * multiplier: the mask output is the multiplier the forward scaled the weights by
    dldW.applyPairwiseTransform(sd::pairwise::Multiply, dropoutMask, &dldW);
  }

  softmaxBp.execute({attentionLogits,&dldW,softmaxWeights},{&dldS},{},{-1},{});

  if(scale != 0.0 && scale != 1.0) {
    // Use applyScalar instead of *= to avoid type mismatch between FLOAT arrays and double scalar
    dldS.applyScalar(sd::scalar::Multiply, scale, &dldS);
  }

  if(mask != nullptr && !mask->isEmpty()) {
    valueMaskCast = mask->cast(query->dataType());
    // Use applyTrueBroadcast to handle potentially different shapes safely
    dldS.applyTrueBroadcast(sd::BroadcastOpsTuple::Multiply(), valueMaskCast, &dldS, false);
  }

  matMulBp.execute({query,key,&dldS},{dLdq,dLdk},{},{0,1,0});

  // The kernels above read these temporaries asynchronously on CUDA, and their buffers are released through the device
  // allocator's free-stream, which can recycle a block while a kernel still reads it. Flush the exec stream before
  // anything is freed (capture-aware, a no-op on CPU), as the other temporaries in this file do.
  const bool ownsMergedMask = mask != nullptr && mask != vMask && mask != causalPointer;
  if(queryMaskCast != nullptr || maskedEps != nullptr || preDropoutWeights != nullptr || valueMaskCast != nullptr ||
     causalPointer != nullptr || ownsMergedMask) {
    dLdq->synchronizeExecStream("dotProductAttentionBpHelper: sync before temp free");
  }
  delete queryMaskCast;
  delete maskedEps;
  delete preDropoutWeights;
  delete valueMaskCast;
  if(ownsMergedMask) delete mask;
  delete causalPointer;
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

  // An absent dropout multiplier is a null entry or no entry at all
  auto dropoutMask = inputs.size() > 7 ? inputs[7] : nullptr;

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

  // Clean up squeezed mask if we created one: the kernels above may still read it on CUDA, so fence first
  if(squeezedVMask != nullptr) {
    dLdq->synchronizeExecStream("doAttentionBp squeezed mask free");
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

  // mergeMasks returns vmaskInternal, casualPointer or (when both exist) a new array: only the last is owned here
  auto scoresMask = mergeMasks(vmaskInternal,casualPointer);

  //compute actual softmax now
  if(training) {
    applyAttentionScores(attentionScores, v, scoresMask, dropout, dropoutSeed, applyScoresOut, attentionLogits,
                         dropoutMask);
  } else {
    applyAttentionScores(attentionScores, v, scoresMask, 0, dropoutSeed, applyScoresOut, attentionLogits, dropoutMask);
  }

  // The dropout mask output is the multiplier the weights were scaled by; when no dropout ran every weight passed
  // through unchanged, so the multiplier is 1 rather than whatever the output buffer held.
  if(dropoutMask != nullptr && !(training && dropout > 0)) {
    double passThrough = 1.0;
    dropoutMask->assign(passThrough);
  }

  // Keras applies the query mask to the RESULT (result *= query_mask[..., None]) and returns the attention weights
  // unmasked: a masked query's output row is zero, and the weights it attended with stay visible.
  //inputs: scores:  batch size tq tv value:batch size, tv,dim scoresmask: batch size 1 tv or batch size tq tv
  if(qMask != nullptr && !qMask->isEmpty()) {
    qMaskExpandResult = expandDims.evaluate({qMaskInternal},{},{-1});
    qMaskInternal = qMaskExpandResult.at(0);
    REQUIRE_TRUE(qMaskInternal->sizeAt(-2) == applyScoresOut->sizeAt(-2), 0,
                 "dot_product_attention_v2: the query mask needs one entry per query (%lld queries), got %lld",
                 static_cast<long long>(applyScoresOut->sizeAt(-2)), static_cast<long long>(qMaskInternal->sizeAt(-2)));
    auto casted = qMaskInternal->cast(applyScoresOut->dataType());
    // Use applyTrueBroadcast to handle potentially different shapes safely
    applyScoresOut->applyTrueBroadcast(sd::BroadcastOpsTuple::Multiply(), casted, applyScoresOut, false);
    // The broadcast above reads `casted` asynchronously; flush the exec stream before freeing it (see the
    // applyAttentionScores mask-temps comment for the free-under-read hazard).
    applyScoresOut->synchronizeExecStream("doAttention qMask temp");
    delete casted;
  }

  // Free what was created here. A temporary mask is read asynchronously by the kernels above (the mask-add kernels in
  // applyAttentionScores were already fenced by its own stream flush, the merge by the flush below), so fence first.
  const bool ownsMergedMask = scoresMask != nullptr && scoresMask != vmaskInternal && scoresMask != casualPointer;
  if(casualPointer != nullptr || ownsMergedMask || squeezedVMask != nullptr) {
    applyScoresOut->synchronizeExecStream("doAttention mask temps");
  }
  if(ownsMergedMask) delete scoresMask;
  delete casualPointer;
  // Clean up squeezed mask if we created one
  delete squeezedVMask;
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
  auto epsPermDup = epsPerm->dup('c');  // force contiguous before reshape
  std::vector<sd::LongType> epsReshapeVec = {numHeads * projectedSize, miniBatchSize * seqLength};
  auto epsReshaped = epsPermDup->reshape('c', epsReshapeVec);

  std::vector<sd::LongType> inputPermVec = {1, 0, 2};
  auto inputPerm = input->permute(inputPermVec, false, false);
  auto inputPermDup = inputPerm->dup('c');  // force contiguous before reshape
  std::vector<sd::LongType> inputPermShape = {input->sizeAt(1), miniBatchSize * seqLength};
  auto inputPrep = inputPermDup->reshape('c',inputPermShape,false);
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

  delete epsPerm;
  delete epsPermDup;
  delete epsReshaped;
  delete inputPerm;
  delete inputPermDup;
  delete inputPrep;
  delete projectionPrep;

}
}  // namespace sd
#endif

#endif
