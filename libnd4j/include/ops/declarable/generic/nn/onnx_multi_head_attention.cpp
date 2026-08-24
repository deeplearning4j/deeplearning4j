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
// ONNX MultiHeadAttention op - takes pre-projected Q, K, V
// Compatible with Microsoft's ONNX MultiHeadAttention operator
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_onnx_multi_head_attention)

#include <math/templatemath.h>
#include <helpers/FlashAttentionHelper.h>
#include <helpers/AttentionWorkspace.h>
#include <ops/declarable/helpers/kv_scatter.h>
#include <ops/declarable/headers/nn.h>
#include <graph/DspDeviceDispatch.h>
#include <cmath>

namespace sd {
namespace ops {

/**
 * ONNX MultiHeadAttention - for pre-projected queries, keys, values
 *
 * Inputs:
 *   0: query          [batch, seqQ, hidden] - already projected
 *   1: key            [batch, seqKV, hidden] - already projected
 *   2: value          [batch, seqKV, hidden] - already projected
 *   3: attn_bias      [batch, numHeads, seqQ, seqKV] or broadcastable (optional, can be empty)
 *   4: past_key       [batch, numHeads, pastSeq, headDim] (optional, can be empty)
 *   5: past_value     [batch, numHeads, pastSeq, headDim] (optional, can be empty)
 *   6: cache_position [1] INT64 scalar (optional) - when present, enables in-place KV write mode:
 *        writes current K/V at this position in past_key/past_value (in-place),
 *        uses past_key/past_value as the full KV sequence (no concatenation).
 *        This fixes causal mask alignment: the model's causal mask uses position_ids
 *        to determine visibility, so current K must be at the same position as position_ids.
 *        Without this, concat places current K at pastSeq which is > position_ids, causing
 *        the causal mask to mask out the current token's own key.
 *
 * Int args:
 *   0: numHeads
 *   1: useCausalMask (0 or 1)
 *
 * Float args:
 *   0: scale (0 = auto compute 1/sqrt(headDim))
 *
 * Outputs:
 *   0: output         [batch, seqQ, hidden]
 *   1: present_key    [batch, numHeads, totalSeq, headDim] (optional)
 *   2: present_value  [batch, numHeads, totalSeq, headDim] (optional)
 */
CUSTOM_OP_IMPL(onnx_multi_head_attention, 3, -1, false, -2, 2) {
  auto query = INPUT_VARIABLE(0);   // [batch, seqQ, hidden]
  auto key = INPUT_VARIABLE(1);     // [batch, seqKV, hidden]
  auto value = INPUT_VARIABLE(2);   // [batch, seqKV, hidden]
  
  NDArray* attnBias = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  NDArray* pastKey = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;
  NDArray* pastValue = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
  NDArray* cachePosInput = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;

  // Handle empty arrays and scalar placeholders as nullptr
  // Scalars (rank 0 or length <= 1) are used as placeholders for missing optional inputs
  if (attnBias != nullptr && (attnBias->isEmpty() || attnBias->rankOf() == 0 || attnBias->lengthOf() <= 1)) attnBias = nullptr;
  if (pastKey != nullptr && (pastKey->isEmpty() || pastKey->rankOf() == 0 || pastKey->lengthOf() <= 1)) pastKey = nullptr;
  if (pastValue != nullptr && (pastValue->isEmpty() || pastValue->rankOf() == 0 || pastValue->lengthOf() <= 1)) pastValue = nullptr;
  // cache_position is a capture-stable INT64 scalar. Use the selected backend pointer:
  // host builds legitimately have no specialBuffer(), while CUDA uses the device pointer.
  if (cachePosInput != nullptr && (cachePosInput->isEmpty() || cachePosInput->lengthOf() != 1
      || cachePosInput->dataType() != DataType::INT64
      || sd::graph::dspBufferConst(cachePosInput) == nullptr))
    cachePosInput = nullptr;
  bool useInPlaceKv = (cachePosInput != nullptr && pastKey != nullptr && pastValue != nullptr);
  
  auto output = OUTPUT_VARIABLE(0);

  LongType numHeads = INT_ARG(0);
  bool useCausalMask = INT_ARG(1) != 0;

  double scale = block.numT() > 0 ? T_ARG(0) : 0.0;

  // Handle empty K/V inputs — no key-value pairs to attend to.
  // This happens during the first decode step when the KV cache is empty.
  // Zero the output and present_key/present_value since there's nothing to attend to.
  if (key->isEmpty() || key->lengthOf() == 0 || value->isEmpty() || value->lengthOf() == 0) {
    output->nullify();
    if (block.outputWidth() > 1) OUTPUT_VARIABLE(1)->nullify();
    if (block.outputWidth() > 2) OUTPUT_VARIABLE(2)->nullify();
    return sd::Status::OK;
  }

  // Mixed-type auto-cast: cast all inputs to query's dtype.
  // Query dtype is authoritative (matches DECLARE_SHAPE_FN output dtype).
  // IMPORTANT: save original pastKey/pastValue pointers BEFORE casting.
  // kvInPlaceWrite must target the real persistent KV cache buffers,
  // not cast temporaries that get deleted at end of op.
  DataType targetType = query->dataType();
  NDArray* keyCast = nullptr;
  NDArray* valueCast = nullptr;
  NDArray* pastKeyCast = nullptr;
  NDArray* pastValueCast = nullptr;
  NDArray* origPastKey = pastKey;
  NDArray* origPastValue = pastValue;
  if (key->dataType() != targetType) {
    keyCast = key->cast(targetType);
    key = keyCast;
  }
  if (value->dataType() != targetType) {
    valueCast = value->cast(targetType);
    value = valueCast;
  }
  if (pastKey && pastKey->dataType() != targetType) {
    pastKeyCast = pastKey->cast(targetType);
    pastKey = pastKeyCast;
  }
  if (pastValue && pastValue->dataType() != targetType) {
    pastValueCast = pastValue->cast(targetType);
    pastValue = pastValueCast;
  }

  auto batch = query->sizeAt(0);
  auto seqQ = query->sizeAt(1);
  auto hidden = query->sizeAt(2);
  auto seqKV = key->sizeAt(1);
  auto kvHidden = key->sizeAt(2);
  auto headDim = hidden / numHeads;

  REQUIRE_TRUE(query->rankOf() == 3, 0,
               "onnx_multi_head_attention: query must be rank 3 [batch, seq, hidden], got %s",
               ShapeUtils::shapeAsString(query).c_str());
  REQUIRE_TRUE(key->rankOf() == 3, 0,
               "onnx_multi_head_attention: key must be rank 3, got %s",
               ShapeUtils::shapeAsString(key).c_str());
  REQUIRE_TRUE(value->rankOf() == 3, 0,
               "onnx_multi_head_attention: value must be rank 3, got %s",
               ShapeUtils::shapeAsString(value).c_str());
  REQUIRE_TRUE(hidden % numHeads == 0, 0,
               "onnx_multi_head_attention: hidden size %lld must be divisible by numHeads %lld",
               hidden, numHeads);

  // GQA (Grouped Query Attention): K/V may have fewer heads than Q.
  // Q hidden = numHeads * headDim, KV hidden = numKvHeads * headDim
  REQUIRE_TRUE(headDim > 0, 0, "onnx_multi_head_attention: headDim must be > 0");
  REQUIRE_TRUE(kvHidden % headDim == 0, 0,
               "onnx_multi_head_attention: KV hidden size %lld must be divisible by headDim %lld",
               kvHidden, headDim);
  LongType numKvHeads = kvHidden / headDim;
  REQUIRE_TRUE(numHeads % numKvHeads == 0, 0,
               "onnx_multi_head_attention: numHeads %lld must be divisible by numKvHeads %lld (GQA constraint)",
               numHeads, numKvHeads);

  // Compute scale if not provided
  if (scale <= 0.0) {
    scale = 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(headDim));
  }

  // Reshape [batch, seq, hidden] -> [batch, seq, heads, headDim] (BSHD format for FlashAttentionHelper)
  // Q uses numHeads, K/V use numKvHeads (may differ for GQA)
  // reshape('c', ..., false) attempts a zero-copy view but falls back to copy if the source
  // has non-contiguous strides (checked internally via reshapeNoAlloc).
  std::vector<LongType> qShape4d = {batch, seqQ, numHeads, headDim};
  std::vector<LongType> kvShape4d = {batch, seqKV, numKvHeads, headDim};

  NDArray* qReshaped = query->reshape('c', qShape4d, false);
  NDArray* kReshaped = key->reshape('c', kvShape4d, false);
  NDArray* vReshaped = value->reshape('c', kvShape4d, false);
  
  // Handle past key/value concatenation
  // Use pointers and track ownership
  NDArray* kFinal = nullptr;
  NDArray* vFinal = nullptr;
  bool ownKVFinal = false;  // Whether we own kFinal/vFinal memory
  bool skipPresentOutput = false;  // True when present output was written in-place
  LongType totalSeqKV = seqKV;
  
  // Skip past_key concat when the upstream graph already handled it.
  // Detection: if past_key's head count (axis 1) doesn't match numKvHeads derived from
  // the key input, then GQA repeat was applied upstream and past_key is in raw format.
  // Also skip if K already includes past positions (seqKV > seqQ).
  auto pastKvHeadCount = (pastKey != nullptr) ? pastKey->sizeAt(1) : numKvHeads;
  bool pastAlreadyConcat = (pastKey != nullptr && pastKvHeadCount != numKvHeads);

  if (useInPlaceKv) {
    REQUIRE_TRUE(pastKey->rankOf() == 4 && pastValue->rankOf() == 4, 0,
                 "onnx_multi_head_attention: in-place KV caches must be rank-4 BHSD");
    REQUIRE_TRUE(pastKey->sizeAt(0) == batch && pastValue->sizeAt(0) == batch
                     && pastKey->sizeAt(1) == numKvHeads && pastValue->sizeAt(1) == numKvHeads
                     && pastKey->sizeAt(3) == headDim && pastValue->sizeAt(3) == headDim
                     && pastKey->sizeAt(2) == pastValue->sizeAt(2), 0,
                 "onnx_multi_head_attention: in-place KV cache shape does not match current K/V");
    // In-place KV write mode: write new K/V token(s) at cache_position directly
    // into the persistent pastKey/pastValue buffers. This eliminates the bulk
    // past→present copy (4 assign kernels per layer, 120 kernels/step for 30 layers).
    //
    // CUDA graph compatible: kvInPlaceWrite reads cache_position from a device-side
    // pointer. The pointer ADDRESS is baked into the graph at capture time; only
    // the VALUE changes between replays (updated via cudaMemcpyAsync before replay).
    // The old code used cachePosInput->e<LongType>(0) which bakes the HOST VALUE
    // into the graph — broken on replay.
    //
    // We write to pastKey/pastValue (ext inputs = persistent static KV buffers)
    // rather than presentKey/presentValue (plan outputs) because:
    // (a) past buffers have stable addresses across CUDA graph replays
    // (b) the next step reads from past, not present
    // (c) present outputs may be prezeroed on some plan paths

    // pastKey is BHSD [batch, numKvHeads, maxSeqLen, headDim]
    // kReshaped is BSHD [batch, seqKV, numKvHeads, headDim]
    // cachePosPtr: device pointer (CUDA) / host pointer (CPU) to int64 position

    // Use specialBuffer on CUDA (device pointer), buffer on CPU (host pointer)
    const void* cachePosPtr = sd::graph::dspBufferConst(cachePosInput);

    // Write into the ORIGINAL persistent KV cache buffers, not cast temporaries.
    // If pastKey was cast to a different dtype, origPastKey points to the real cache.
    helpers::kvInPlaceWrite(origPastKey, kReshaped, cachePosPtr, block.launchContext());
    helpers::kvInPlaceWrite(origPastValue, vReshaped, cachePosPtr, block.launchContext());

    // Use the (possibly cast) pastKey/pastValue for attention (permute BHSD→BSHD).
    // pastKey/pastValue were auto-cast to query dtype at lines 127-134 if needed.
    // origPastKey/origPastValue are the real cache buffers (may be HALF);
    // pastKey/pastValue are FLOAT copies if cast was needed.
    // totalSeqKV = maxSeqLen: the causal mask handles token visibility,
    // so we pass the full buffer. Positions beyond cachePos+seqKV contain
    // zeros or stale data but are masked out by the attention mask.
    totalSeqKV = pastKey->sizeAt(2);  // maxSeqLen

    std::vector<LongType> permBHSDtoBSHD = {0, 2, 1, 3};
    kFinal = pastKey->permute(permBHSDtoBSHD, false, false);
    vFinal = pastValue->permute(permBHSDtoBSHD, false, false);
    ownKVFinal = true;   // We own the permuted views (need delete)
    skipPresentOutput = true;  // No present output copy needed — past IS the cache
  } else if (pastKey != nullptr && pastValue != nullptr && !pastAlreadyConcat) {
    // Standard concat mode: concatenate past + current KV
    // Past is [batch, numKvHeads, pastSeq, headDim] (BHSD format from ONNX)
    // Need to permute to [batch, pastSeq, numKvHeads, headDim] (BSHD) for concat
    auto pastSeq = pastKey->sizeAt(2);
    totalSeqKV = pastSeq + seqKV;

    // Get output buffers for direct write
    auto presentKeyOut = block.outputWidth() > 1 ? OUTPUT_VARIABLE(1) : nullptr;
    auto presentValueOut = block.outputWidth() > 2 ? OUTPUT_VARIABLE(2) : nullptr;

    if (presentKeyOut != nullptr && presentValueOut != nullptr) {
      // Write directly to output buffers - they are already the correct size [batch, numKvHeads, totalSeqKV, headDim]
      // Permute output from BHSD to BSHD for use in attention
      std::vector<LongType> permBHSDtoBSHD = {0, 2, 1, 3};
      kFinal = presentKeyOut->permute(permBHSDtoBSHD, false, false);
      vFinal = presentValueOut->permute(permBHSDtoBSHD, false, false);
      ownKVFinal = false;

      // Copy past KV to [0:pastSeq] positions
      NDArray* pastKeyBSHD = pastKey->permute(permBHSDtoBSHD, false, false);
      NDArray* pastValueBSHD = pastValue->permute(permBHSDtoBSHD, false, false);

      std::vector<LongType> pastSliceIdx = {0, batch, 0, pastSeq, 0, numKvHeads, 0, headDim};
      NDArray* kPastSlice = (*kFinal)(pastSliceIdx);
      NDArray* vPastSlice = (*vFinal)(pastSliceIdx);
      kPastSlice->assign(pastKeyBSHD);
      vPastSlice->assign(pastValueBSHD);
      delete kPastSlice;
      delete vPastSlice;
      delete pastKeyBSHD;
      delete pastValueBSHD;

      // Write current K/V to [pastSeq:totalSeqKV] position
      std::vector<LongType> curSliceIdx = {0, batch, pastSeq, totalSeqKV, 0, numKvHeads, 0, headDim};
      NDArray* kCurSlice = (*kFinal)(curSliceIdx);
      NDArray* vCurSlice = (*vFinal)(curSliceIdx);

      if (kCurSlice->lengthOf() > 0 && kReshaped->lengthOf() > 0) {
        kCurSlice->applyTrueBroadcast(BroadcastOpsTuple::Assign(), kReshaped, kCurSlice, false);
        vCurSlice->applyTrueBroadcast(BroadcastOpsTuple::Assign(), vReshaped, vCurSlice, false);
      }

      delete kCurSlice;
      delete vCurSlice;
    } else {
      // Fallback: allocate new buffers
      std::vector<LongType> permBHSDtoBSHD = {0, 2, 1, 3};
      NDArray* pastKeyBSHD = pastKey->permute(permBHSDtoBSHD, false, false);
      NDArray* pastValueBSHD = pastValue->permute(permBHSDtoBSHD, false, false);

      std::vector<LongType> finalShape = {batch, totalSeqKV, numKvHeads, headDim};
      kFinal = new NDArray('c', finalShape, key->dataType(), block.launchContext());
      vFinal = new NDArray('c', finalShape, value->dataType(), block.launchContext());
      ownKVFinal = true;

      std::vector<LongType> pastSliceIdx = {0, batch, 0, pastSeq, 0, numKvHeads, 0, headDim};
      std::vector<LongType> curSliceIdx = {0, batch, pastSeq, totalSeqKV, 0, numKvHeads, 0, headDim};

      NDArray* kPastSlice = (*kFinal)(pastSliceIdx);
      NDArray* vPastSlice = (*vFinal)(pastSliceIdx);
      kPastSlice->assign(pastKeyBSHD);
      vPastSlice->assign(pastValueBSHD);
      delete kPastSlice;
      delete vPastSlice;

      NDArray* kCurSlice = (*kFinal)(curSliceIdx);
      NDArray* vCurSlice = (*vFinal)(curSliceIdx);
      kCurSlice->assign(kReshaped);
      vCurSlice->assign(vReshaped);
      delete kCurSlice;
      delete vCurSlice;

      delete pastKeyBSHD;
      delete pastValueBSHD;
    }
  } else {
    // No past - just use reshaped k/v directly
    kFinal = kReshaped;
    vFinal = vReshaped;
    ownKVFinal = false;  // Don't delete, kReshaped/vReshaped will be deleted later

    // Write current K/V as present outputs even without past
    // kReshaped is BSHD [batch, seqKV, numKvHeads, headDim], permute to BHSD for ONNX format
    std::vector<LongType> permBSHDtoBHSD = {0, 2, 1, 3};
    if (block.outputWidth() > 1) {
      auto presentKey = OUTPUT_VARIABLE(1);
      NDArray* kBHSD = kReshaped->permute(permBSHDtoBHSD, false, false);
      presentKey->assign(kBHSD);
      delete kBHSD;
    }
    if (block.outputWidth() > 2) {
      auto presentValue = OUTPUT_VARIABLE(2);
      NDArray* vBHSD = vReshaped->permute(permBSHDtoBHSD, false, false);
      presentValue->assign(vBHSD);
      delete vBHSD;
    }
  }
  
  // Setup FlashAttentionHelper config
  FlashAttentionHelper::Config config;
  config.scale = static_cast<float>(scale);
  // In-place cache mode has a device-resident logical sequence length. Keep the physical
  // cache shape fixed for capture/replay, but make the fused attention kernel stop at
  // cache_position + currentSeq instead of scanning the entire padded envelope.
  config.isCausal = useCausalMask || useInPlaceKv;
  config.dropout = 0.0f;
  config.numHeads = numHeads;
  config.numKvHeads = numKvHeads;
  if (useInPlaceKv) {
    config.currentKeyWindow = kReshaped;
    config.currentValueWindow = vReshaped;
    config.currentKvPosition = sd::graph::dspBufferConst(cachePosInput);
  }
  
  // Cast attention bias to query dtype if needed
  std::unique_ptr<NDArray> attnBiasCastOwner;
  if (attnBias != nullptr && attnBias->dataType() != query->dataType()) {
    attnBiasCastOwner.reset(attnBias->cast(query->dataType()));
    attnBias = attnBiasCastOwner.get();
  }

  // Output in BSHD format [batch, seqQ, numHeads, headDim].
  //
  // reshape() may return either a zero-copy view or an allocated copy when the
  // DSP output has a non-contiguous layout. Write directly only when the
  // reshaped array demonstrably shares the output DataBuffer. Otherwise retain
  // the workspace + explicit copy-back path so non-contiguous outputs remain
  // correct. This removes one full output copy per layer on the common
  // contiguous decode path without weakening the allocator/layout contract.
  std::vector<LongType> outShape4d = {batch, seqQ, numHeads, headDim};
  std::unique_ptr<NDArray> outputView4d(output->reshape('c', outShape4d, false));
  const bool directOutput =
      outputView4d != nullptr && outputView4d->dataBuffer() != nullptr &&
      outputView4d->dataBuffer() == output->dataBuffer();

  auto workspace = AttentionWorkspace::getInstance();
  NDArray* attnOut4d = directOutput
      ? outputView4d.get()
      : workspace->getBuffer("mha_attnOut4d", outShape4d, query->dataType(),
                             block.launchContext());

  // Call FlashAttentionHelper::forward with 4D tensors (BSHD format)
  FlashAttentionHelper::forward(qReshaped, kFinal, vFinal, attnOut4d, config,
                                nullptr, nullptr, nullptr,
                                block.launchContext(), attnBias);

  if (!directOutput) {
    std::vector<LongType> outShape3d = {batch, seqQ, hidden};
    std::unique_ptr<NDArray> attnOutFinal(attnOut4d->reshape('c', outShape3d, false));
    output->assign(attnOutFinal.get());
  }
  
  // Output present key/value if requested (in BHSD format for ONNX compatibility)
  // Skip if we already wrote directly to output buffers (in-place KV or direct write mode)
  if (block.outputWidth() > 1 && ownKVFinal && !skipPresentOutput) {
    auto presentKey = OUTPUT_VARIABLE(1);
    // kFinal is BSHD [batch, totalSeqKV, numKvHeads, headDim], permute to BHSD
    std::vector<LongType> permBSHDtoBHSD = {0, 2, 1, 3};
    NDArray* kBHSD = kFinal->permute(permBSHDtoBHSD, false, false);
    presentKey->assign(kBHSD);
    delete kBHSD;
  }
  if (block.outputWidth() > 2 && ownKVFinal && !skipPresentOutput) {
    auto presentValue = OUTPUT_VARIABLE(2);
    std::vector<LongType> permBSHDtoBHSD2 = {0, 2, 1, 3};
    NDArray* vBHSD = vFinal->permute(permBSHDtoBHSD2, false, false);
    presentValue->assign(vBHSD);
    delete vBHSD;
  }
  
  // Clean up
  if (ownKVFinal) {
    delete kFinal;
    delete vFinal;
  }
  delete qReshaped;
  delete kReshaped;
  delete vReshaped;

  delete keyCast;
  delete valueCast;
  delete pastKeyCast;
  delete pastValueCast;

  return sd::Status::OK;
}

DECLARE_TYPES(onnx_multi_head_attention) {
  getOpDescriptor()->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // query
      ->setAllowedInputTypes(1, {ALL_FLOATS})   // key
      ->setAllowedInputTypes(2, {ALL_FLOATS})   // value
      ->setAllowedInputTypes(3, {ALL_FLOATS, ALL_INTS, BOOL})  // attn_bias (optional)
      ->setAllowedInputTypes(4, {ALL_FLOATS})   // past_key (optional)
      ->setAllowedInputTypes(5, {ALL_FLOATS})   // past_value (optional)
      ->setAllowedInputTypes(6, {DataType::INT64})  // cache_position (optional scalar)
      ->setAllowedOutputTypes({ALL_FLOATS})
      ;
}

DECLARE_SHAPE_FN(onnx_multi_head_attention) {
  auto queryShape = inputShape->at(0);
  auto keyShape = inputShape->at(1);

  REQUIRE_TRUE(shape::rank(queryShape) >= 3, 0,
               "onnx_multi_head_attention: query shape must have rank >= 3, got rank %d", shape::rank(queryShape));
  REQUIRE_TRUE(shape::rank(keyShape) >= 3, 0,
               "onnx_multi_head_attention: key shape must have rank >= 3, got rank %d", shape::rank(keyShape));

  auto batch = shape::sizeAt(queryShape, static_cast<LongType>(0));
  auto seqQ = shape::sizeAt(queryShape, static_cast<LongType>(1));
  auto hidden = shape::sizeAt(queryShape, static_cast<LongType>(2));
  auto seqKV = shape::sizeAt(keyShape, static_cast<LongType>(1));
  auto kvHidden = shape::sizeAt(keyShape, static_cast<LongType>(2));

  LongType numHeads = INT_ARG(0);
  auto headDim = hidden / numHeads;
  // GQA: K/V may have fewer heads than Q
  LongType numKvHeads = (headDim > 0) ? (kvHidden / headDim) : numHeads;

  // Check for past key/value to determine total sequence length
  // Always use concat shape (pastSeq + seqKV) even with in-place KV write,
  // because the model's attn_mask_reformat subgraph derives shapes from
  // past_key_shape[2] + current_key_shape[1] and expects consistent dimensions.
  LongType totalSeqKV = seqKV;
  if (inputShape->size() > 4) {
    auto pastKeyShape = inputShape->at(4);
    auto pastKeyRank = shape::rank(pastKeyShape);
    if (pastKeyShape != nullptr && !shape::isEmpty(pastKeyShape) && pastKeyRank == 4 && shape::length(pastKeyShape) > 1) {
      totalSeqKV += shape::sizeAt(pastKeyShape, static_cast<LongType>(2));
    }
  }

  // Promote to widest FP type among Q/K/V (mirrors runtime auto-cast in CUSTOM_OP_IMPL)
  auto valueShape = inputShape->at(2);
  auto dtype = ArrayOptions::dataType(queryShape);
  auto keyDtype = ArrayOptions::dataType(keyShape);
  auto valueDtype = ArrayOptions::dataType(valueShape);
  if (DataTypeUtils::sizeOfElement(keyDtype) > DataTypeUtils::sizeOfElement(dtype))
    dtype = keyDtype;
  if (DataTypeUtils::sizeOfElement(valueDtype) > DataTypeUtils::sizeOfElement(dtype))
    dtype = valueDtype;

  // Handle empty K/V — produce empty present shapes with ARRAY_EMPTY flag
  if (shape::isEmpty(keyShape) || seqKV == 0) {
    auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {batch, seqQ, hidden});
    std::vector<LongType> presentDims = {batch, numKvHeads, totalSeqKV, headDim};
    auto presentKeyShape = ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(dtype, presentDims);
    auto presentValueShape = ConstantShapeHelper::getInstance().emptyShapeInfoWithShape(dtype, presentDims);
    return SHAPELIST(outputShape, CONSTANT(presentKeyShape), CONSTANT(presentValueShape));
  }

  // Output: [batch, seqQ, hidden]
  auto outputShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {batch, seqQ, hidden});

  // Present key/value: [batch, numKvHeads, totalSeqKV, headDim] (BHSD for ONNX)
  auto presentKeyShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c',
                                                                             {batch, numKvHeads, totalSeqKV, headDim});
  auto presentValueShape = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c',
                                                                               {batch, numKvHeads, totalSeqKV, headDim});

  return SHAPELIST(outputShape, presentKeyShape, presentValueShape);
}

}  // namespace ops
}  // namespace sd

#endif
