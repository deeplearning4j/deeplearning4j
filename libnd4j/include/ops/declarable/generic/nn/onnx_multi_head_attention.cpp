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

#include <helpers/FlashAttentionHelper.h>
#include <ops/declarable/headers/nn.h>
#include <cmath>

namespace sd {
namespace ops {

/**
 * ONNX MultiHeadAttention - for pre-projected queries, keys, values
 *
 * Inputs:
 *   0: query        [batch, seqQ, hidden] - already projected
 *   1: key          [batch, seqKV, hidden] - already projected  
 *   2: value        [batch, seqKV, hidden] - already projected
 *   3: attn_bias    [batch, numHeads, seqQ, seqKV] or broadcastable (optional, can be empty)
 *   4: past_key     [batch, numHeads, pastSeq, headDim] (optional, can be empty)
 *   5: past_value   [batch, numHeads, pastSeq, headDim] (optional, can be empty)
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
  
  // Handle empty arrays and scalar placeholders as nullptr
  // Scalars (rank 0 or length <= 1) are used as placeholders for missing optional inputs
  if (attnBias != nullptr && (attnBias->isEmpty() || attnBias->rankOf() == 0 || attnBias->lengthOf() <= 1)) attnBias = nullptr;
  if (pastKey != nullptr && (pastKey->isEmpty() || pastKey->rankOf() == 0 || pastKey->lengthOf() <= 1)) pastKey = nullptr;
  if (pastValue != nullptr && (pastValue->isEmpty() || pastValue->rankOf() == 0 || pastValue->lengthOf() <= 1)) pastValue = nullptr;
  
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
    scale = 1.0 / std::sqrt(static_cast<double>(headDim));
  }

  // Reshape [batch, seq, hidden] -> [batch, seq, heads, headDim] (BSHD format for FlashAttentionHelper)
  // Q uses numHeads, K/V use numKvHeads (may differ for GQA)
  std::vector<LongType> qShape4d = {batch, seqQ, numHeads, headDim};
  std::vector<LongType> kvShape4d = {batch, seqKV, numKvHeads, headDim};
  
  NDArray* qReshaped = query->reshape('c', qShape4d);
  NDArray* kReshaped = key->reshape('c', kvShape4d);
  NDArray* vReshaped = value->reshape('c', kvShape4d);
  
  // Handle past key/value concatenation
  // Use pointers and track ownership
  NDArray* kFinal = nullptr;
  NDArray* vFinal = nullptr;
  bool ownKVFinal = false;  // Whether we own kFinal/vFinal memory
  LongType totalSeqKV = seqKV;
  
  if (pastKey != nullptr && pastValue != nullptr) {
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
      kPastSlice->syncToDevice();
      vPastSlice->syncToDevice();
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
        kCurSlice->syncToDevice();
        vCurSlice->syncToDevice();
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
  config.isCausal = useCausalMask;
  config.dropout = 0.0f;
  config.numHeads = numHeads;
  config.numKvHeads = numKvHeads;
  
  // Output in BSHD format [batch, seqQ, numHeads, headDim]
  std::vector<LongType> outShape4d = {batch, seqQ, numHeads, headDim};
  NDArray attnOut4d('c', outShape4d, query->dataType(), block.launchContext());
  
  // Cast attention bias to query dtype if needed
  std::unique_ptr<NDArray> attnBiasCastOwner;
  if (attnBias != nullptr && attnBias->dataType() != query->dataType()) {
    attnBiasCastOwner.reset(attnBias->cast(query->dataType()));
    attnBias = attnBiasCastOwner.get();
  }
  
  // Call FlashAttentionHelper::forward with 4D tensors (BSHD format)
  FlashAttentionHelper::forward(qReshaped, kFinal, vFinal, &attnOut4d, config,
                                nullptr, nullptr, nullptr,
                                block.launchContext(), attnBias);
  
  // Reshape output back to [batch, seqQ, hidden]
  std::vector<LongType> outShape3d = {batch, seqQ, hidden};
  NDArray* attnOutFinal = attnOut4d.reshape('c', outShape3d);
  output->assign(attnOutFinal);
  delete attnOutFinal;
  
  // Output present key/value if requested (in BHSD format for ONNX compatibility)
  // Skip if we already wrote directly to output buffers (ownKVFinal == false && using output view)
  if (block.outputWidth() > 1 && ownKVFinal) {
    auto presentKey = OUTPUT_VARIABLE(1);
    // kFinal is BSHD [batch, totalSeqKV, numKvHeads, headDim], permute to BHSD
    std::vector<LongType> permBSHDtoBHSD = {0, 2, 1, 3};
    NDArray* kBHSD = kFinal->permute(permBSHDtoBHSD, false, false);
    presentKey->assign(kBHSD);
    delete kBHSD;
  }
  if (block.outputWidth() > 2 && ownKVFinal) {
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
  
  return sd::Status::OK;
}

DECLARE_TYPES(onnx_multi_head_attention) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})   // query
      ->setAllowedInputTypes(1, {ALL_FLOATS})   // key
      ->setAllowedInputTypes(2, {ALL_FLOATS})   // value
      ->setAllowedInputTypes(3, {ALL_FLOATS, ALL_INTS, BOOL})  // attn_bias (optional)
      ->setAllowedInputTypes(4, {ALL_FLOATS})   // past_key (optional)
      ->setAllowedInputTypes(5, {ALL_FLOATS})   // past_value (optional)
      ->setAllowedOutputTypes({ALL_FLOATS});
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
  LongType totalSeqKV = seqKV;
  if (inputShape->size() > 4) {
    auto pastKeyShape = inputShape->at(4);
    auto pastKeyRank = shape::rank(pastKeyShape);
    if (pastKeyShape != nullptr && !shape::isEmpty(pastKeyShape) && pastKeyRank == 4 && shape::length(pastKeyShape) > 1) {
      totalSeqKV += shape::sizeAt(pastKeyShape, static_cast<LongType>(2));
    }
  }

  auto dtype = ArrayOptions::dataType(queryShape);

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
