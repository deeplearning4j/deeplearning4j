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
// @author Adam Gibson
//
// unpad_input: removes padding tokens from a batch using attention_mask.
// pad_input:   scatters unpadded tokens back to padded positions.
// pad_input_bp: backward pass for pad_input (gather from padded grad to unpadded).
//

#include <system/op_boilerplate.h>
#include <array/NDArrayFactory.h>
#include <cstring>

#if NOT_EXCLUDED(OP_unpad_input) || NOT_EXCLUDED(OP_pad_input) || NOT_EXCLUDED(OP_pad_input_bp)

#include <ops/declarable/headers/llm.h>
#include <helpers/ShapeUtils.h>

namespace sd {
namespace ops {

//////////////////////////////////////////////////////////////////////////
// unpad_input
// Removes padding tokens from a batch of sequences using attention_mask.
//
// Inputs:
//   0: hidden_states [batch, max_seq_len, hidden_dim] — padded input (any float)
//   1: attention_mask [batch, max_seq_len] — 1=real token, 0=pad (INT32 or BOOL)
//
// Outputs:
//   0: unpadded   [total_tokens, hidden_dim] — concatenated real tokens
//   1: indices    [total_tokens] INT64 — flat index into the padded batch
//   2: cu_seqlens [batch+1] INT64 — cumulative sequence lengths (starts with 0)
//   3: max_seqlen_in_batch — INT64 scalar
//
#if NOT_EXCLUDED(OP_unpad_input)
CUSTOM_OP_IMPL(unpad_input, 2, 4, false, 0, 0) {
    auto hiddenStates  = INPUT_VARIABLE(0);   // [batch, max_seq_len, hidden_dim]
    auto attentionMask = INPUT_VARIABLE(1);   // [batch, max_seq_len]

    auto unpadded      = OUTPUT_VARIABLE(0);  // [total_tokens, hidden_dim]
    auto indices       = OUTPUT_VARIABLE(1);  // [total_tokens] INT64
    auto cuSeqlens     = OUTPUT_VARIABLE(2);  // [batch+1] INT64
    auto maxSeqlen     = OUTPUT_VARIABLE(3);  // scalar INT64

    REQUIRE_TRUE(hiddenStates->rankOf() == 3, 0,
        "unpad_input: hidden_states must be rank 3, got %d", hiddenStates->rankOf());
    REQUIRE_TRUE(attentionMask->rankOf() == 2, 0,
        "unpad_input: attention_mask must be rank 2, got %d", attentionMask->rankOf());

    const LongType batch      = hiddenStates->sizeAt(0);
    const LongType maxSeqLen  = hiddenStates->sizeAt(1);
    const LongType hiddenDim  = hiddenStates->sizeAt(2);

    REQUIRE_TRUE(attentionMask->sizeAt(0) == batch, 0,
        "unpad_input: attention_mask batch (%lld) != hidden_states batch (%lld)",
        (long long)attentionMask->sizeAt(0), (long long)batch);
    REQUIRE_TRUE(attentionMask->sizeAt(1) == maxSeqLen, 0,
        "unpad_input: attention_mask seq_len (%lld) != hidden_states seq_len (%lld)",
        (long long)attentionMask->sizeAt(1), (long long)maxSeqLen);

    // ----------------------------------------------------------------
    // Pass 1: count real tokens per sequence and total, find max_seqlen
    // ----------------------------------------------------------------
    std::vector<LongType> seqLens(batch, 0);
    LongType totalTokens = 0;
    LongType maxSeqLenVal = 0;

    for (LongType b = 0; b < batch; ++b) {
        LongType count = 0;
        for (LongType s = 0; s < maxSeqLen; ++s) {
            // attentionMask may be BOOL or INT32
            LongType maskVal = attentionMask->e<LongType>(b * maxSeqLen + s);
            if (maskVal != 0) count++;
        }
        seqLens[b] = count;
        totalTokens += count;
        if (count > maxSeqLenVal) maxSeqLenVal = count;
    }

    // ----------------------------------------------------------------
    // Write cu_seqlens: prefix sum [0, seqLens[0], seqLens[0]+seqLens[1], ...]
    // ----------------------------------------------------------------
    LongType* cuSeqPtr = cuSeqlens->bufferAsT<LongType>();
    cuSeqPtr[0] = 0;
    for (LongType b = 0; b < batch; ++b) {
        cuSeqPtr[b + 1] = cuSeqPtr[b] + seqLens[b];
    }

    // ----------------------------------------------------------------
    // Write max_seqlen scalar
    // ----------------------------------------------------------------
    maxSeqlen->p(0, maxSeqLenVal);

    // ----------------------------------------------------------------
    // Pass 2: copy real token rows and record flat indices.
    // hiddenStates and unpadded are C-order (guaranteed by DECLARE_SHAPE_FN),
    // so we compute byte offsets directly and use memcpy for efficiency.
    // ----------------------------------------------------------------
    LongType* idxPtr = indices->bufferAsT<LongType>();
    const LongType typeBytes = static_cast<LongType>(hiddenStates->sizeOfT());
    const LongType rowBytes  = hiddenDim * typeBytes;

    // Strides for hiddenStates [batch, max_seq_len, hidden_dim] in C-order
    const LongType strideB = hiddenStates->strideAt(0);  // max_seq_len * hidden_dim
    const LongType strideS = hiddenStates->strideAt(1);  // hidden_dim

    // Stride for unpadded [worst_case_total, hidden_dim] in C-order
    const LongType strideOut = unpadded->strideAt(0);    // hidden_dim

    auto srcBuf = reinterpret_cast<const char*>(hiddenStates->buffer());
    auto dstBuf = reinterpret_cast<char*>(unpadded->buffer());

    LongType outRow = 0;

    for (LongType b = 0; b < batch; ++b) {
        for (LongType s = 0; s < maxSeqLen; ++s) {
            LongType maskVal = attentionMask->e<LongType>(b * maxSeqLen + s);
            if (maskVal != 0) {
                // flat index into hidden_states: b * max_seq_len + s
                idxPtr[outRow] = b * maxSeqLen + s;

                // memcpy one row of hiddenDim elements
                const char* src = srcBuf + (b * strideB + s * strideS) * typeBytes;
                char*       dst = dstBuf + (outRow * strideOut)         * typeBytes;
                std::memcpy(dst, src, rowBytes);

                ++outRow;
            }
        }
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(unpad_input) {
    auto hiddenShape = inputShape->at(0);  // [batch, max_seq_len, hidden_dim]
    auto maskShape   = inputShape->at(1);  // [batch, max_seq_len]

    const LongType batch     = shape::sizeAt(hiddenShape, static_cast<LongType>(0));
    const LongType maxSeqLen = shape::sizeAt(hiddenShape, static_cast<LongType>(1));
    const LongType hiddenDim = shape::sizeAt(hiddenShape, static_cast<LongType>(2));
    const auto dtype = ArrayOptions::dataType(hiddenShape);

    // total_tokens is data-dependent; use an upper-bound shape [batch*max_seq_len, hidden_dim]
    // The actual contents are written in execute; shape is worst-case.
    const LongType maxTotal = batch * maxSeqLen;

    // output 0: unpadded [total_tokens, hidden_dim] — worst-case shape
    auto unpaddedShape = ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {maxTotal, hiddenDim});

    // output 1: indices [total_tokens] INT64
    auto indicesShape = ConstantShapeHelper::getInstance().createShapeInfo(
        DataType::INT64, 'c', {maxTotal});

    // output 2: cu_seqlens [batch+1] INT64
    auto cuSeqlensShape = ConstantShapeHelper::getInstance().createShapeInfo(
        DataType::INT64, 'c', {batch + 1});

    // output 3: max_seqlen scalar INT64
    auto maxSeqlenShape = ConstantShapeHelper::getInstance().createShapeInfo(
        DataType::INT64, 'c', {1});

    return SHAPELIST(unpaddedShape, indicesShape, cuSeqlensShape, maxSeqlenShape);
}

DECLARE_TYPES(unpad_input) {
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_DATA_DEPENDENT);
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {DataType::BOOL, DataType::INT32, DataType::INT64})
        ->setAllowedOutputTypes(0, {ALL_FLOATS})
        ->setAllowedOutputTypes(1, {DataType::INT64})
        ->setAllowedOutputTypes(2, {DataType::INT64})
        ->setAllowedOutputTypes(3, {DataType::INT64});
}
#endif  // OP_unpad_input

//////////////////////////////////////////////////////////////////////////
// pad_input
// Reverses unpad_input: scatters unpadded tokens back to padded positions.
//
// Inputs:
//   0: unpadded   [total_tokens, hidden_dim]
//   1: indices    [total_tokens] INT64 — from unpad_input output 1
//   2: batch_size INT64 scalar
//   3: max_seq_len INT64 scalar
//
// Outputs:
//   0: padded [batch, max_seq_len, hidden_dim] — zero-filled, real tokens placed back
//
#if NOT_EXCLUDED(OP_pad_input)
CUSTOM_OP_IMPL(pad_input, 4, 1, false, 0, 0) {
    auto unpadded   = INPUT_VARIABLE(0);  // [total_tokens, hidden_dim]
    auto indices    = INPUT_VARIABLE(1);  // [total_tokens] INT64
    auto batchSize  = INPUT_VARIABLE(2);  // scalar
    auto maxSeqLen  = INPUT_VARIABLE(3);  // scalar

    auto padded = OUTPUT_VARIABLE(0);     // [batch, max_seq_len, hidden_dim]

    REQUIRE_TRUE(unpadded->rankOf() == 2, 0,
        "pad_input: unpadded must be rank 2, got %d", unpadded->rankOf());
    REQUIRE_TRUE(indices->rankOf() == 1, 0,
        "pad_input: indices must be rank 1, got %d", indices->rankOf());

    const LongType totalTokens = unpadded->sizeAt(0);
    const LongType hiddenDim   = unpadded->sizeAt(1);
    const LongType batch       = batchSize->e<LongType>(0);
    const LongType maxSeq      = maxSeqLen->e<LongType>(0);

    // Zero-fill output
    double zero = 0.0;
    padded->assign(zero);

    const LongType* idxPtr    = indices->bufferAsT<LongType>();
    const LongType typeBytes  = static_cast<LongType>(unpadded->sizeOfT());
    const LongType rowBytes   = hiddenDim * typeBytes;

    // unpadded is C-order [totalTokens, hiddenDim]; padded is C-order [batch, maxSeq, hiddenDim]
    const LongType strideUnpad    = unpadded->strideAt(0);  // hiddenDim elements per row
    // indices[i] = b * maxSeq + s; padded[b,s,:] byte offset = flatIdx * strideAt(1) * typeBytes
    const LongType paddedStrideBs = padded->strideAt(1);    // hiddenDim elements per (b,s) row

    auto srcBuf = reinterpret_cast<const char*>(unpadded->buffer());
    auto dstBuf = reinterpret_cast<char*>(padded->buffer());

    PRAGMA_OMP_PARALLEL_FOR
    for (LongType i = 0; i < totalTokens; ++i) {
        LongType flatIdx = idxPtr[i];  // b * maxSeq + s
        const char* src = srcBuf + (i       * strideUnpad) * typeBytes;
        char*       dst = dstBuf + (flatIdx * paddedStrideBs) * typeBytes;
        std::memcpy(dst, src, rowBytes);
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(pad_input) {
    auto unpaddedShape = inputShape->at(0);  // [total_tokens, hidden_dim]
    auto batchArr      = INPUT_VARIABLE(2);
    auto maxSeqArr     = INPUT_VARIABLE(3);

    const LongType batch      = batchArr->e<LongType>(0);
    const LongType maxSeq     = maxSeqArr->e<LongType>(0);
    const LongType hiddenDim  = shape::sizeAt(unpaddedShape, static_cast<LongType>(1));
    const auto dtype = ArrayOptions::dataType(unpaddedShape);

    auto paddedShape = ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {batch, maxSeq, hiddenDim});

    return SHAPELIST(paddedShape);
}

DECLARE_TYPES(pad_input) {
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_VALUE_DEPENDENT_SHAPE);
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {DataType::INT64})
        ->setAllowedInputTypes(2, {DataType::INT64})
        ->setAllowedInputTypes(3, {DataType::INT64})
        ->setAllowedOutputTypes(0, {ALL_FLOATS});
}

//////////////////////////////////////////////////////////////////////////
// pad_input_bp
// Backward pass for pad_input: gather gradients from padded positions to
// unpadded positions (the reverse of the scatter in pad_input).
//
// Inputs:
//   0: grad_padded [batch, max_seq_len, hidden_dim]
//   1: indices     [total_tokens] INT64
//
// Outputs:
//   0: grad_unpadded [total_tokens, hidden_dim]
//
CUSTOM_OP_IMPL(pad_input_bp, 2, 1, false, 0, 0) {
    auto gradPadded   = INPUT_VARIABLE(0);  // [batch, max_seq_len, hidden_dim]
    auto indices      = INPUT_VARIABLE(1);  // [total_tokens] INT64
    auto gradUnpadded = OUTPUT_VARIABLE(0); // [total_tokens, hidden_dim]

    REQUIRE_TRUE(gradPadded->rankOf() == 3, 0,
        "pad_input_bp: grad_padded must be rank 3, got %d", gradPadded->rankOf());
    REQUIRE_TRUE(indices->rankOf() == 1, 0,
        "pad_input_bp: indices must be rank 1, got %d", indices->rankOf());

    const LongType totalTokens = indices->lengthOf();
    const LongType hiddenDim   = gradPadded->sizeAt(2);

    const LongType* idxPtr   = indices->bufferAsT<LongType>();
    const LongType typeBytes = static_cast<LongType>(gradPadded->sizeOfT());
    const LongType rowBytes  = hiddenDim * typeBytes;

    // gradPadded is C-order [batch, maxSeq, hiddenDim]; gradUnpadded is C-order [totalTokens, hiddenDim]
    const LongType strideGradPad   = gradPadded->strideAt(1);    // hiddenDim
    const LongType strideGradUnpad = gradUnpadded->strideAt(0);  // hiddenDim

    auto srcBuf = reinterpret_cast<const char*>(gradPadded->buffer());
    auto dstBuf = reinterpret_cast<char*>(gradUnpadded->buffer());

    // Gather: for each token i, copy from grad_padded at flat position indices[i]
    PRAGMA_OMP_PARALLEL_FOR
    for (LongType i = 0; i < totalTokens; ++i) {
        LongType flatIdx = idxPtr[i];  // b * maxSeq + s
        const char* src = srcBuf + (flatIdx * strideGradPad)   * typeBytes;
        char*       dst = dstBuf + (i       * strideGradUnpad) * typeBytes;
        std::memcpy(dst, src, rowBytes);
    }

    return Status::OK;
}

DECLARE_SHAPE_FN(pad_input_bp) {
    // Output shape: [total_tokens, hidden_dim]
    // total_tokens = indices.length, hidden_dim = gradPadded.sizeAt(2)
    auto gradPaddedShape = inputShape->at(0);
    auto indicesShape    = inputShape->at(1);

    const LongType totalTokens = shape::sizeAt(indicesShape, static_cast<LongType>(0));
    const LongType hiddenDim   = shape::sizeAt(gradPaddedShape, static_cast<LongType>(2));
    const auto dtype = ArrayOptions::dataType(gradPaddedShape);

    auto outShape = ConstantShapeHelper::getInstance().createShapeInfo(
        dtype, 'c', {totalTokens, hiddenDim});

    return SHAPELIST(outShape);
}

DECLARE_TYPES(pad_input_bp) {
  getOpDescriptor()->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_VALUE_DEPENDENT_SHAPE | OP_TRAIT_BACKWARD);
    getOpDescriptor()
        ->setAllowedInputTypes(0, {ALL_FLOATS})
        ->setAllowedInputTypes(1, {DataType::INT64})
        ->setAllowedOutputTypes(0, {ALL_FLOATS});
}
#endif  // OP_pad_input

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED guards
