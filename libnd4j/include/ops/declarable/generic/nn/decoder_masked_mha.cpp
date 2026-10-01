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
// decoder_masked_mha - masked multi-head attention for autoregressive decoders
//
// Five inputs (hidden states, fused QKV weight, output weight, past keys and
// values): the hidden states project through the fused [q | k | v] weight, the
// queries and keys rotate by their absolute positions (past + s), the current
// keys and values append to the past into the present cache, and the queries
// attend causally over the present cache before the output projection.
//
// Three inputs (queries, keys and values with the heads packed in the last
// axis): the queries attend over the keys and values, causally when requested,
// and the keys and values return per head as the present cache.
//
// Masked scores are biased by the mask filter value, saturated into the finite
// range of the attention type, so a masked key gets no weight and every score
// stays finite.
//

#include <system/op_boilerplate.h>
#if NOT_EXCLUDED(OP_decoder_masked_mha)
#include <array/NDArrayFactory.h>
#include <helpers/ConstantShapeHelper.h>
#include <helpers/FlashAttentionHelper.h>
#include <helpers/MmulHelper.h>
#include <math/templatemath.h>
#include <ops/declarable/CustomOperations.h>
#include <ops/declarable/headers/llm.h>
#include <ops/declarable/helpers/fused_llm_ops.h>
#include <ops/declarable/helpers/int8_gemm.h>

#include <vector>

namespace sd {
namespace ops {

// Extents of one decoder_masked_mha call; the op and its shape function resolve
// them the same way.
struct DecoderMaskedMhaGeometry {
  LongType batch = 0;
  LongType querySeq = 0;
  LongType pastSeq = 0;
  LongType totalSeq = 0;
  LongType numHeads = 0;
  LongType numKvHeads = 0;
  LongType headDim = 0;
  LongType outputWidth = 0;
};

static DecoderMaskedMhaGeometry decoderMaskedMhaGeometry(graph::Context& block,
                                                         const std::vector<const LongType*>& shapes) {
  const int width = static_cast<int>(shapes.size());
  REQUIRE_TRUE(width == 3 || width == 5, 0,
               "decoder_masked_mha: expected queries, keys and values (3 inputs), or hidden states, the fused QKV "
               "weight, the output weight and the past keys and values (5 inputs), got %i inputs",
               width);
  REQUIRE_TRUE(block.numI() > 0, 0, "decoder_masked_mha: the number of heads (iArg 0) is required");

  DecoderMaskedMhaGeometry g;
  g.numHeads = INT_ARG(0);
  REQUIRE_TRUE(g.numHeads > 0, 0, "decoder_masked_mha: the number of heads must be positive, got %lld", g.numHeads);
  const LongType requestedKvHeads = INT_ARG_OR(1, 0);
  const LongType requestedHeadDim = INT_ARG_OR(2, 0);

  const LongType* first = shapes[0];
  REQUIRE_TRUE(shape::rank(first) == 3, 0, "decoder_masked_mha: %s must be [batch, sequence, hidden], got rank %i",
               width == 5 ? "the hidden states" : "the queries", shape::rank(first));
  g.batch = shape::sizeAt(first, 0);
  g.querySeq = shape::sizeAt(first, 1);

  if (width == 5) {
    const LongType* qkvWeight = shapes[1];
    const LongType* outWeight = shapes[2];
    const LongType* pastKey = shapes[3];
    const LongType* pastValue = shapes[4];
    const LongType hidden = shape::sizeAt(first, 2);
    REQUIRE_TRUE(shape::rank(qkvWeight) == 2 && shape::sizeAt(qkvWeight, 0) == hidden, 0,
                 "decoder_masked_mha: the fused QKV weight must be [%lld, (heads + 2 * kvHeads) * headDim]", hidden);
    const LongType fused = shape::sizeAt(qkvWeight, 1);
    REQUIRE_TRUE(shape::rank(pastKey) == 4, 0,
                 "decoder_masked_mha: the past keys must be [batch, kvHeads, past, headDim], got rank %i",
                 shape::rank(pastKey));
    // The past keys carry the number of key/value heads unless it is given.
    g.numKvHeads = requestedKvHeads > 0 ? requestedKvHeads : shape::sizeAt(pastKey, 1);
    REQUIRE_TRUE(g.numKvHeads > 0, 0, "decoder_masked_mha: the number of key/value heads must be positive, got %lld",
                 g.numKvHeads);
    g.headDim = requestedHeadDim > 0 ? requestedHeadDim : fused / (g.numHeads + 2 * g.numKvHeads);
    REQUIRE_TRUE(g.headDim > 0 && fused == (g.numHeads + 2 * g.numKvHeads) * g.headDim, 0,
                 "decoder_masked_mha: the fused QKV weight has %lld columns, expected (%lld + 2 * %lld) * %lld", fused,
                 g.numHeads, g.numKvHeads, g.headDim);
    REQUIRE_TRUE(shape::rank(outWeight) == 2 && shape::sizeAt(outWeight, 0) == g.numHeads * g.headDim, 0,
                 "decoder_masked_mha: the output weight must be [%lld, outputHidden]", g.numHeads * g.headDim);
    g.outputWidth = shape::sizeAt(outWeight, 1);
    REQUIRE_TRUE(shape::rank(pastKey) == 4 && shape::sizeAt(pastKey, 0) == g.batch &&
                     shape::sizeAt(pastKey, 1) == g.numKvHeads && shape::sizeAt(pastKey, 3) == g.headDim,
                 0, "decoder_masked_mha: the past keys must be [%lld, %lld, past, %lld]", g.batch, g.numKvHeads,
                 g.headDim);
    REQUIRE_TRUE(shape::shapeEquals(pastKey, pastValue), 0,
                 "decoder_masked_mha: the past keys and values must have the same shape");
    g.pastSeq = shape::sizeAt(pastKey, 2);
    g.totalSeq = g.pastSeq + g.querySeq;
  } else {
    const LongType* key = shapes[1];
    const LongType* value = shapes[2];
    const LongType queryWidth = shape::sizeAt(first, 2);
    REQUIRE_TRUE(shape::rank(key) == 3 && shape::sizeAt(key, 0) == g.batch, 0,
                 "decoder_masked_mha: the keys must be [%lld, sequence, kvHeads * headDim]", g.batch);
    REQUIRE_TRUE(shape::shapeEquals(key, value), 0, "decoder_masked_mha: the keys and values must have the same shape");
    g.headDim = requestedHeadDim > 0 ? requestedHeadDim : queryWidth / g.numHeads;
    REQUIRE_TRUE(g.headDim > 0 && queryWidth == g.numHeads * g.headDim, 0,
                 "decoder_masked_mha: the queries hold %lld values per position, expected %lld heads of the head "
                 "dimension",
                 queryWidth, g.numHeads);
    const LongType kvWidth = shape::sizeAt(key, 2);
    g.numKvHeads = requestedKvHeads > 0 ? requestedKvHeads : kvWidth / g.headDim;
    REQUIRE_TRUE(g.numKvHeads > 0 && kvWidth == g.numKvHeads * g.headDim, 0,
                 "decoder_masked_mha: the keys hold %lld values per position, expected a multiple of the head "
                 "dimension %lld",
                 kvWidth, g.headDim);
    g.totalSeq = shape::sizeAt(key, 1);
    g.outputWidth = queryWidth;
  }
  REQUIRE_TRUE(g.numHeads % g.numKvHeads == 0, 0,
               "decoder_masked_mha: %lld query heads do not group evenly over %lld key/value heads", g.numHeads,
               g.numKvHeads);
  return g;
}

// Masks every score whose key lies past visibleOffset + the query's row: the
// mask filter saturates into float, then into T, so it stays finite in both.
template <typename T>
static void decoderMaskedMhaFilterBias_(NDArray* bias, LongType visibleOffset, double filterValue) {
  const float filter =
      static_cast<float>(sd::math::sd_saturate<float, T>(sd::math::sd_saturate<double, float>(filterValue)));
  bias->nullify();
  bias->fillAsTriangular<T>(filter, static_cast<int>(visibleOffset + 1), 0, *bias, 'u', true);
}

// query [B, Sq, heads, headDim] attends over key/value [B, Skv, kvHeads, headDim]
// into attention; a causal query at row s sees the keys up to Skv - Sq + s.
static void decoderMaskedMhaAttend(graph::Context& block, const DecoderMaskedMhaGeometry& g, NDArray* query,
                                   NDArray* key, NDArray* value, NDArray* attention, bool causal) {
  auto* context = block.launchContext();
  NDArray* bias = nullptr;
  if (causal && g.querySeq > 1) {
    std::vector<LongType> biasShape = {g.querySeq, g.totalSeq};
    bias = new NDArray('c', biasShape, query->dataType(), context);
    const double filterValue = T_ARG_OR(0, -static_cast<double>(DataTypeUtils::max<float>()));
    BUILD_SINGLE_SELECTOR(bias->dataType(), decoderMaskedMhaFilterBias_,
                          (bias, g.totalSeq - g.querySeq, filterValue), SD_FLOAT_TYPES);
  }

  // The mask bias carries the causal structure, so the kernel's own mask stays off.
  FlashAttentionHelper::Config config;
  config.scale = 0.0f;
  config.isCausal = false;
  config.numHeads = static_cast<int>(g.numHeads);
  config.numKvHeads = static_cast<int>(g.numKvHeads);
  FlashAttentionHelper::forward(query, key, value, attention, config, nullptr, nullptr, nullptr, context, bias);
  if (bias != nullptr) MmulHelper::deleteTemporary(bias);
}

// A fresh c-order copy of packed [B, S, heads * headDim] in dataType, split into
// [B, S, heads, headDim]: the copy follows packed's strides in packed's own shape
// (converting its storage type), and the contiguous copy then splits its last
// axis into heads.
static NDArray* decoderMaskedMhaHeads(NDArray* packed, LongType heads, LongType headDim, DataType dataType,
                                      LaunchContext* context) {
  std::vector<LongType> packedShape(packed->shapeOf(), packed->shapeOf() + packed->rankOf());
  auto* split = new NDArray('c', packedShape, dataType, context);
  split->assign(packed);
  split->reshapei('c', {packed->sizeAt(0), packed->sizeAt(1), heads, headDim});
  return split;
}

CUSTOM_OP_IMPL(decoder_masked_mha, 3, 3, false, 0, 1) {
  const int width = block.width();
  std::vector<const LongType*> shapes;
  for (int i = 0; i < width; i++) shapes.push_back(INPUT_VARIABLE(i)->shapeInfo());
  const DecoderMaskedMhaGeometry g = decoderMaskedMhaGeometry(block, shapes);
  for (int i = 1; i < width; i++) {
    REQUIRE_TRUE(DataTypeUtils::isR(INPUT_VARIABLE(i)->dataType()), 0,
                 "decoder_masked_mha: input %i must hold floating values, got %s", i,
                 DataTypeUtils::asString(INPUT_VARIABLE(i)->dataType()).c_str());
  }

  auto* context = block.launchContext();
  auto output = OUTPUT_VARIABLE(0);
  auto presentKey = OUTPUT_VARIABLE(1);
  auto presentValue = OUTPUT_VARIABLE(2);
  const DataType dtype = INPUT_VARIABLE(0)->dataType();
  const LongType B = g.batch;
  const LongType S = g.querySeq;
  const LongType nH = g.numHeads;
  const LongType nKV = g.numKvHeads;
  const LongType hd = g.headDim;
  std::vector<LongType> queryHeadsShape = {B, S, nH, hd};
  std::vector<LongType> keyHeadsShape = {B, S, nKV, hd};
  const std::vector<LongType> presentShape = {B, nKV, g.totalSeq, hd};
  const std::vector<LongType> outputShape = {B, S, g.outputWidth};
  REQUIRE_TRUE(output->isSameShape(outputShape), 0, "decoder_masked_mha: output 0 must be [%lld, %lld, %lld]", B, S,
               g.outputWidth);
  REQUIRE_TRUE(presentKey->isSameShape(presentShape) && presentValue->isSameShape(presentShape), 0,
               "decoder_masked_mha: the present keys and values must be [%lld, %lld, %lld, %lld]", B, nKV,
               g.totalSeq, hd);
  REQUIRE_TRUE(output->dataType() == dtype && presentKey->dataType() == dtype && presentValue->dataType() == dtype,
               0, "decoder_masked_mha: the outputs must hold the data type of input 0 (%s)",
               DataTypeUtils::asString(dtype).c_str());
  REQUIRE_TRUE(B == 0 || S == 0 || g.totalSeq > 0, 0,
               "decoder_masked_mha: the queries need at least one key to attend over");
  // An empty cache leaves no query to answer.
  if (presentKey->isEmpty()) return Status::OK;

  // The present cache viewed position-major, [B, total, kvHeads, headDim].
  std::vector<LongType> toPositionMajor = {0, 2, 1, 3};
  NDArray* keyCache = presentKey->permute(toPositionMajor, false, false);
  NDArray* valueCache = presentValue->permute(toPositionMajor, false, false);

  if (width == 3) {
    const bool causal = INT_ARG_OR(3, 0) != 0;
    REQUIRE_TRUE(!causal || S <= g.totalSeq, 0,
                 "decoder_masked_mha: causal attention needs at least as many keys (%lld) as queries (%lld)",
                 g.totalSeq, S);
    // The keys and values split their heads out of the last axis into the cache.
    NDArray* keyHeads = decoderMaskedMhaHeads(INPUT_VARIABLE(1), nKV, hd, dtype, context);
    keyCache->assign(keyHeads);
    MmulHelper::deleteTemporary(keyHeads);
    NDArray* valueHeads = decoderMaskedMhaHeads(INPUT_VARIABLE(2), nKV, hd, dtype, context);
    valueCache->assign(valueHeads);
    MmulHelper::deleteTemporary(valueHeads);

    if (S > 0) {
      NDArray* query = decoderMaskedMhaHeads(INPUT_VARIABLE(0), nH, hd, dtype, context);
      auto* attention = new NDArray('c', queryHeadsShape, dtype, context);
      decoderMaskedMhaAttend(block, g, query, keyCache, valueCache, attention, causal);
      MmulHelper::deleteTemporary(query);
      attention->reshapei('c', {B, S, nH * hd});
      output->assign(attention);
      MmulHelper::deleteTemporary(attention);
    }
  } else {
    auto hidden = INPUT_VARIABLE(0);
    auto qkvWeight = INPUT_VARIABLE(1);
    auto outWeight = INPUT_VARIABLE(2);
    auto pastKey = INPUT_VARIABLE(3);
    auto pastValue = INPUT_VARIABLE(4);
    const int useRoPE = static_cast<int>(INT_ARG_OR(3, 1));
    const float ropeBase = static_cast<float>(INT_ARG_OR(4, 10000));
    REQUIRE_TRUE(useRoPE >= 0 && useRoPE <= 2, 0,
                 "decoder_masked_mha: useRoPE must be 0 (none), 1 (rotate halves) or 2 (rotate interleaved pairs), "
                 "got %i",
                 useRoPE);
    REQUIRE_TRUE(useRoPE == 0 || hd % 2 == 0, 0,
                 "decoder_masked_mha: rotary embedding needs an even head dimension, got %lld", hd);

    // The past keys and values lead the present cache.
    if (g.pastSeq > 0) {
      std::vector<LongType> pastIdx = {0, B, 0, nKV, 0, g.pastSeq, 0, hd};
      NDArray* keyPast = (*presentKey)(pastIdx, true);
      NDArray* valuePast = (*presentValue)(pastIdx, true);
      keyPast->assign(pastKey);
      valuePast->assign(pastValue);
      delete keyPast;
      delete valuePast;
    }

    if (S > 0) {
      // The current keys and values follow the past in the cache.
      std::vector<LongType> currentIdx = {0, B, g.pastSeq, g.totalSeq, 0, nKV, 0, hd};
      NDArray* keyCurrent = (*keyCache)(currentIdx, true);
      NDArray* valueCurrent = (*valueCache)(currentIdx, true);

      // The fused projection holds the query, key and value heads side by side.
      const LongType fusedHeads = nH + 2 * nKV;
      std::vector<LongType> qkvShape = {B * S, fusedHeads * hd};
      auto* qkv = new NDArray('c', qkvShape, dtype, context);
      helpers::scaledGemm(context, hidden, qkvWeight, nullptr, nullptr, nullptr, qkv, false, false);
      qkv->reshapei('c', {B, S, fusedHeads, hd});
      std::vector<LongType> queryIdx = {0, B, 0, S, 0, nH, 0, hd};
      std::vector<LongType> keyIdx = {0, B, 0, S, nH, nH + nKV, 0, hd};
      std::vector<LongType> valueIdx = {0, B, 0, S, nH + nKV, fusedHeads, 0, hd};
      NDArray* queryProjected = (*qkv)(queryIdx, true);
      NDArray* keyProjected = (*qkv)(keyIdx, true);
      NDArray* valueProjected = (*qkv)(valueIdx, true);
      valueCurrent->assign(valueProjected);

      auto* query = new NDArray('c', queryHeadsShape, dtype, context);
      if (useRoPE == 0) {
        query->assign(queryProjected);
        keyCurrent->assign(keyProjected);
      } else {
        // useRoPE 1 rotates the halves of each head (i, i + hd/2), 2 rotates
        // interleaved pairs (2i, 2i + 1); the current positions follow the past.
        // The rotation reads and writes contiguous heads.
        const int ropeType = useRoPE == 1 ? 0 : 1;
        auto* position = NDArrayFactory::create_<LongType>(g.pastSeq, context);
        auto* queryUnrotated = new NDArray('c', queryHeadsShape, dtype, context);
        queryUnrotated->assign(queryProjected);
        helpers::fusedRoPE(queryUnrotated, query, position, ropeBase, 1.0f, ropeType, context);
        auto* keyUnrotated = new NDArray('c', keyHeadsShape, dtype, context);
        keyUnrotated->assign(keyProjected);
        auto* keyRotated = new NDArray('c', keyHeadsShape, dtype, context);
        helpers::fusedRoPE(keyUnrotated, keyRotated, position, ropeBase, 1.0f, ropeType, context);
        keyCurrent->assign(keyRotated);
        MmulHelper::deleteTemporary(queryUnrotated);
        MmulHelper::deleteTemporary(keyUnrotated);
        MmulHelper::deleteTemporary(keyRotated);
        MmulHelper::deleteTemporary(position);
      }
      delete queryProjected;
      delete keyProjected;
      delete valueProjected;
      delete keyCurrent;
      delete valueCurrent;
      MmulHelper::deleteTemporary(qkv);

      auto* attention = new NDArray('c', queryHeadsShape, dtype, context);
      decoderMaskedMhaAttend(block, g, query, keyCache, valueCache, attention, true);
      MmulHelper::deleteTemporary(query);
      attention->reshapei('c', {B * S, nH * hd});
      helpers::scaledGemm(context, attention, outWeight, nullptr, nullptr, nullptr, output, false, false);
      MmulHelper::deleteTemporary(attention);
    }
  }

  delete keyCache;
  delete valueCache;
  return Status::OK;
}

DECLARE_TYPES(decoder_masked_mha) {
  // Inputs 1-4 take any floating storage type, FP8 included (checked in the op).
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, DataType::ANY)
      ->setAllowedInputTypes(2, DataType::ANY)
      ->setAllowedInputTypes(3, DataType::ANY)
      ->setAllowedInputTypes(4, DataType::ANY)
      ->setAllowedOutputTypes({ALL_FLOATS})
      ->setShapeValueInputs({})
      ->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_EXTERNAL_WORKSPACE | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(decoder_masked_mha) {
  std::vector<const LongType*> shapes;
  for (int i = 0; i < inputShape->size(); i++) shapes.push_back(inputShape->at(i));
  const DecoderMaskedMhaGeometry g = decoderMaskedMhaGeometry(block, shapes);
  const DataType dtype = ArrayOptions::dataType(inputShape->at(0));
  auto output = ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {g.batch, g.querySeq, g.outputWidth});
  auto present =
      ConstantShapeHelper::getInstance().createShapeInfo(dtype, 'c', {g.batch, g.numKvHeads, g.totalSeq, g.headDim});
  return SHAPELIST(output, present, present);
}

}  // namespace ops
}  // namespace sd

#endif
