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

#include <system/op_boilerplate.h>

#if NOT_EXCLUDED(OP_paged_attention_forward)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/paged_attention.h>

namespace sd {
namespace ops {

// ──────────────────────────────────────────────────────────────────────────
// paged_attention_forward
//
// Computes scaled dot-product attention with K/V gathered from a block pool
// via page table indirection. Optimized for decode (query seqLen = 1).
//
// Input 0: query          [batch, 1, numHeads, headDim]
// Input 1: keyBlockPool   [numBlocks, blockSize, numKvHeads, headDim]
// Input 2: valueBlockPool [numBlocks, blockSize, numKvHeads, headDim]
// Input 3: pageTables     [batch, maxBlocksPerSeq] int32
// Input 4: contextLens    [batch] int32
//
// IArgs: 0=blockSize, 1=numHeads, 2=numKvHeads, 3=headDim
// TArgs: 0=scale (0 = auto)
//
// Output 0: [batch, 1, numHeads, headDim]
// ──────────────────────────────────────────────────────────────────────────
CUSTOM_OP_IMPL(paged_attention_forward, 5, 1, false, 1, 4) {
  auto query          = INPUT_VARIABLE(0);
  auto keyBlockPool   = INPUT_VARIABLE(1);
  auto valueBlockPool = INPUT_VARIABLE(2);
  auto pageTables     = INPUT_VARIABLE(3);
  auto contextLens    = INPUT_VARIABLE(4);
  auto output         = OUTPUT_VARIABLE(0);

  int blockSize  = INT_ARG(0);
  int numHeads   = INT_ARG(1);
  int numKvHeads = INT_ARG(2);
  int headDim    = INT_ARG(3);
  float scale    = static_cast<float>(T_ARG(0));

  REQUIRE_TRUE(query->rankOf() == 4, 0,
               "paged_attention_forward: query must be rank 4, got %d", query->rankOf());
  REQUIRE_TRUE(keyBlockPool->rankOf() == 4, 0,
               "paged_attention_forward: keyBlockPool must be rank 4, got %d", keyBlockPool->rankOf());
  REQUIRE_TRUE(valueBlockPool->rankOf() == 4, 0,
               "paged_attention_forward: valueBlockPool must be rank 4, got %d", valueBlockPool->rankOf());
  REQUIRE_TRUE(pageTables->rankOf() == 2, 0,
               "paged_attention_forward: pageTables must be rank 2, got %d", pageTables->rankOf());
  REQUIRE_TRUE(contextLens->rankOf() == 1, 0,
               "paged_attention_forward: contextLens must be rank 1, got %d", contextLens->rankOf());
  REQUIRE_TRUE(blockSize > 0, 0,
               "paged_attention_forward: blockSize must be > 0, got %d", blockSize);

  helpers::PagedAttentionConfig config;
  config.blockSize  = blockSize;
  config.numHeads   = numHeads;
  config.numKvHeads = numKvHeads;
  config.headDim    = headDim;
  config.scale      = scale;
  config.isCausal   = true;

  helpers::pagedAttentionForward(query, keyBlockPool, valueBlockPool,
                                  pageTables, contextLens, output,
                                  config, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(paged_attention_forward) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {INT32})
      ->setAllowedInputTypes(4, {INT32})
      ->setAllowedOutputTypes(0, {ALL_FLOATS})
      ->addTraits(OP_TRAIT_ATTENTION | OP_TRAIT_FULLY_WRITING);
}

DECLARE_SHAPE_FN(paged_attention_forward) {
  // Output shape = same as query: [batch, 1, numHeads, headDim]
  auto queryShape = inputShape->at(0);
  auto rank = shape::rank(queryShape);
  std::vector<sd::LongType> outShape(rank);
  for (int i = 0; i < rank; i++) {
    outShape[i] = shape::sizeAt(queryShape, static_cast<sd::LongType>(i));
  }
  return SHAPELIST(ConstantShapeHelper::getInstance().createShapeInfo(
      ArrayOptions::dataType(queryShape), shape::order(queryShape), outShape));
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_paged_attention_forward)


#if NOT_EXCLUDED(OP_paged_kv_append)

#include <ops/declarable/headers/nn.h>
#include <ops/declarable/helpers/paged_attention.h>

namespace sd {
namespace ops {

// ──────────────────────────────────────────────────────────────────────────
// paged_kv_append
//
// Appends new KV entries to a paged block pool via page table indirection.
//
// Input 0: keyBlockPool   [numBlocks, blockSize, numKvHeads, headDim] (modified)
// Input 1: valueBlockPool [numBlocks, blockSize, numKvHeads, headDim] (modified)
// Input 2: newKeys        [batch, newLen, numKvHeads, headDim]
// Input 3: newValues      [batch, newLen, numKvHeads, headDim]
// Input 4: pageTables     [batch, maxBlocksPerSeq] int32
// Input 5: contextLens    [batch] int32 (lengths BEFORE append)
//
// IArgs: 0=blockSize
//
// Output 0: keyBlockPool (view of input 0, modified in-place)
// ──────────────────────────────────────────────────────────────────────────
CUSTOM_OP_IMPL(paged_kv_append, 6, 1, false, 0, 1) {
  auto keyBlockPool   = INPUT_VARIABLE(0);
  auto valueBlockPool = INPUT_VARIABLE(1);
  auto newKeys        = INPUT_VARIABLE(2);
  auto newValues      = INPUT_VARIABLE(3);
  auto pageTables     = INPUT_VARIABLE(4);
  auto contextLens    = INPUT_VARIABLE(5);

  int blockSize = INT_ARG(0);

  REQUIRE_TRUE(keyBlockPool->rankOf() == 4, 0,
               "paged_kv_append: keyBlockPool must be rank 4, got %d", keyBlockPool->rankOf());
  REQUIRE_TRUE(valueBlockPool->rankOf() == 4, 0,
               "paged_kv_append: valueBlockPool must be rank 4, got %d", valueBlockPool->rankOf());
  REQUIRE_TRUE(newKeys->rankOf() == 4, 0,
               "paged_kv_append: newKeys must be rank 4, got %d", newKeys->rankOf());
  REQUIRE_TRUE(newValues->rankOf() == 4, 0,
               "paged_kv_append: newValues must be rank 4, got %d", newValues->rankOf());
  REQUIRE_TRUE(blockSize > 0, 0,
               "paged_kv_append: blockSize must be > 0, got %d", blockSize);

  helpers::pagedKvCacheAppend(keyBlockPool, valueBlockPool,
                               newKeys, newValues,
                               pageTables, contextLens,
                               blockSize, block.launchContext());

  return sd::Status::OK;
}

DECLARE_TYPES(paged_kv_append) {
  getOpDescriptor()
      ->setAllowedInputTypes(0, {ALL_FLOATS})
      ->setAllowedInputTypes(1, {ALL_FLOATS})
      ->setAllowedInputTypes(2, {ALL_FLOATS})
      ->setAllowedInputTypes(3, {ALL_FLOATS})
      ->setAllowedInputTypes(4, {INT32})
      ->setAllowedInputTypes(5, {INT32})
      ->setAllowedOutputTypes(0, {ALL_FLOATS})
      ->addTraits(OP_TRAIT_DATA_MOVEMENT | OP_TRAIT_FULLY_WRITING | OP_TRAIT_VALUE_DEPENDENT_SHAPE);
}

DECLARE_SHAPE_FN(paged_kv_append) {
  // Output = view of input 0 (keyBlockPool)
  auto inShape = inputShape->at(0);
  auto rank = shape::rank(inShape);
  auto dtype = ArrayOptions::dataType(inShape);

  auto newShapeInfo = new LongType[shape::shapeInfoLength(rank)];
  memcpy(newShapeInfo, inShape, shape::shapeInfoByteLength(rank));
  ArrayOptions::resetFlags(newShapeInfo);
  ArrayOptions::setDataType(newShapeInfo, dtype);
  ArrayOptions::togglePropertyBit(newShapeInfo, ARRAY_COPY_OFFSET_INPUT_0);

  auto result = ConstantShapeHelper::getInstance().createFromExisting(newShapeInfo);
  delete[] newShapeInfo;
  return SHAPELIST(result);
}

}  // namespace ops
}  // namespace sd

#endif  // NOT_EXCLUDED(OP_paged_kv_append)
