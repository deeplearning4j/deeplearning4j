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
// OneDNN Flash Attention implementation
// Uses OneDNN Graph API for fused SDPA with compiled partition caching
// Supports both 3D [batch, seq, dim] and 4D [batch, seq, heads, dim] inputs
//

#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <math/templatemath.h>

#include <oneapi/dnnl/dnnl_graph.hpp>

#include <unordered_map>
#include <mutex>

#include "mkldnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

namespace dg = dnnl::graph;

//////////////////////////////////////////////////////////////////////////
// Cache for compiled Flash Attention partitions
struct FlashAttentionCache {
  struct Key {
    int64_t batch, seqQ, seqKV, numHeads, headDim;
    int dtype;  // 0=f32, 1=f16, 2=bf16
    bool isCausal;
    bool is3D;

    bool operator==(const Key& o) const {
      return batch == o.batch && seqQ == o.seqQ && seqKV == o.seqKV &&
             numHeads == o.numHeads && headDim == o.headDim &&
             dtype == o.dtype && isCausal == o.isCausal && is3D == o.is3D;
    }
  };

  struct KeyHash {
    size_t operator()(const Key& k) const {
      size_t h = std::hash<int64_t>()(k.batch);
      h ^= std::hash<int64_t>()(k.seqQ) << 1;
      h ^= std::hash<int64_t>()(k.seqKV) << 2;
      h ^= std::hash<int64_t>()(k.numHeads) << 3;
      h ^= std::hash<int64_t>()(k.headDim) << 4;
      h ^= std::hash<int>()(k.dtype) << 5;
      h ^= std::hash<bool>()(k.isCausal) << 6;
      h ^= std::hash<bool>()(k.is3D) << 7;
      return h;
    }
  };

  struct Entry {
    dg::compiled_partition cp;
    bool valid = false;
  };

  std::unordered_map<Key, Entry, KeyHash> cache;
  std::mutex mtx;
  dnnl::engine eng{dnnl::engine::kind::cpu, 0};

  static FlashAttentionCache& instance() {
    static FlashAttentionCache inst;
    return inst;
  }
};

//////////////////////////////////////////////////////////////////////////
// Build and compile Flash Attention graph for 3D inputs
static FlashAttentionCache::Entry buildFlashAttention3D(
    int64_t batch, int64_t seqQ, int64_t seqKV, int64_t dim,
    dg::logical_tensor::data_type dtype, bool isCausal,
    dnnl::engine& eng) {
  FlashAttentionCache::Entry entry;

  size_t id = 0;

  // Shapes: [batch, seqQ/seqKV, dim] for Q,K,V
  std::vector<int64_t> q_shape = {batch, seqQ, dim};
  std::vector<int64_t> k_shape = {batch, seqKV, dim};
  std::vector<int64_t> v_shape = {batch, seqKV, dim};
  std::vector<int64_t> score_shape = {batch, seqQ, seqKV};
  std::vector<int64_t> out_shape = {batch, seqQ, dim};

  // Create logical tensors
  auto query_lt = dg::logical_tensor(id++, dtype, q_shape, dg::logical_tensor::layout_type::strided);
  auto key_lt = dg::logical_tensor(id++, dtype, k_shape, dg::logical_tensor::layout_type::strided);
  auto value_lt = dg::logical_tensor(id++, dtype, v_shape, dg::logical_tensor::layout_type::strided);
  auto score_lt = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32, score_shape,
                                      dg::logical_tensor::layout_type::strided);
  auto scale_lt = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32, {1},
                                      dg::logical_tensor::layout_type::strided);
  auto scaled_lt = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32, score_shape,
                                       dg::logical_tensor::layout_type::strided);
  auto probs_lt = dg::logical_tensor(id++, dtype, score_shape, dg::logical_tensor::layout_type::strided);
  auto output_lt = dg::logical_tensor(id++, dtype, out_shape, dg::logical_tensor::layout_type::strided);

  // Build ops: BMM1 (Q @ K^T), Scale, Softmax, BMM2 (probs @ V)
  dg::op bmm1(id++, dg::op::kind::MatMul, "bmm1");
  bmm1.set_attr<bool>(dg::op::attr::transpose_b, true);
  bmm1.add_inputs({query_lt, key_lt});
  bmm1.add_outputs({score_lt});

  dg::op scale_op(id++, dg::op::kind::Multiply, "scale");
  scale_op.add_inputs({score_lt, scale_lt});
  scale_op.add_outputs({scaled_lt});

  dg::op softmax_op(id++, dg::op::kind::SoftMax, "softmax");
  softmax_op.set_attr<int64_t>(dg::op::attr::axis, -1);
  softmax_op.add_inputs({scaled_lt});
  softmax_op.add_outputs({probs_lt});

  dg::op bmm2(id++, dg::op::kind::MatMul, "bmm2");
  bmm2.add_inputs({probs_lt, value_lt});
  bmm2.add_outputs({output_lt});

  // Build and finalize graph
  dg::graph g(dnnl::engine::kind::cpu);
  g.add_op(bmm1);
  g.add_op(scale_op);
  g.add_op(softmax_op);
  g.add_op(bmm2);
  g.finalize();

  // Get partitions and compile
  auto partitions = g.get_partitions();
  if (partitions.empty()) {
    entry.valid = false;
    return entry;
  }

  std::vector<dg::logical_tensor> inputs = {query_lt, key_lt, scale_lt, value_lt};
  std::vector<dg::logical_tensor> outputs = {output_lt};

  try {
    entry.cp = partitions[0].compile(inputs, outputs, eng);
    entry.valid = true;
  } catch (const std::exception& e) {
    entry.valid = false;
  }

  return entry;
}

//////////////////////////////////////////////////////////////////////////
// Build and compile Flash Attention graph for 4D inputs
static FlashAttentionCache::Entry buildFlashAttention4D(
    int64_t batch, int64_t seqQ, int64_t seqKV, int64_t numHeads, int64_t headDim,
    dg::logical_tensor::data_type dtype, bool isCausal,
    dnnl::engine& eng) {
  FlashAttentionCache::Entry entry;

  size_t id = 0;

  // For 4D, we reshape to [batch * numHeads, seq, headDim] internally
  int64_t batchHeads = batch * numHeads;
  std::vector<int64_t> q_shape = {batchHeads, seqQ, headDim};
  std::vector<int64_t> k_shape = {batchHeads, seqKV, headDim};
  std::vector<int64_t> v_shape = {batchHeads, seqKV, headDim};
  std::vector<int64_t> score_shape = {batchHeads, seqQ, seqKV};
  std::vector<int64_t> out_shape = {batchHeads, seqQ, headDim};

  auto query_lt = dg::logical_tensor(id++, dtype, q_shape, dg::logical_tensor::layout_type::strided);
  auto key_lt = dg::logical_tensor(id++, dtype, k_shape, dg::logical_tensor::layout_type::strided);
  auto value_lt = dg::logical_tensor(id++, dtype, v_shape, dg::logical_tensor::layout_type::strided);
  auto score_lt = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32, score_shape,
                                      dg::logical_tensor::layout_type::strided);
  auto scale_lt = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32, {1},
                                      dg::logical_tensor::layout_type::strided);
  auto scaled_lt = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32, score_shape,
                                       dg::logical_tensor::layout_type::strided);
  auto probs_lt = dg::logical_tensor(id++, dtype, score_shape, dg::logical_tensor::layout_type::strided);
  auto output_lt = dg::logical_tensor(id++, dtype, out_shape, dg::logical_tensor::layout_type::strided);

  dg::op bmm1(id++, dg::op::kind::MatMul, "bmm1");
  bmm1.set_attr<bool>(dg::op::attr::transpose_b, true);
  bmm1.add_inputs({query_lt, key_lt});
  bmm1.add_outputs({score_lt});

  dg::op scale_op(id++, dg::op::kind::Multiply, "scale");
  scale_op.add_inputs({score_lt, scale_lt});
  scale_op.add_outputs({scaled_lt});

  dg::op softmax_op(id++, dg::op::kind::SoftMax, "softmax");
  softmax_op.set_attr<int64_t>(dg::op::attr::axis, -1);
  softmax_op.add_inputs({scaled_lt});
  softmax_op.add_outputs({probs_lt});

  dg::op bmm2(id++, dg::op::kind::MatMul, "bmm2");
  bmm2.add_inputs({probs_lt, value_lt});
  bmm2.add_outputs({output_lt});

  dg::graph g(dnnl::engine::kind::cpu);
  g.add_op(bmm1);
  g.add_op(scale_op);
  g.add_op(softmax_op);
  g.add_op(bmm2);
  g.finalize();

  auto partitions = g.get_partitions();
  if (partitions.empty()) {
    entry.valid = false;
    return entry;
  }

  std::vector<dg::logical_tensor> inputs = {query_lt, key_lt, scale_lt, value_lt};
  std::vector<dg::logical_tensor> outputs = {output_lt};

  try {
    entry.cp = partitions[0].compile(inputs, outputs, eng);
    entry.valid = true;
  } catch (const std::exception& e) {
    entry.valid = false;
  }

  return entry;
}

//////////////////////////////////////////////////////////////////////////
// Get or create compiled Flash Attention partition
static FlashAttentionCache::Entry& getFlashAttention(
    int64_t batch, int64_t seqQ, int64_t seqKV, int64_t numHeads, int64_t headDim,
    int dtype, bool isCausal, bool is3D) {

  auto& cache = FlashAttentionCache::instance();
  FlashAttentionCache::Key key{batch, seqQ, seqKV, numHeads, headDim, dtype, isCausal, is3D};

  std::lock_guard<std::mutex> lock(cache.mtx);

  auto it = cache.cache.find(key);
  if (it != cache.cache.end() && it->second.valid) {
    return it->second;
  }

  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  if (is3D) {
    cache.cache[key] = buildFlashAttention3D(batch, seqQ, seqKV, headDim, dt, isCausal, cache.eng);
  } else {
    cache.cache[key] = buildFlashAttention4D(batch, seqQ, seqKV, numHeads, headDim, dt, isCausal, cache.eng);
  }
  return cache.cache[key];
}

//////////////////////////////////////////////////////////////////////////
// Execute Flash Attention for 3D inputs
static void executeFlashAttention3D(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
                                     float scale, bool isCausal, LaunchContext* context) {
  const auto batch = query->sizeAt(0);
  const auto seqQ = query->sizeAt(1);
  const auto seqKV = key->sizeAt(1);
  const auto dim = query->sizeAt(2);

  int dtype = 0;
  if (query->dataType() == DataType::HALF) dtype = 1;
  else if (query->dataType() == DataType::BFLOAT16) dtype = 2;

  auto& entry = getFlashAttention(batch, seqQ, seqKV, 1, dim, dtype, isCausal, true);
  if (!entry.valid) {
    THROW_EXCEPTION("Flash Attention graph compilation failed for 3D inputs");
  }

  auto& cache = FlashAttentionCache::instance();
  dnnl::stream strm(cache.eng);

  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  std::vector<int64_t> q_shape = {batch, seqQ, dim};
  std::vector<int64_t> k_shape = {batch, seqKV, dim};
  std::vector<int64_t> v_shape = {batch, seqKV, dim};
  std::vector<int64_t> out_shape = {batch, seqQ, dim};

  std::vector<int64_t> q_strides = {query->strideAt(0), query->strideAt(1), query->strideAt(2)};
  std::vector<int64_t> k_strides = {key->strideAt(0), key->strideAt(1), key->strideAt(2)};
  std::vector<int64_t> v_strides = {value->strideAt(0), value->strideAt(1), value->strideAt(2)};
  std::vector<int64_t> out_strides = {output->strideAt(0), output->strideAt(1), output->strideAt(2)};

  auto query_lt = dg::logical_tensor(0, dt, q_shape, q_strides);
  auto key_lt = dg::logical_tensor(1, dt, k_shape, k_strides);
  auto scale_lt = dg::logical_tensor(2, dg::logical_tensor::data_type::f32, {1}, {1});
  auto value_lt = dg::logical_tensor(3, dt, v_shape, v_strides);
  auto output_lt = dg::logical_tensor(4, dt, out_shape, out_strides);

  dg::tensor t_query(query_lt, cache.eng, query->buffer());
  dg::tensor t_key(key_lt, cache.eng, key->buffer());
  dg::tensor t_scale(scale_lt, cache.eng, &scale);
  dg::tensor t_value(value_lt, cache.eng, value->buffer());
  dg::tensor t_output(output_lt, cache.eng, output->buffer());

  entry.cp.execute(strm, {t_query, t_key, t_scale, t_value}, {t_output});
  strm.wait();
}

//////////////////////////////////////////////////////////////////////////
// Execute Flash Attention for 4D inputs
static void executeFlashAttention4D(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
                                     float scale, bool isCausal, LaunchContext* context) {
  const auto batch = query->sizeAt(0);
  const auto seqQ = query->sizeAt(1);
  const auto numHeads = query->sizeAt(2);
  const auto headDim = query->sizeAt(3);
  const auto seqKV = key->sizeAt(1);
  const auto numKvHeads = key->sizeAt(2);

  // For GQA support, we need to handle numHeads != numKvHeads
  // For now, only support MHA (numHeads == numKvHeads)
  if (numHeads != numKvHeads) {
    THROW_EXCEPTION("OneDNN Flash Attention: GQA (numHeads != numKvHeads) not yet supported");
  }

  int dtype = 0;
  if (query->dataType() == DataType::HALF) dtype = 1;
  else if (query->dataType() == DataType::BFLOAT16) dtype = 2;

  // Permute and reshape for OneDNN: [batch, seq, heads, dim] -> [batch*heads, seq, dim]
  std::vector<LongType> permOrder = {0, 2, 1, 3};
  auto qPerm = query->permute(permOrder, false, false);
  auto kPerm = key->permute(permOrder, false, false);
  auto vPerm = value->permute(permOrder, false, false);

  int64_t batchHeads = batch * numHeads;
  std::vector<LongType> reshapeShape = {batchHeads, seqQ, headDim};
  std::vector<LongType> kvReshapeShape = {batchHeads, seqKV, headDim};

  auto qReshaped = qPerm->reshape('c', reshapeShape);
  auto kReshaped = kPerm->reshape('c', kvReshapeShape);
  auto vReshaped = vPerm->reshape('c', kvReshapeShape);

  // Create output buffer
  NDArray outReshaped('c', reshapeShape, query->dataType(), context);

  auto& entry = getFlashAttention(batch, seqQ, seqKV, numHeads, headDim, dtype, isCausal, false);
  if (!entry.valid) {
    THROW_EXCEPTION("Flash Attention graph compilation failed for 4D inputs");
  }

  auto& cache = FlashAttentionCache::instance();
  dnnl::stream strm(cache.eng);

  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  std::vector<int64_t> q_shape = {batchHeads, seqQ, headDim};
  std::vector<int64_t> k_shape = {batchHeads, seqKV, headDim};
  std::vector<int64_t> v_shape = {batchHeads, seqKV, headDim};
  std::vector<int64_t> out_shape = {batchHeads, seqQ, headDim};

  std::vector<int64_t> q_strides = {qReshaped->strideAt(0), qReshaped->strideAt(1), qReshaped->strideAt(2)};
  std::vector<int64_t> k_strides = {kReshaped->strideAt(0), kReshaped->strideAt(1), kReshaped->strideAt(2)};
  std::vector<int64_t> v_strides = {vReshaped->strideAt(0), vReshaped->strideAt(1), vReshaped->strideAt(2)};
  std::vector<int64_t> out_strides = {outReshaped.strideAt(0), outReshaped.strideAt(1), outReshaped.strideAt(2)};

  auto query_lt = dg::logical_tensor(0, dt, q_shape, q_strides);
  auto key_lt = dg::logical_tensor(1, dt, k_shape, k_strides);
  auto scale_lt = dg::logical_tensor(2, dg::logical_tensor::data_type::f32, {1}, {1});
  auto value_lt = dg::logical_tensor(3, dt, v_shape, v_strides);
  auto output_lt = dg::logical_tensor(4, dt, out_shape, out_strides);

  dg::tensor t_query(query_lt, cache.eng, qReshaped->buffer());
  dg::tensor t_key(key_lt, cache.eng, kReshaped->buffer());
  dg::tensor t_scale(scale_lt, cache.eng, &scale);
  dg::tensor t_value(value_lt, cache.eng, vReshaped->buffer());
  dg::tensor t_output(output_lt, cache.eng, outReshaped.buffer());

  entry.cp.execute(strm, {t_query, t_key, t_scale, t_value}, {t_output});
  strm.wait();

  // Reshape and permute back: [batch*heads, seq, dim] -> [batch, heads, seq, dim] -> [batch, seq, heads, dim]
  std::vector<LongType> outPermShape = {batch, numHeads, seqQ, headDim};
  auto outPerm = outReshaped.reshape('c', outPermShape);
  std::vector<LongType> permBack = {0, 2, 1, 3};
  auto outFinal = outPerm->permute(permBack, false, false);
  output->assign(outFinal);

  // Cleanup
  delete qPerm;
  delete kPerm;
  delete vPerm;
  delete qReshaped;
  delete kReshaped;
  delete vReshaped;
  delete outPerm;
  delete outFinal;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(flash_attention, ENGINE_CPU) {
  auto query = INPUT_VARIABLE(0);
  auto key = INPUT_VARIABLE(1);
  auto value = INPUT_VARIABLE(2);
  auto output = OUTPUT_VARIABLE(0);

  auto rank = query->rankOf();
  REQUIRE_TRUE(rank == 3 || rank == 4, 0,
               "flash_attention: Input rank must be 3 or 4, got %i", rank);

  // Get scale from config
  auto scale = block.numT() > 0 ? T_ARG(0) : 0.0;
  if (scale <= 0) {
    if (rank == 3) {
      scale = 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(query->sizeAt(2)));
    } else {
      scale = 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(query->sizeAt(3)));
    }
  }

  auto isCausal = block.numB() > 0 ? B_ARG(0) : false;

  if (rank == 3) {
    executeFlashAttention3D(query, key, value, output, static_cast<float>(scale), isCausal, block.launchContext());
  } else {
    executeFlashAttention4D(query, key, value, output, static_cast<float>(scale), isCausal, block.launchContext());
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(flash_attention, ENGINE_CPU) {
  auto query = INPUT_VARIABLE(0);
  auto key = INPUT_VARIABLE(1);
  auto value = INPUT_VARIABLE(2);

  const auto qType = query->dataType();
  // Runtime ISA detection: BF16 requires AVX512_CORE_BF16, HALF requires AVX512_CORE_AMX_FP16
  bool isSupportedType = (qType == DataType::FLOAT32);
  if (!isSupportedType && qType == DataType::BFLOAT16) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isSupportedType = (isa >= dnnl_cpu_isa_avx512_core_bf16);
  }
  if (!isSupportedType && qType == DataType::HALF) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isSupportedType = (isa >= dnnl_cpu_isa_avx512_core_amx_fp16);
  }
  const bool isSupportedRank = (query->rankOf() == 3 || query->rankOf() == 4);

  // For 4D, check if GQA is used (currently not supported)
  bool isGqa = false;
  if (query->rankOf() == 4) {
    isGqa = query->sizeAt(2) != key->sizeAt(2);
  }

  Requirements req("ONEDNN FLASH ATTENTION");
  req.expectFalse(makeInfoVariable(query->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(key->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(value->isEmpty(), IS_EMPTY_MSG_INPUT2), EXPECTED_FALSE) &&
      req.expectTrue(makeInfoVariable(isSupportedType, TYPE_MSG_INPUT), "Must be FLOAT32, HALF, or BFLOAT16") &&
      req.expectTrue(makeInfoVariable(isSupportedRank, RANK_MSG_INPUT0), "Must be rank 3 or 4") &&
      req.expectFalse(makeInfoVariable(isGqa, "IS_GQA"), "GQA not supported in OneDNN path");

  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(flash_attention_bp, ENGINE_CPU) {
  // OneDNN Graph API doesn't provide fused backward for SDPA yet
  // This falls through to the generic CPU implementation
  return sd::Status::KERNEL_FAILURE;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(flash_attention_bp, ENGINE_CPU) {
  // Always return false to use the generic CPU implementation for backward
  // OneDNN Graph API doesn't support fused SDPA backward pass yet
  Requirements req("ONEDNN FLASH ATTENTION BP");
  req.expectFalse(makeInfoVariable(true, "ONEDNN_SDPA_BP"), "OneDNN SDPA backward not yet supported");
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
