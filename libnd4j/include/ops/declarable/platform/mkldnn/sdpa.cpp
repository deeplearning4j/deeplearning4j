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
// OneDNN Scaled Dot Product Attention (SDPA) implementation
// Uses OneDNN Graph API for fused SDPA with compiled partition caching
//

#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include <oneapi/dnnl/dnnl_graph.hpp>

#include <unordered_map>
#include <mutex>

#include "mkldnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

// Namespace aliases to avoid collision with sd::graph
namespace dg = dnnl::graph;

//////////////////////////////////////////////////////////////////////////
// Cache for compiled SDPA partitions
struct SDPACache {
  // Key: concatenated dimensions
  struct Key {
    int64_t batch, seqQ, seqKV, dim;
    int dtype;  // 0=f32, 1=f16, 2=bf16

    bool operator==(const Key& o) const {
      return batch == o.batch && seqQ == o.seqQ && seqKV == o.seqKV &&
             dim == o.dim && dtype == o.dtype;
    }
  };

  struct KeyHash {
    size_t operator()(const Key& k) const {
      return std::hash<int64_t>()(k.batch) ^ (std::hash<int64_t>()(k.seqQ) << 1) ^
             (std::hash<int64_t>()(k.seqKV) << 2) ^ (std::hash<int64_t>()(k.dim) << 3) ^
             (std::hash<int>()(k.dtype) << 4);
    }
  };

  struct Entry {
    dg::compiled_partition cp;
    // Store logical tensor IDs for input/output ordering
    std::vector<size_t> input_ids;
    std::vector<size_t> output_ids;
    bool valid = false;
  };

  std::unordered_map<Key, Entry, KeyHash> cache;
  std::mutex mtx;
  dnnl::engine eng{dnnl::engine::kind::cpu, 0};

  static SDPACache& instance() {
    static SDPACache inst;
    return inst;
  }
};

//////////////////////////////////////////////////////////////////////////
// Build and compile SDPA graph
static SDPACache::Entry buildSDPAGraph(int64_t batch, int64_t seqQ, int64_t seqKV,
                                        int64_t dim, dg::logical_tensor::data_type dtype,
                                        dnnl::engine& eng) {
  SDPACache::Entry entry;

  size_t id = 0;

  // Shapes: [batch, seqQ/seqKV, dim] for Q,K,V - 3D tensors directly
  std::vector<int64_t> q_shape = {batch, seqQ, dim};
  std::vector<int64_t> k_shape = {batch, seqKV, dim};
  std::vector<int64_t> v_shape = {batch, seqKV, dim};
  std::vector<int64_t> score_shape = {batch, seqQ, seqKV};
  std::vector<int64_t> out_shape = {batch, seqQ, dim};

  // Create logical tensors with proper strides for row-major layout
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

  // Build ops
  // BMM1: Q @ K^T
  dg::op bmm1(id++, dg::op::kind::MatMul, "bmm1");
  bmm1.set_attr<bool>(dg::op::attr::transpose_b, true);
  bmm1.add_inputs({query_lt, key_lt});
  bmm1.add_outputs({score_lt});

  // Scale: scores * scale_factor
  dg::op scale_op(id++, dg::op::kind::Multiply, "scale");
  scale_op.add_inputs({score_lt, scale_lt});
  scale_op.add_outputs({scaled_lt});

  // Softmax
  dg::op softmax_op(id++, dg::op::kind::SoftMax, "softmax");
  softmax_op.set_attr<int64_t>(dg::op::attr::axis, -1);
  softmax_op.add_inputs({scaled_lt});
  softmax_op.add_outputs({probs_lt});

  // BMM2: probs @ V
  dg::op bmm2(id++, dg::op::kind::MatMul, "bmm2");
  bmm2.add_inputs({probs_lt, value_lt});
  bmm2.add_outputs({output_lt});

  // Build graph
  dg::graph g(dnnl::engine::kind::cpu);
  g.add_op(bmm1);
  g.add_op(scale_op);
  g.add_op(softmax_op);
  g.add_op(bmm2);
  g.finalize();

  // Get partitions
  auto partitions = g.get_partitions();
  if (partitions.empty()) {
    entry.valid = false;
    return entry;
  }

  // Compile - inputs: Q, K, scale, V; outputs: out
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
// Get or create compiled SDPA partition
static SDPACache::Entry& getSDPA(int64_t batch, int64_t seqQ, int64_t seqKV, int64_t dim, int dtype) {
  auto& cache = SDPACache::instance();
  SDPACache::Key key{batch, seqQ, seqKV, dim, dtype};

  std::lock_guard<std::mutex> lock(cache.mtx);

  auto it = cache.cache.find(key);
  if (it != cache.cache.end() && it->second.valid) {
    return it->second;
  }

  // Build new entry
  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  cache.cache[key] = buildSDPAGraph(batch, seqQ, seqKV, dim, dt, cache.eng);
  return cache.cache[key];
}

//////////////////////////////////////////////////////////////////////////
// Execute SDPA - minimal overhead path
static void executeSDPA(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
                        float scale, LaunchContext* context) {
  const auto batch = query->sizeAt(0);
  const auto seqQ = query->sizeAt(1);
  const auto seqKV = key->sizeAt(1);
  const auto dim = query->sizeAt(2);

  int dtype = 0;
  if (query->dataType() == DataType::HALF) dtype = 1;
  else if (query->dataType() == DataType::BFLOAT16) dtype = 2;

  auto& entry = getSDPA(batch, seqQ, seqKV, dim, dtype);
  if (!entry.valid) {
    THROW_EXCEPTION("SDPA graph compilation failed");
  }

  auto& cache = SDPACache::instance();
  dnnl::stream strm(cache.eng);

  // Data type for logical tensors
  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  // Create input/output logical tensors with actual strides
  std::vector<int64_t> q_shape = {batch, seqQ, dim};
  std::vector<int64_t> k_shape = {batch, seqKV, dim};
  std::vector<int64_t> v_shape = {batch, seqKV, dim};
  std::vector<int64_t> out_shape = {batch, seqQ, dim};

  // Get strides from NDArrays
  std::vector<int64_t> q_strides = {query->strideAt(0), query->strideAt(1), query->strideAt(2)};
  std::vector<int64_t> k_strides = {key->strideAt(0), key->strideAt(1), key->strideAt(2)};
  std::vector<int64_t> v_strides = {value->strideAt(0), value->strideAt(1), value->strideAt(2)};
  std::vector<int64_t> out_strides = {output->strideAt(0), output->strideAt(1), output->strideAt(2)};

  // Create logical tensors with actual data strides
  auto query_lt = dg::logical_tensor(0, dt, q_shape, q_strides);
  auto key_lt = dg::logical_tensor(1, dt, k_shape, k_strides);
  auto scale_lt = dg::logical_tensor(2, dg::logical_tensor::data_type::f32, {1}, {1});
  auto value_lt = dg::logical_tensor(3, dt, v_shape, v_strides);
  auto output_lt = dg::logical_tensor(4, dt, out_shape, out_strides);

  // Create tensors with direct buffer pointers - no copies
  dg::tensor t_query(query_lt, cache.eng, query->buffer());
  dg::tensor t_key(key_lt, cache.eng, key->buffer());
  dg::tensor t_scale(scale_lt, cache.eng, &scale);
  dg::tensor t_value(value_lt, cache.eng, value->buffer());
  dg::tensor t_output(output_lt, cache.eng, output->buffer());

  // Execute
  entry.cp.execute(strm, {t_query, t_key, t_scale, t_value}, {t_output});
  strm.wait();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(dot_product_attention_v2, ENGINE_CPU) {
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = block.width() > 2 ? INPUT_VARIABLE(2) : values;

  auto output = OUTPUT_VARIABLE(0);
  auto attentionScores = OUTPUT_VARIABLE(1);
  auto attentionLogits = OUTPUT_VARIABLE(2);

  REQUIRE_TRUE(queries->rankOf() >= 2 && queries->rankOf() <= 3, 0,
               "dot_product_attention_v2: rank must be 2 or 3, got %i", queries->rankOf());

  auto scale = block.numT() > 0 ? T_ARG(0) : 0.0;
  if (scale <= 0) {
    scale = 1.0 / std::sqrt(static_cast<double>(queries->sizeAt(-1)));
  }

  // Handle rank 2 by adding batch dimension
  NDArray *q3d = nullptr, *k3d = nullptr, *v3d = nullptr, *out3d = nullptr;
  bool needReshape = queries->rankOf() == 2;

  if (needReshape) {
    std::vector<sd::LongType> shape3d_q = {1, queries->sizeAt(0), queries->sizeAt(1)};
    std::vector<sd::LongType> shape3d_kv = {1, keys->sizeAt(0), keys->sizeAt(1)};
    std::vector<sd::LongType> shape3d_out = {1, output->sizeAt(0), output->sizeAt(1)};
    q3d = queries->reshape('c', shape3d_q);
    k3d = keys->reshape('c', shape3d_kv);
    v3d = values->reshape('c', shape3d_kv);
    out3d = output->reshape('c', shape3d_out);
  } else {
    q3d = queries;
    k3d = keys;
    v3d = values;
    out3d = output;
  }

  // Execute fused SDPA
  executeSDPA(q3d, k3d, v3d, out3d, static_cast<float>(scale), block.launchContext());

  // Attention scores/logits not available from fused kernel
  if (!attentionScores->isEmpty()) attentionScores->nullify();
  if (!attentionLogits->isEmpty()) attentionLogits->nullify();

  // Cleanup reshaped arrays
  if (needReshape) {
    delete q3d;
    delete k3d;
    delete v3d;
    delete out3d;
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(dot_product_attention_v2, ENGINE_CPU) {
  auto query = INPUT_VARIABLE(0);
  auto value = INPUT_VARIABLE(1);

  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;

  // Check masks - OneDNN Graph SDPA doesn't support custom masks
  auto qMask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  auto vMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;
  bool hasMasks = (qMask != nullptr && !qMask->isEmpty()) || (vMask != nullptr && !vMask->isEmpty());

  const auto qType = query->dataType();
  const bool isSupportedType = (qType == DataType::FLOAT32 || qType == DataType::HALF || qType == DataType::BFLOAT16);

  Requirements req("ONEDNN GRAPH SDPA");
  req.expectFalse(makeInfoVariable(query->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(value->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectTrue(makeInfoVariable(isSupportedType, TYPE_MSG_INPUT), "Must be FLOAT32, HALF, or BFLOAT16") &&
      req.expectFalse(makeInfoVariable(hasMasks, "HAS_MASKS"), "Custom masks not supported") &&
      req.expectEq(makeInfoVariable(dropout, "DROPOUT"), 0.0) &&
      req.expectGreaterEq(makeInfoVariable(query->rankOf(), RANK_MSG_INPUT0), 2) &&
      req.expectLessEq(makeInfoVariable(query->rankOf(), RANK_MSG_INPUT0), 3);

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
