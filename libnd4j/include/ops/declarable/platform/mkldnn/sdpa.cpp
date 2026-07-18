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
// Optimized OneDNN Scaled Dot Product Attention (SDPA) implementation
// Features:
// - 3D and 4D tensor support
// - Thread-local stream caching for reduced overhead
// - Compiled partition caching with shape-based lookup
// - Direct buffer access - no memory copies
//

#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <system/openmp_pragmas.h>
#include <math/templatemath.h>

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <dnnl.hpp>
#include <dnnl.h>

#ifdef HAVE_MKL
#include <mkl.h>
#endif

#include <unordered_map>
#include <mutex>
#include <thread>
#include <algorithm>
#include <vector>
#include <cmath>
#include <limits>

#include "mkldnnUtils.h"
#include <helpers/FlashAttentionHelper.h>
#include <ops/declarable/helpers/kv_scatter.h>

#include <system/BackendNamespace.h>

namespace sd {
namespace ops {
namespace platforms {
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_BEGIN

// Namespace aliases to avoid collision with sd::graph
namespace dg = dnnl::graph;

//////////////////////////////////////////////////////////////////////////
// Thread-local stream for reduced allocation overhead
static thread_local std::unique_ptr<dnnl::stream> tls_stream;

static dnnl::stream& getThreadStream(dnnl::engine& eng) {
  if (!tls_stream) {
    tls_stream = std::make_unique<dnnl::stream>(eng);
  }
  return *tls_stream;
}

//////////////////////////////////////////////////////////////////////////
// Enhanced cache for compiled SDPA partitions - supports 3D and 4D
struct SDPACache {
  struct Key {
    int64_t batch, seqQ, seqKV, numHeads, headDim;
    int dtype;   // 0=f32, 1=f16, 2=bf16
    bool is4D;   // true for 4D [batch, seq, heads, dim], false for 3D [batch, seq, dim]
    bool hasBias; // true when an additive attention bias is fused into the graph

    bool operator==(const Key& o) const {
      return batch == o.batch && seqQ == o.seqQ && seqKV == o.seqKV &&
             numHeads == o.numHeads && headDim == o.headDim &&
             dtype == o.dtype && is4D == o.is4D && hasBias == o.hasBias;
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
      h ^= std::hash<bool>()(k.is4D) << 6;
      h ^= std::hash<bool>()(k.hasBias) << 7;
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

  static SDPACache& instance() {
    static SDPACache inst;
    return inst;
  }
};

//////////////////////////////////////////////////////////////////////////
// Build and compile 3D SDPA graph: [batch, seq, dim]
static SDPACache::Entry buildSDPAGraph3D(int64_t batch, int64_t seqQ, int64_t seqKV,
                                          int64_t dim, dg::logical_tensor::data_type dtype,
                                          dnnl::engine& eng) {
  SDPACache::Entry entry;
  size_t id = 0;

  // Shapes for 3D: [batch, seqQ/seqKV, dim]
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

  // Build ops: Q @ K^T -> scale -> softmax -> @ V
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

  auto partitions = g.get_partitions();
  if (partitions.empty()) {
    entry.valid = false;
    return entry;
  }

  // Compile: inputs=[Q, K, scale, V], outputs=[out]
  std::vector<dg::logical_tensor> inputs = {query_lt, key_lt, scale_lt, value_lt};
  std::vector<dg::logical_tensor> outputs = {output_lt};

  try {
    entry.cp = partitions[0].compile(inputs, outputs, eng);
    entry.valid = true;
  } catch (...) {
    entry.valid = false;
  }

  return entry;
}

//////////////////////////////////////////////////////////////////////////
// Build and compile 4D SDPA graph: [batch, heads, seqQ, headDim]
// Uses true 4D tensors with strided layout for zero-copy execution
static SDPACache::Entry buildSDPAGraph4D(int64_t batch, int64_t seqQ, int64_t seqKV,
                                          int64_t numHeads, int64_t headDim,
                                          dg::logical_tensor::data_type dtype,
                                          dnnl::engine& eng) {
  SDPACache::Entry entry;
  size_t id = 0;

  // Use true 4D shape: [batch, heads, seq, headDim]
  // This allows strided access to [batch, seq, heads, headDim] input data
  std::vector<int64_t> q_shape = {batch, numHeads, seqQ, headDim};
  std::vector<int64_t> k_shape = {batch, numHeads, seqKV, headDim};
  std::vector<int64_t> v_shape = {batch, numHeads, seqKV, headDim};
  std::vector<int64_t> score_shape = {batch, numHeads, seqQ, seqKV};
  std::vector<int64_t> out_shape = {batch, numHeads, seqQ, headDim};

  // Create logical tensors with strided layout (strides provided at execution)
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

  // Build 4D batched attention: Q @ K^T -> scale -> softmax -> @ V
  // MatMul operates on last two dims, batch dims are [batch, heads]
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
  } catch (...) {
    entry.valid = false;
  }

  return entry;
}

//////////////////////////////////////////////////////////////////////////
// Build and compile 4D SDPA graph with fused additive bias (attention mask).
// Graph: Q @ K^T -> scale -> Add(bias) -> softmax -> @ V
// bias shape [1, 1, seqQ, seqKV] is broadcastable to [batch, numHeads, seqQ, seqKV].
// Using f32 for bias regardless of Q/K/V dtype (mask values are always float).
static SDPACache::Entry buildSDPAGraph4D_WithBias(int64_t batch, int64_t seqQ, int64_t seqKV,
                                                   int64_t numHeads, int64_t headDim,
                                                   dg::logical_tensor::data_type dtype,
                                                   dnnl::engine& eng) {
  SDPACache::Entry entry;
  size_t id = 0;

  std::vector<int64_t> q_shape     = {batch, numHeads, seqQ, headDim};
  std::vector<int64_t> k_shape     = {batch, numHeads, seqKV, headDim};
  std::vector<int64_t> v_shape     = {batch, numHeads, seqKV, headDim};
  std::vector<int64_t> score_shape = {batch, numHeads, seqQ, seqKV};
  std::vector<int64_t> bias_shape  = {1, 1, seqQ, seqKV};   // broadcast over batch/heads
  std::vector<int64_t> out_shape   = {batch, numHeads, seqQ, headDim};

  auto query_lt   = dg::logical_tensor(id++, dtype,                                q_shape,     dg::logical_tensor::layout_type::strided);
  auto key_lt     = dg::logical_tensor(id++, dtype,                                k_shape,     dg::logical_tensor::layout_type::strided);
  auto value_lt   = dg::logical_tensor(id++, dtype,                                v_shape,     dg::logical_tensor::layout_type::strided);
  auto score_lt   = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32,  score_shape, dg::logical_tensor::layout_type::strided);
  auto scale_lt   = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32,  {1},         dg::logical_tensor::layout_type::strided);
  auto scaled_lt  = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32,  score_shape, dg::logical_tensor::layout_type::strided);
  auto bias_lt    = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32,  bias_shape,  dg::logical_tensor::layout_type::strided);
  auto biased_lt  = dg::logical_tensor(id++, dg::logical_tensor::data_type::f32,  score_shape, dg::logical_tensor::layout_type::strided);
  auto probs_lt   = dg::logical_tensor(id++, dtype,                                score_shape, dg::logical_tensor::layout_type::strided);
  auto output_lt  = dg::logical_tensor(id++, dtype,                                out_shape,   dg::logical_tensor::layout_type::strided);

  dg::op bmm1(id++, dg::op::kind::MatMul, "bmm1");
  bmm1.set_attr<bool>(dg::op::attr::transpose_b, true);
  bmm1.add_inputs({query_lt, key_lt});
  bmm1.add_outputs({score_lt});

  dg::op scale_op(id++, dg::op::kind::Multiply, "scale");
  scale_op.add_inputs({score_lt, scale_lt});
  scale_op.add_outputs({scaled_lt});

  dg::op bias_add(id++, dg::op::kind::Add, "bias_add");
  bias_add.add_inputs({scaled_lt, bias_lt});
  bias_add.add_outputs({biased_lt});

  dg::op softmax_op(id++, dg::op::kind::SoftMax, "softmax");
  softmax_op.set_attr<int64_t>(dg::op::attr::axis, -1);
  softmax_op.add_inputs({biased_lt});
  softmax_op.add_outputs({probs_lt});

  dg::op bmm2(id++, dg::op::kind::MatMul, "bmm2");
  bmm2.add_inputs({probs_lt, value_lt});
  bmm2.add_outputs({output_lt});

  dg::graph g(dnnl::engine::kind::cpu);
  g.add_op(bmm1);
  g.add_op(scale_op);
  g.add_op(bias_add);
  g.add_op(softmax_op);
  g.add_op(bmm2);
  g.finalize();

  auto partitions = g.get_partitions();
  if (partitions.empty()) {
    entry.valid = false;
    return entry;
  }

  // inputs=[Q, K, scale, bias, V], outputs=[out]
  std::vector<dg::logical_tensor> inputs = {query_lt, key_lt, scale_lt, bias_lt, value_lt};
  std::vector<dg::logical_tensor> outputs = {output_lt};

  try {
    entry.cp = partitions[0].compile(inputs, outputs, eng);
    entry.valid = true;
  } catch (...) {
    entry.valid = false;
  }

  return entry;
}

//////////////////////////////////////////////////////////////////////////
// Get or create compiled SDPA partition
static SDPACache::Entry& getSDPA(int64_t batch, int64_t seqQ, int64_t seqKV,
                                  int64_t numHeads, int64_t headDim, int dtype,
                                  bool is4D, bool hasBias = false) {
  auto& cache = SDPACache::instance();
  SDPACache::Key key{batch, seqQ, seqKV, numHeads, headDim, dtype, is4D, hasBias};

  std::lock_guard<std::mutex> lock(cache.mtx);

  auto it = cache.cache.find(key);
  if (it != cache.cache.end() && it->second.valid) {
    return it->second;
  }

  // Build new entry
  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  if (is4D) {
    if (hasBias) {
      cache.cache[key] = buildSDPAGraph4D_WithBias(batch, seqQ, seqKV, numHeads, headDim, dt, cache.eng);
    } else {
      cache.cache[key] = buildSDPAGraph4D(batch, seqQ, seqKV, numHeads, headDim, dt, cache.eng);
    }
  } else {
    // For 3D, headDim is the full dimension, numHeads=1
    cache.cache[key] = buildSDPAGraph3D(batch, seqQ, seqKV, headDim, dt, cache.eng);
  }
  return cache.cache[key];
}

//////////////////////////////////////////////////////////////////////////
// Execute 3D SDPA - minimal overhead path
// Returns true on success, false if the OneDNN graph compilation failed
// (caller should fall back to FlashAttentionHelper).
static bool executeSDPA3D(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
                          float scale, LaunchContext* context) {
  const auto batch = query->sizeAt(0);
  const auto seqQ = query->sizeAt(1);
  const auto seqKV = key->sizeAt(1);
  const auto dim = query->sizeAt(2);

  int dtype = 0;
  if (query->dataType() == DataType::HALF) dtype = 1;
  else if (query->dataType() == DataType::BFLOAT16) dtype = 2;

  auto& entry = getSDPA(batch, seqQ, seqKV, 1, dim, dtype, false);
  if (!entry.valid) {
    return false;
  }

  auto& cache = SDPACache::instance();
  auto& strm = getThreadStream(cache.eng);

  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  // Create logical tensors with actual strides.
  // IDs MUST match those assigned in buildSDPAGraph3D:
  //   query=0, key=1, value=2, score=3, scale=4, scaled=5, probs=6, output=7
  // The compile() call used inputs {query(0), key(1), scale(4), value(2)} and output {output(7)}.
  // execute() tensors are matched by logical_tensor ID, not by vector position,
  // so using wrong IDs here causes the output to be unwritten (all-zeros).
  std::vector<int64_t> q_shape = {batch, seqQ, dim};
  std::vector<int64_t> k_shape = {batch, seqKV, dim};
  std::vector<int64_t> v_shape = {batch, seqKV, dim};
  std::vector<int64_t> out_shape = {batch, seqQ, dim};

  std::vector<int64_t> q_strides = {query->strideAt(0), query->strideAt(1), query->strideAt(2)};
  std::vector<int64_t> k_strides = {key->strideAt(0), key->strideAt(1), key->strideAt(2)};
  std::vector<int64_t> v_strides = {value->strideAt(0), value->strideAt(1), value->strideAt(2)};
  std::vector<int64_t> out_strides = {output->strideAt(0), output->strideAt(1), output->strideAt(2)};

  auto query_lt = dg::logical_tensor(0, dt, q_shape, q_strides);   // id=0 matches build
  auto key_lt   = dg::logical_tensor(1, dt, k_shape, k_strides);   // id=1 matches build
  auto value_lt = dg::logical_tensor(2, dt, v_shape, v_strides);   // id=2 matches build
  auto scale_lt = dg::logical_tensor(4, dg::logical_tensor::data_type::f32, {1}, {1}); // id=4 matches build
  auto output_lt = dg::logical_tensor(7, dt, out_shape, out_strides); // id=7 matches build

  // Create tensors with direct buffer pointers
  dg::tensor t_query(query_lt, cache.eng, query->buffer());
  dg::tensor t_key(key_lt, cache.eng, key->buffer());
  dg::tensor t_scale(scale_lt, cache.eng, &scale);
  dg::tensor t_value(value_lt, cache.eng, value->buffer());
  dg::tensor t_output(output_lt, cache.eng, output->buffer());

  entry.cp.execute(strm, {t_query, t_key, t_scale, t_value}, {t_output});
  strm.wait();
  return true;
}

//////////////////////////////////////////////////////////////////////////
// Helper to get OneDNN data type for reorder operations
static dnnl::memory::data_type getDnnlDataType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32: return dnnl::memory::data_type::f32;
    case DataType::BFLOAT16: return dnnl::memory::data_type::bf16;
    case DataType::HALF: return dnnl::memory::data_type::f16;
    case DataType::DOUBLE: return dnnl::memory::data_type::f64;
    default: return dnnl::memory::data_type::f32;
  }
}

//////////////////////////////////////////////////////////////////////////
// Thread-local buffer pool for 4D SDPA to avoid repeated allocations
// Uses char vectors to support all data types (float32, bf16, fp16, fp64)
struct SDPA4DBufferPool {
  std::vector<char> q_buffer;
  std::vector<char> k_buffer;
  std::vector<char> v_buffer;
  std::vector<char> out_buffer;
  size_t q_capacity = 0;
  size_t kv_capacity = 0;
  size_t out_capacity = 0;

  static SDPA4DBufferPool& instance() {
    thread_local SDPA4DBufferPool pool;
    return pool;
  }

  void ensureCapacity(size_t qBytes, size_t kvBytes, size_t outBytes) {
    // Allocate each buffer independently with 2x over-allocation
    if (q_capacity < qBytes) {
      q_capacity = qBytes * 2;
      q_buffer.resize(q_capacity);
    }
    if (kv_capacity < kvBytes) {
      kv_capacity = kvBytes * 2;
      k_buffer.resize(kv_capacity);
      v_buffer.resize(kv_capacity);
    }
    if (out_capacity < outBytes) {
      out_capacity = outBytes * 2;
      out_buffer.resize(out_capacity);
    }
  }
};

//////////////////////////////////////////////////////////////////////////
// Fast permute using OneDNN reorder: [B,S,H,D] -> [B,H,S,D] with buffer
// Supports both contiguous and strided source arrays
static void fastPermute_BSHD_to_BHSD(NDArray* src, void* dstBuffer, dnnl::engine& eng,
                                      dnnl::stream& strm) {
  const auto B = src->sizeAt(0);
  const auto S = src->sizeAt(1);
  const auto H = src->sizeAt(2);
  const auto D = src->sizeAt(3);

  auto dnnlType = getDnnlDataType(src->dataType());

  // Destination shape [B,H,S,D] with contiguous strides
  dnnl::memory::dims dstDims = {B, H, S, D};
  dnnl::memory::dims dstStrides = {H*S*D, S*D, D, 1};

  // Get actual source strides from the NDArray (handles non-contiguous data)
  // Permutation [0,2,1,3] maps [B,S,H,D] -> [B,H,S,D]
  // So for destination dim order [B,H,S,D], we need strides from source dims [0,2,1,3]
  dnnl::memory::dims srcViewStrides = {
    src->strideAt(0),  // B stride
    src->strideAt(2),  // H stride (was dim 2 in source)
    src->strideAt(1),  // S stride (was dim 1 in source)
    src->strideAt(3)   // D stride
  };

  dnnl::memory::desc src_md(dstDims, dnnlType, srcViewStrides);
  dnnl::memory::desc dst_md(dstDims, dnnlType, dstStrides);

  dnnl::memory src_mem(src_md, eng, src->buffer());
  dnnl::memory dst_mem(dst_md, eng, dstBuffer);

  dnnl::reorder reorder_prim(src_mem, dst_mem);
  reorder_prim.execute(strm, src_mem, dst_mem);
}

//////////////////////////////////////////////////////////////////////////
// Fast permute using OneDNN reorder: [B,H,S,D] -> [B,S,H,D] with buffer
// Source is always contiguous (from buffer pool), destination may be strided
static void fastPermute_BHSD_to_BSHD(void* srcBuffer, NDArray* dst, int64_t B, int64_t H,
                                      int64_t S, int64_t D, dnnl::engine& eng,
                                      dnnl::stream& strm) {
  auto dnnlType = getDnnlDataType(dst->dataType());

  // Destination shape [B,S,H,D]
  dnnl::memory::dims dstDims = {B, S, H, D};

  // Source is in [B,H,S,D] contiguous layout, we create a [B,S,H,D] view of it
  // Permutation [0,2,1,3] maps [B,H,S,D] -> [B,S,H,D]
  // Source strides for [B,H,S,D] contiguous: {H*S*D, S*D, D, 1}
  // View strides for [B,S,H,D]: permute as [0,2,1,3] -> {H*S*D, D, S*D, 1}
  dnnl::memory::dims srcViewStrides = {H*S*D, D, S*D, 1};

  // Get actual destination strides from NDArray (handles non-contiguous output)
  dnnl::memory::dims dstStrides = {
    dst->strideAt(0),
    dst->strideAt(1),
    dst->strideAt(2),
    dst->strideAt(3)
  };

  dnnl::memory::desc src_md(dstDims, dnnlType, srcViewStrides);
  dnnl::memory::desc dst_md(dstDims, dnnlType, dstStrides);

  dnnl::memory src_mem(src_md, eng, srcBuffer);
  dnnl::memory dst_mem(dst_md, eng, dst->buffer());

  dnnl::reorder reorder_prim(src_mem, dst_mem);
  reorder_prim.execute(strm, src_mem, dst_mem);
}

#ifdef HAVE_MKL
//////////////////////////////////////////////////////////////////////////
// Thread-local score buffer for MKL SDPA
struct MKLSDPABuffer {
  std::vector<float> scores;
  size_t capacity = 0;

  static MKLSDPABuffer& instance() {
    thread_local MKLSDPABuffer buf;
    return buf;
  }

  float* ensureCapacity(size_t needed) {
    if (capacity < needed) {
      capacity = needed * 2;
      scores.resize(capacity);
    }
    return scores.data();
  }
};

//////////////////////////////////////////////////////////////////////////
// Fast vectorized softmax using MKL VML
static void mklSoftmaxInPlace(float* data, MKL_INT rows, MKL_INT cols, float scale) {
  for (MKL_INT r = 0; r < rows; r++) {
    float* row = data + r * cols;

    // Scale and find max in single pass
    float maxVal = -std::numeric_limits<float>::infinity();
    PRAGMA_OMP_SIMD_ARGS(reduction(max:maxVal))
    for (MKL_INT c = 0; c < cols; c++) {
      row[c] *= scale;
      if (row[c] > maxVal) maxVal = row[c];
    }

    // Subtract max
    PRAGMA_OMP_SIMD
    for (MKL_INT c = 0; c < cols; c++) row[c] -= maxVal;

    // Vectorized exp
    vsExp(cols, row, row);

    // Sum and normalize with SIMD instead of per-row cblas_sasum/cblas_sscal. Those were
    // two tiny level-1 BLAS calls PER ROW (thousands per attention op) whose call +
    // OpenBLAS threading overhead dominated the length-`cols` work. After vsExp all values
    // are >= 0, so a plain sum equals cblas_sasum.
    float sum = 0.0f;
    PRAGMA_OMP_SIMD_ARGS(reduction(+:sum))
    for (MKL_INT c = 0; c < cols; c++) sum += row[c];
    if (sum > 0.0f) {
      const float inv = 1.0f / sum;
      PRAGMA_OMP_SIMD
      for (MKL_INT c = 0; c < cols; c++) row[c] *= inv;
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// Execute 4D SDPA using MKL strided batch GEMM (prefill) or GEMV (decode).
//
// Input layout: [batch, seq, heads, headDim] (BSHD).
// For seqQ=1 (decode), uses cblas_sgemv per head — avoids GEMM dispatch overhead
// and uses the dedicated dot-product path in OpenBLAS/MKL for M=1.
// For seqQ>1 (prefill), uses cblas_sgemm_batch_strided over all heads at once.
//
// GQA: handled via stride_b=0 trick — no K/V tiling needed.
static void executeSDPA4D_MKL(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
                               float scale, LaunchContext* context,
                               NDArray* attentionBias = nullptr, bool isCausal = false) {
  const MKL_INT batch = query->sizeAt(0);
  const MKL_INT seqQ = query->sizeAt(1);
  const MKL_INT seqKV = key->sizeAt(1);
  const MKL_INT numHeads = query->sizeAt(2);
  const MKL_INT numKVHeads = key->sizeAt(2);
  const MKL_INT headDim = query->sizeAt(3);

  float* qPtr = query->bufferAsT<float>();
  float* kPtr = key->bufferAsT<float>();
  float* vPtr = value->bufferAsT<float>();
  float* outPtr = output->bufferAsT<float>();

  // Input: [B, S, H, D] — strides index as [0,1,2,3]
  const MKL_INT qBatchStride = query->strideAt(0);  // S*H*D
  const MKL_INT qSeqStride   = query->strideAt(1);  // H*D  (= leading dim for GEMM rows)
  const MKL_INT qHeadStride  = query->strideAt(2);  // D    (batch stride between Q heads)

  const MKL_INT kBatchStride = key->strideAt(0);
  const MKL_INT kSeqStride   = key->strideAt(1);    // leading dim of K rows
  const MKL_INT kHeadStride  = key->strideAt(2);    // batch stride between KV heads

  const MKL_INT vBatchStride = value->strideAt(0);
  const MKL_INT vSeqStride   = value->strideAt(1);
  const MKL_INT vHeadStride  = value->strideAt(2);

  const MKL_INT outBatchStride = output->strideAt(0);
  const MKL_INT outSeqStride   = output->strideAt(1);
  const MKL_INT outHeadStride  = output->strideAt(2);

  const MKL_INT scoreSize = seqQ * seqKV;
  const bool isGqa = (numKVHeads != numHeads);
  const MKL_INT headsPerGroup = isGqa ? (numHeads / numKVHeads) : 1;

  // Attention bias: extract contiguous FP32 pointer if present.
  // Bias shape is [1,1,1,seqKV] or [1,1,seqQ,seqKV] — one vector of length seqKV,
  // broadcast across batch and heads. We only need the raw float* row.
  float* biasPtr = nullptr;
  NDArray* biasF32 = nullptr;
  bool ownsBiasLocal = false;
  if (attentionBias != nullptr && !attentionBias->isEmpty()) {
    if (attentionBias->dataType() != DataType::FLOAT32) {
      biasF32 = attentionBias->cast(DataType::FLOAT32);
      ownsBiasLocal = true;
    } else {
      biasF32 = attentionBias;
    }
    biasPtr = biasF32->bufferAsT<float>();
  }

  auto& scratch = MKLSDPABuffer::instance();
  float* allScores = scratch.ensureCapacity(numHeads * seqQ * seqKV);

  // -------------------------------------------------------------------------
  // Decode path: seqQ == 1
  // Use cblas_sgemv per head instead of cblas_sgemm_batch_strided.
  // For M=1, sgemv is faster: no GEMM packing overhead, direct dot-product path.
  // K layout for sgemv: K is [seqKV, headDim] with leading dim kSeqStride.
  //   sgemv(CblasNoTrans) computes y = K * x where x=[headDim] (the query vector),
  //   giving y[j] = sum_d K[j,d]*q[d] for j in [0,seqKV].
  //   This equals q @ K^T element-wise, which is what we want for scores.
  // V layout for sgemv: V is [seqKV, headDim] with leading dim vSeqStride.
  //   sgemv(CblasTrans) computes out = V^T * attn where attn=[seqKV],
  //   giving out[d] = sum_j V[j,d]*attn[j] — correct attention-weighted sum.
  // -------------------------------------------------------------------------
  if (seqQ == 1) {
    for (MKL_INT b = 0; b < batch; b++) {
      float* Q = qPtr  + b * qBatchStride;
      float* K = kPtr  + b * kBatchStride;
      float* V = vPtr  + b * vBatchStride;
      float* O = outPtr + b * outBatchStride;

      for (MKL_INT h = 0; h < numHeads; h++) {
        // GQA: map query head h to KV head g = h / headsPerGroup
        const MKL_INT g = isGqa ? (h / headsPerGroup) : h;

        float* qHead = Q + h * qHeadStride;  // [headDim] — the query vector for this head
        float* kHead = K + g * kHeadStride;  // [seqKV, headDim] — keys for this KV head
        float* vHead = V + g * vHeadStride;  // [seqKV, headDim] — values for this KV head
        float* sHead = allScores + h * seqKV; // [seqKV] — score scratch for this head
        float* oHead = O + h * outHeadStride; // [headDim] — output for this head

        // scores[j] = scale * sum_d K[j,d] * q[d]  for j in [0, seqKV)
        // K is row-major [seqKV, headDim]: CblasNoTrans with lda=kSeqStride
        cblas_sgemv(CblasRowMajor, CblasNoTrans,
                    seqKV, headDim,
                    scale,
                    kHead, kSeqStride,
                    qHead, 1,
                    0.0f, sHead, 1);

        // Add attention bias before softmax: scores[j] += bias[j]
        // Bias is [1,1,1,seqKV] — same values for every batch/head (broadcast).
        if (biasPtr != nullptr) {
          cblas_saxpy(seqKV, 1.0f, biasPtr, 1, sHead, 1);
        }

        // softmax(sHead) in-place — mklSoftmaxInPlace handles scale internally
        // but we already applied scale in sgemv above, so pass scale=1.0f here.
        mklSoftmaxInPlace(sHead, 1, seqKV, 1.0f);

        // out[d] = sum_j attn[j] * V[j,d]  for d in [0, headDim)
        // V is row-major [seqKV, headDim]: CblasTrans gives V^T * attn
        cblas_sgemv(CblasRowMajor, CblasTrans,
                    seqKV, headDim,
                    1.0f,
                    vHead, vSeqStride,
                    sHead, 1,
                    0.0f, oHead, 1);
      }
    }
    if (ownsBiasLocal) delete biasF32;
    return;
  }

  // -------------------------------------------------------------------------
  // Prefill path: seqQ > 1
  // Per-head cblas_sgemm: the [B, S, H, D] layout interleaves heads within
  // each sequence position, so stride between heads (D) is smaller than the
  // per-head matrix size (seqQ * H*D). cblas_sgemm_batch_strided requires
  // stride >= M * lda which can't hold here. Loop over heads explicitly.
  // -------------------------------------------------------------------------
  for (MKL_INT b = 0; b < batch; b++) {
    float* Q = qPtr + b * qBatchStride;
    float* K = kPtr + b * kBatchStride;
    float* V = vPtr + b * vBatchStride;
    float* O = outPtr + b * outBatchStride;

    // Step 1: Q @ K^T -> scores  per head. Parallelize ACROSS heads; MKL_DYNAMIC (default)
    // keeps each nested per-head cblas_sgemm single-threaded inside this OMP region, so we get
    // head-level parallelism without BLAS-thread oversubscription on the small
    // [seqQ,headDim]@[headDim,seqKV] GEMMs. Per-head writes to allScores[h] are disjoint.
    PRAGMA_OMP_PARALLEL_FOR
    for (MKL_INT h = 0; h < numHeads; h++) {
      const MKL_INT g = isGqa ? (h / headsPerGroup) : h;
      float* qHead = Q + h * qHeadStride;    // [seqQ, headDim] with lda=qSeqStride
      float* kHead = K + g * kHeadStride;    // [seqKV, headDim] with lda=kSeqStride
      float* sHead = allScores + h * scoreSize; // [seqQ, seqKV] contiguous

      // scores = Q @ K^T: [seqQ, headDim] x [headDim, seqKV] -> [seqQ, seqKV]
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans,
                  seqQ, seqKV, headDim, 1.0f,
                  qHead, qSeqStride,
                  kHead, kSeqStride,
                  0.0f, sHead, seqKV);
    }

    // Step 1.5: Apply causal bias between Q@K^T and softmax
    // Bias shape is [1,1,seqQ,seqKV] OR [1,1,1,seqKV] (broadcast single row).
    // Determine actual bias rows from the shape to avoid buffer overread.
    if (biasPtr != nullptr) {
      const sd::LongType biasSeqQ = biasF32->sizeAt(-2);  // penultimate dim: 1 or seqQ
      for (MKL_INT h = 0; h < numHeads; h++) {
        float* sHead = allScores + h * scoreSize;  // [seqQ, seqKV] contiguous
        for (MKL_INT q = 0; q < seqQ; q++) {
          // If bias has 1 row, broadcast it to every query position; else use row q.
          // SIMD add instead of cblas_saxpy: this was seqQ*numHeads tiny level-1 BLAS
          // calls per attention op (e.g. 6144 for [1,12,512,512]) whose call + OpenBLAS
          // threading overhead dwarfed the length-seqKV work.
          const float* biasRow = biasPtr + (biasSeqQ == 1 ? 0 : q) * seqKV;
          float* sRow = sHead + q * seqKV;
          PRAGMA_OMP_SIMD
          for (MKL_INT j = 0; j < seqKV; j++) sRow[j] += biasRow[j];
        }
      }
    }


    // Step 1.75: Apply causal mask — set scores[i][j] = -1e9 where j > i + causalOffset
    if (isCausal) {
      const MKL_INT causalOffset = (seqKV > seqQ) ? (seqKV - seqQ) : 0;
      for (MKL_INT h = 0; h < numHeads; h++) {
        float* sHead = allScores + h * scoreSize;
        for (MKL_INT i = 0; i < seqQ; i++) {
          for (MKL_INT j = i + causalOffset + 1; j < seqKV; j++) {
            sHead[i * seqKV + j] = -1.0e9f;
          }
        }
      }
    }

    // Step 2: softmax with scale — independent per head, parallelize across heads
    PRAGMA_OMP_PARALLEL_FOR
    for (MKL_INT h = 0; h < numHeads; h++) {
      mklSoftmaxInPlace(allScores + h * scoreSize, seqQ, seqKV, scale);
    }

    // Step 3: scores @ V -> output  per head (parallel across heads; see Step 1 note)
    PRAGMA_OMP_PARALLEL_FOR
    for (MKL_INT h = 0; h < numHeads; h++) {
      const MKL_INT g = isGqa ? (h / headsPerGroup) : h;
      float* sHead = allScores + h * scoreSize; // [seqQ, seqKV] contiguous
      float* vHead = V + g * vHeadStride;    // [seqKV, headDim] with lda=vSeqStride
      float* oHead = O + h * outHeadStride;  // [seqQ, headDim] with lda=outSeqStride

      // output = scores @ V: [seqQ, seqKV] x [seqKV, headDim] -> [seqQ, headDim]
      cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                  seqQ, headDim, seqKV, 1.0f,
                  sHead, seqKV,
                  vHead, vSeqStride,
                  0.0f, oHead, outSeqStride);
    }
  }
  if (ownsBiasLocal) delete biasF32;
}
#endif

//////////////////////////////////////////////////////////////////////////
// Execute 4D SDPA with fused additive attention bias.
// Used for the padded-KV decode path: seqQ=1, seqKV=maxKvLen (constant).
// The bias [1, 1, 1, seqKV] masks out unfilled KV cache slots so that the
// OneDNN JIT-compiled partition is reused for every decode step without
// recompilation (seqKV is fixed at maxKvLen throughout the generation loop).
//
// input layout:  query  [batch, 1, numHeads, headDim]   (BSHD)
//                key    [batch, maxKvLen, numHeads, headDim]
//                value  [batch, maxKvLen, numHeads, headDim]
//                bias   [1, 1, 1, maxKvLen] — additive mask, f32
// output layout: output [batch, 1, numHeads, headDim]
static void executeSDPA4D_WithBias(NDArray* query, NDArray* key, NDArray* value,
                                   NDArray* output, NDArray* bias,
                                   float scale, LaunchContext* context) {
  const auto batch    = query->sizeAt(0);
  const auto seqQ     = query->sizeAt(1);
  const auto seqKV    = key->sizeAt(1);
  const auto numHeads = query->sizeAt(2);
  const auto headDim  = query->sizeAt(3);

  // Bias must be f32 — cast if the model passed it in a different type.
  NDArray* biasF32 = nullptr;
  bool ownsBias = false;
  if (bias->dataType() != DataType::FLOAT32) {
    biasF32 = bias->cast(DataType::FLOAT32);
    ownsBias = true;
  } else {
    biasF32 = bias;
  }

  int dtype = 0;
  if (query->dataType() == DataType::HALF) dtype = 1;
  else if (query->dataType() == DataType::BFLOAT16) dtype = 2;

  auto& entry = getSDPA(batch, seqQ, seqKV, numHeads, headDim, dtype, /*is4D=*/true, /*hasBias=*/true);
  if (!entry.valid) {
    // OneDNN Graph failed to compile this configuration — fall back to FlashAttentionHelper.
    // This should not normally happen on a supported ISA; log and fall back gracefully.
    // Pass the original (possibly non-f32) bias — FlashAttentionHelper casts internally.
    if (ownsBias) { delete biasF32; biasF32 = nullptr; ownsBias = false; }
    FlashAttentionHelper::Config cfg;
    cfg.scale      = scale;
    cfg.isCausal   = false;
    cfg.dropout    = 0.0f;
    cfg.numHeads   = static_cast<int>(numHeads);
    cfg.numKvHeads = static_cast<int>(key->sizeAt(2));
    FlashAttentionHelper::forward(query, key, value, output, cfg,
                                  nullptr, nullptr, nullptr, context, bias);
    return;
  }

  auto& cache = SDPACache::instance();
  auto& strm  = getThreadStream(cache.eng);

  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  // True 4D shapes: [batch, heads, seq, headDim] — same transposed-stride trick as non-bias path
  std::vector<int64_t> q_shape   = {batch, numHeads, seqQ,  headDim};
  std::vector<int64_t> k_shape   = {batch, numHeads, seqKV, headDim};
  std::vector<int64_t> v_shape   = {batch, numHeads, seqKV, headDim};
  std::vector<int64_t> bias_shape = {1, 1, seqQ, seqKV};
  std::vector<int64_t> out_shape  = {batch, numHeads, seqQ, headDim};

  // Strides for [B,H,S,D] view of [B,S,H,D] data (permutation [0,2,1,3])
  std::vector<int64_t> q_strides = {
    query->strideAt(0), query->strideAt(2), query->strideAt(1), query->strideAt(3)
  };
  std::vector<int64_t> k_strides = {
    key->strideAt(0), key->strideAt(2), key->strideAt(1), key->strideAt(3)
  };
  std::vector<int64_t> v_strides = {
    value->strideAt(0), value->strideAt(2), value->strideAt(1), value->strideAt(3)
  };
  std::vector<int64_t> out_strides = {
    output->strideAt(0), output->strideAt(2), output->strideAt(1), output->strideAt(3)
  };
  // bias strides from the actual array (handles contiguous and view cases)
  std::vector<int64_t> bias_strides = {
    biasF32->strideAt(0), biasF32->strideAt(1), biasF32->strideAt(2), biasF32->strideAt(3)
  };

  // IDs MUST match buildSDPAGraph4D_WithBias: query=0, key=1, value=2, scale=4, bias=6, output=9
  // (score=3, scaled=5, biased=7, probs=8 are internal intermediates, not provided at execute time)
  auto query_lt  = dg::logical_tensor(0, dt,                               q_shape,    q_strides);
  auto key_lt    = dg::logical_tensor(1, dt,                               k_shape,    k_strides);
  auto value_lt  = dg::logical_tensor(2, dt,                               v_shape,    v_strides);    // id=2 matches build
  auto scale_lt  = dg::logical_tensor(4, dg::logical_tensor::data_type::f32, {1},     {1});            // id=4 matches build
  auto bias_lt   = dg::logical_tensor(6, dg::logical_tensor::data_type::f32, bias_shape, bias_strides); // id=6 matches build
  auto output_lt = dg::logical_tensor(9, dt,                               out_shape,  out_strides);  // id=9 matches build

  dg::tensor t_query (query_lt,  cache.eng, query->buffer());
  dg::tensor t_key   (key_lt,    cache.eng, key->buffer());
  dg::tensor t_scale (scale_lt,  cache.eng, &scale);
  dg::tensor t_bias  (bias_lt,   cache.eng, biasF32->buffer());
  dg::tensor t_value (value_lt,  cache.eng, value->buffer());
  dg::tensor t_output(output_lt, cache.eng, output->buffer());

  entry.cp.execute(strm, {t_query, t_key, t_scale, t_bias, t_value}, {t_output});
  strm.wait();

  if (ownsBias) delete biasF32;
}

//////////////////////////////////////////////////////////////////////////
// Execute 4D SDPA with [batch, seq, heads, dim] layout
static void executeSDPA4D(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
                          float scale, LaunchContext* context,
                          NDArray* attentionBias = nullptr, bool isCausal = false) {
  // Decode path (seqQ=1): route directly to MKL strided-batch GEMV path for FP32,
  // or promote to FP32 and use MKL for FP16/BF16.
  // Do NOT delegate to FlashAttentionHelper here — it copies and tiles K/V for GQA
  // (O(headsPerKvHead * seqKV * headDim) wasted copies per layer per token) and
  // does not use cblas_sgemv for the M=1 GEMV case.
  // The MKL path (executeSDPA4D_MKL) uses cblas_sgemm_batch_strided with stride_b=0
  // for GQA — no K/V tiling at all — and handles all heads in a single BLAS call.
  // The OneDNN graph JIT recompile concern does NOT apply to executeSDPA4D_MKL since
  // it uses cblas_sgemm_batch_strided directly, not OneDNN graph partitions.
  if (query->sizeAt(1) == 1) {
#ifdef HAVE_MKL
    if (query->dataType() == DataType::FLOAT32) {
      executeSDPA4D_MKL(query, key, value, output, scale, context, attentionBias, isCausal);
      return;
    }
#endif
    // FP16/BF16 decode: promote to FP32 and use MKL path.
    // FlashAttentionHelper has unacceptable overhead for decode (see comment above).
    if (query->dataType() == DataType::HALF || query->dataType() == DataType::BFLOAT16) {
      auto q32 = query->cast(DataType::FLOAT32);
      auto k32 = key->cast(DataType::FLOAT32);
      auto v32 = value->cast(DataType::FLOAT32);
      auto outShapePtr = output->getShapeAsVector();
      NDArray out32(query->ordering(), *outShapePtr, DataType::FLOAT32, context);
      delete outShapePtr;
      executeSDPA4D(q32, k32, v32, &out32, scale, context, attentionBias, isCausal);
      output->assign(&out32);
      delete q32;
      delete k32;
      delete v32;
      return;
    }
    // Fallback for non-MKL CPU or unsupported types: use FlashAttentionHelper.
    FlashAttentionHelper::Config cfg;
    cfg.scale    = scale;
    cfg.isCausal = isCausal;
    cfg.dropout  = 0.0f;
    cfg.numHeads   = static_cast<int>(query->sizeAt(2));
    cfg.numKvHeads = static_cast<int>(key->sizeAt(2));
    FlashAttentionHelper::forward(query, key, value, output, cfg,
                                  nullptr, nullptr, nullptr, context, attentionBias);
    return;
  }

  // For FP16/BF16 on CPUs without native ISA support, promote to FP32 so we can
  // use the MKL or OneDNN FP32 path instead of failing.  The decode path (above)
  // already handles all types via FlashAttentionHelper/MmulHelper.
  bool needsPromotion = false;
#if HAVE_ONEDNN
  if (query->dataType() == DataType::HALF || query->dataType() == DataType::BFLOAT16) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    bool hasNativeHalf = (isa >= dnnl_cpu_isa_avx512_core_amx_fp16);
    bool hasNativeBf16 = (isa >= dnnl_cpu_isa_avx512_core_bf16);
    if ((query->dataType() == DataType::HALF && !hasNativeHalf) ||
        (query->dataType() == DataType::BFLOAT16 && !hasNativeBf16)) {
      needsPromotion = true;
    }
  }
#else
  if (query->dataType() != DataType::FLOAT32) {
    needsPromotion = true;
  }
#endif

  if (needsPromotion) {
    auto q32 = query->cast(DataType::FLOAT32);
    auto k32 = key->cast(DataType::FLOAT32);
    auto v32 = value->cast(DataType::FLOAT32);
    auto outShapePtr = output->getShapeAsVector();
    NDArray out32(q32->ordering(), *outShapePtr, DataType::FLOAT32, context);
    delete outShapePtr;
    executeSDPA4D(q32, k32, v32, &out32, scale, context, attentionBias, isCausal);
    output->assign(&out32);
    delete q32;
    delete k32;
    delete v32;
    return;
  }

#ifdef HAVE_MKL
  if (query->dataType() == DataType::FLOAT32) {
    executeSDPA4D_MKL(query, key, value, output, scale, context, attentionBias, isCausal);
    return;
  }
#endif
  const auto batch = query->sizeAt(0);
  const auto seqQ = query->sizeAt(1);
  const auto seqKV = key->sizeAt(1);
  const auto numHeads = query->sizeAt(2);
  const auto headDim = query->sizeAt(3);

  int dtype = 0;
  if (query->dataType() == DataType::HALF) dtype = 1;
  else if (query->dataType() == DataType::BFLOAT16) dtype = 2;

  auto& entry = getSDPA(batch, seqQ, seqKV, numHeads, headDim, dtype, true);
  if (!entry.valid) {
    THROW_EXCEPTION("SDPA 4D graph compilation failed");
  }

  auto& cache = SDPACache::instance();
  auto& strm = getThreadStream(cache.eng);

  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  // True 4D shapes: [batch, heads, seq, headDim]
  std::vector<int64_t> q_shape = {batch, numHeads, seqQ, headDim};
  std::vector<int64_t> k_shape = {batch, numHeads, seqKV, headDim};
  std::vector<int64_t> v_shape = {batch, numHeads, seqKV, headDim};
  std::vector<int64_t> out_shape = {batch, numHeads, seqQ, headDim};

  // Strides for [B,H,S,D] view of [B,S,H,D] data (permutation [0,2,1,3])
  // Input [B,S,H,D] has physical strides [S*H*D, H*D, D, 1]
  // View as [B,H,S,D] uses strides: [S*H*D, D, H*D, 1]
  std::vector<int64_t> q_strides = {
    query->strideAt(0),   // B stride from original
    query->strideAt(2),   // H stride (maps to dim 1 in view)
    query->strideAt(1),   // S stride (maps to dim 2 in view)
    query->strideAt(3)    // D stride
  };
  std::vector<int64_t> k_strides = {
    key->strideAt(0), key->strideAt(2), key->strideAt(1), key->strideAt(3)
  };
  std::vector<int64_t> v_strides = {
    value->strideAt(0), value->strideAt(2), value->strideAt(1), value->strideAt(3)
  };
  std::vector<int64_t> out_strides = {
    output->strideAt(0), output->strideAt(2), output->strideAt(1), output->strideAt(3)
  };

  // Create logical tensors with strided layouts.
  // IDs MUST match buildSDPAGraph4D: query=0, key=1, value=2, scale=4, output=7
  // (score=3, scaled=5, probs=6 are internal intermediates, not provided at execute time)
  auto query_lt  = dg::logical_tensor(0, dt,                                 q_shape,   q_strides);
  auto key_lt    = dg::logical_tensor(1, dt,                                 k_shape,   k_strides);
  auto value_lt  = dg::logical_tensor(2, dt,                                 v_shape,   v_strides);   // id=2 matches build
  auto scale_lt  = dg::logical_tensor(4, dg::logical_tensor::data_type::f32, {1},       {1});         // id=4 matches build
  auto output_lt = dg::logical_tensor(7, dt,                                 out_shape, out_strides); // id=7 matches build

  // Create tensors pointing directly to input/output buffers - ZERO COPY
  dg::tensor t_query(query_lt, cache.eng, query->buffer());
  dg::tensor t_key(key_lt, cache.eng, key->buffer());
  dg::tensor t_scale(scale_lt, cache.eng, &scale);
  dg::tensor t_value(value_lt, cache.eng, value->buffer());
  dg::tensor t_output(output_lt, cache.eng, output->buffer());

  entry.cp.execute(strm, {t_query, t_key, t_scale, t_value}, {t_output});
  strm.wait();
}

// Close the file-level backend-namespace wrap here: the op macros below open
// the backend inline namespace themselves (platform_boilerplate.h); nesting
// SD_NS inside SD_NS makes every unqualified SD_NS reference ambiguous. The
// utilities above stay backend-qualified and remain visible to the op bodies
// through inline-namespace lookup.
SD_BACKEND_PLATFORMS_INLINE_NAMESPACE_END

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(dot_product_attention_v2, ENGINE_ONEDNN) {
  auto queries = INPUT_VARIABLE(0);
  auto values = INPUT_VARIABLE(1);
  auto keys = block.width() > 2 ? INPUT_VARIABLE(2) : values;

  auto output = OUTPUT_VARIABLE(0);
  auto attentionScores = OUTPUT_VARIABLE(1);
  auto attentionLogits = OUTPUT_VARIABLE(2);

  auto rank = queries->rankOf();
  bool isRank4 = (rank == 4);
  REQUIRE_TRUE(rank >= 2 && rank <= 4, 0,
               "dot_product_attention_v2: rank must be 2, 3, or 4, got %i", rank);

  auto scale = block.numT() > 0 ? T_ARG(0) : 0.0;
  if (scale <= 0) {
    scale = 1.0 / sd::math::sd_sqrt<double, double>(static_cast<double>(queries->sizeAt(-1)));
  }

  // KV cache scatter: write current K/V into cache, then use full cache as K/V
  auto kvCacheK = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
  auto kvCacheV = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;
  auto cachePosInput = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;
  bool hasKvCache = (kvCacheK != nullptr && !kvCacheK->isEmpty() && kvCacheK->rankOf() >= 2) &&
                    (kvCacheV != nullptr && !kvCacheV->isEmpty() && kvCacheV->rankOf() >= 2);
  bool useInPlaceKv = hasKvCache && (cachePosInput != nullptr);

  if (useInPlaceKv) {
    // Use kvInPlaceWriteBSHD which handles multi-position writes (prefill seqLen > 1)
    // and matches the generic op's behavior.
    const void* cachePosPtr = cachePosInput->buffer();
    helpers::kvInPlaceWriteBSHD(kvCacheK, keys, cachePosPtr, block.launchContext());
    helpers::kvInPlaceWriteBSHD(kvCacheV, values, cachePosPtr, block.launchContext());
    // Use full cache as K/V for attention
    keys = kvCacheK;
    values = kvCacheV;
  }

  // Detect attention bias.
  // Input layout with KV cache: [Q, V, K, qMask, vMask, keyCache, valueCache, cachePos, bias]
  // Input layout without KV cache: [Q, V, K, qMask, vMask, bias]
  NDArray* attentionBias = nullptr;
  NDArray* slicedBiasOwner = nullptr;

  if (useInPlaceKv) {
    // KV cache active: bias is at input[8] (inputs 5/6 are keyCache/valueCache)
    auto kvCacheBias = block.width() > 8 ? INPUT_VARIABLE(8) : nullptr;
    if (kvCacheBias != nullptr && kvCacheBias->isEmpty()) kvCacheBias = nullptr;
    if (kvCacheBias != nullptr && kvCacheBias->rankOf() >= 2) {
      // Bias may be wider than K's seq dim (sized for maxKvLen).
      // Slice to match K's actual sequence length.
      auto biasLastDim = kvCacheBias->sizeAt(kvCacheBias->rankOf() - 1);
      auto kSeqDim = isRank4 ? keys->sizeAt(1) : keys->sizeAt(0);
      if (biasLastDim == kSeqDim) {
        attentionBias = kvCacheBias;
      } else if (biasLastDim > kSeqDim) {
        std::vector<LongType> sliceIdx;
        for (int d = 0; d < kvCacheBias->rankOf() - 1; d++) {
          sliceIdx.push_back(0);
          sliceIdx.push_back(kvCacheBias->sizeAt(d));
        }
        sliceIdx.push_back(0);
        sliceIdx.push_back(kSeqDim);
        auto* slicedView = (*kvCacheBias)(sliceIdx);
        slicedBiasOwner = new NDArray(slicedView->dup());
        attentionBias = slicedBiasOwner;
        delete slicedView;
      }
    }
  } else if (!hasKvCache && block.width() > 8) {
    // Prefill: KV cache placeholders are empty, but bias at input[8] has the causal mask
    auto prefillBias = INPUT_VARIABLE(8);
    if (prefillBias != nullptr && !prefillBias->isEmpty() && prefillBias->rankOf() >= 2) {
      auto biasLastDim = prefillBias->sizeAt(prefillBias->rankOf() - 1);
      auto kSeqDim = isRank4 ? keys->sizeAt(1) : keys->sizeAt(0);
      if (biasLastDim == kSeqDim) {
        attentionBias = prefillBias;
      } else if (biasLastDim > kSeqDim) {
        std::vector<LongType> sliceIdx;
        for (int d = 0; d < prefillBias->rankOf() - 1; d++) {
          sliceIdx.push_back(0);
          sliceIdx.push_back(prefillBias->sizeAt(d));
        }
        sliceIdx.push_back(0);
        sliceIdx.push_back(kSeqDim);
        auto* slicedView = (*prefillBias)(sliceIdx);
        slicedBiasOwner = new NDArray(slicedView->dup());
        attentionBias = slicedBiasOwner;
        delete slicedView;
      }
    }
  } else {
    // No KV cache: check input[5] as bias (legacy path)
    auto extraInput = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
    auto extraInput2 = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;
    if (extraInput != nullptr && !extraInput->isEmpty() &&
        (extraInput2 == nullptr || extraInput2->isEmpty())) {
      auto tq = (rank == 2) ? queries->sizeAt(0) : queries->sizeAt(1);
      auto tv = (rank == 2) ? values->sizeAt(0) : values->sizeAt(1);
      bool looksLikeBias = false;
      if (extraInput->rankOf() >= 2) {
        auto d0 = extraInput->sizeAt(extraInput->rankOf() - 2);
        auto d1 = extraInput->sizeAt(extraInput->rankOf() - 1);
        looksLikeBias = (d0 == tq && d1 == tv) || (d0 == tv && d1 == tq);
      }
      if (looksLikeBias) {
        attentionBias = extraInput;
      }
    }
  }

  bool hasAttentionBias = (attentionBias != nullptr && !attentionBias->isEmpty());

  bool useCausalMask = block.numB() > 0 ? B_ARG(0) : false;

  NDArray* keysForAttention = keys;
  NDArray* valuesForAttention = values;
  NDArray* keysCastOwner = nullptr;
  NDArray* valuesCastOwner = nullptr;

  if (keysForAttention->dataType() != queries->dataType()) {
    keysCastOwner = keysForAttention->cast(queries->dataType());
    keysForAttention = keysCastOwner;
  }
  if (values == keys) {
    valuesForAttention = keysForAttention;
  } else if (valuesForAttention->dataType() != queries->dataType()) {
    valuesCastOwner = valuesForAttention->cast(queries->dataType());
    valuesForAttention = valuesCastOwner;
  }

  auto cleanupAttentionInputs = [&]() {
    if (keysCastOwner != nullptr) delete keysCastOwner;
    if (valuesCastOwner != nullptr) delete valuesCastOwner;
    if (slicedBiasOwner != nullptr) delete slicedBiasOwner;
  };

  // 4D path: route to executeSDPA4D which handles bias, GQA, causal mask, and all dtypes via MKL GEMV.
  // This avoids FlashAttentionHelper's O(headsPerKvHead * seqKV * headDim) K/V tiling.
  if (rank == 4) {
    executeSDPA4D(queries, keysForAttention, valuesForAttention, output, static_cast<float>(scale),
                  block.launchContext(), hasAttentionBias ? attentionBias : nullptr, useCausalMask);
    if (!attentionScores->isEmpty()) attentionScores->nullify();
    if (!attentionLogits->isEmpty())  attentionLogits->nullify();
    cleanupAttentionInputs();
    return sd::Status::OK;
  }

  // Rank 2/3 with bias: delegate to FlashAttentionHelper.
  if (hasAttentionBias) {
    NDArray* q = queries;
    NDArray* k = keysForAttention;
    NDArray* v = valuesForAttention;
    NDArray* out = output;
    bool reshapedQ = false;

    if (rank == 2) {
      reshapedQ = true;
      std::vector<sd::LongType> qShape = {1, queries->sizeAt(0), queries->sizeAt(1)};
      std::vector<sd::LongType> vShape = {1, valuesForAttention->sizeAt(0), valuesForAttention->sizeAt(1)};
      q = queries->reshape('c', qShape);
      v = valuesForAttention->reshape('c', vShape);
      if (keysForAttention != valuesForAttention) {
        std::vector<sd::LongType> kShape = {1, keysForAttention->sizeAt(0), keysForAttention->sizeAt(1)};
        k = keysForAttention->reshape('c', kShape);
      } else {
        k = v;
      }
      std::vector<sd::LongType> outShape = {1, output->sizeAt(0), output->sizeAt(1)};
      out = output->reshape('c', outShape);
    }

    NDArray* biasCast = nullptr;
    NDArray* biasForHelper = attentionBias;
    if (attentionBias->dataType() != q->dataType()) {
      biasCast = attentionBias->cast(q->dataType());
      biasForHelper = biasCast;
    }

    FlashAttentionHelper::Config config;
    config.scale = static_cast<float>(scale);
    config.isCausal = block.numB() > 0 ? B_ARG(0) : false;
    config.dropout = 0.0f;
    config.numHeads = 1;
    config.numKvHeads = 1;

    FlashAttentionHelper::forward(q, k, v, out, config,
                                  nullptr, attentionScores, attentionLogits,
                                  block.launchContext(), biasForHelper);

    if (biasCast != nullptr) delete biasCast;
    if (reshapedQ) {
      delete q;
      delete v;
      if (keysForAttention != valuesForAttention) delete k;
      delete out;
    }

    cleanupAttentionInputs();
    return sd::Status::OK;
  }

  // No bias, rank 2/3: try OneDNN 3D graph path, fall back to FlashAttentionHelper
  {
    NDArray *q3d = nullptr, *k3d = nullptr, *v3d = nullptr, *out3d = nullptr;
    bool needReshape = (rank == 2);

    if (needReshape) {
      std::vector<sd::LongType> shape3d_q = {1, queries->sizeAt(0), queries->sizeAt(1)};
      std::vector<sd::LongType> shape3d_k = {1, keysForAttention->sizeAt(0), keysForAttention->sizeAt(1)};
      std::vector<sd::LongType> shape3d_v = {1, valuesForAttention->sizeAt(0), valuesForAttention->sizeAt(1)};
      std::vector<sd::LongType> shape3d_out = {1, output->sizeAt(0), output->sizeAt(1)};
      q3d = queries->reshape('c', shape3d_q);
      k3d = keysForAttention->reshape('c', shape3d_k);
      v3d = valuesForAttention->reshape('c', shape3d_v);
      out3d = output->reshape('c', shape3d_out);
    } else {
      q3d = queries;
      k3d = keysForAttention;
      v3d = valuesForAttention;
      out3d = output;
    }

    bool onednnOk = executeSDPA3D(q3d, k3d, v3d, out3d, static_cast<float>(scale), block.launchContext());

    if (!onednnOk) {
      // OneDNN graph compilation failed for this shape/dtype — fall back to FlashAttentionHelper
      FlashAttentionHelper::Config config;
      config.scale = static_cast<float>(scale);
      config.isCausal = block.numB() > 0 ? B_ARG(0) : false;
      config.dropout = 0.0f;
      config.numHeads = 1;
      config.numKvHeads = 1;

      FlashAttentionHelper::forward(q3d, k3d, v3d, out3d, config,
                                    nullptr, attentionScores, attentionLogits,
                                    block.launchContext(), nullptr);
    } else {
      if (!attentionScores->isEmpty()) attentionScores->nullify();
      if (!attentionLogits->isEmpty()) attentionLogits->nullify();
    }

    if (needReshape) {
      delete q3d;
      delete k3d;
      delete v3d;
      delete out3d;
    }
  }

  cleanupAttentionInputs();
  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(dot_product_attention_v2, ENGINE_ONEDNN) {
  auto query = INPUT_VARIABLE(0);
  auto value = INPUT_VARIABLE(1);

  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;

  // Check masks — scalars (rank 0) and 1D arrays are scale constants, not masks.
  // Only rank >= 2 tensors count as actual attention masks.
  auto qMask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  auto vMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;
  bool hasMasks = (qMask != nullptr && !qMask->isEmpty() && qMask->rankOf() >= 2) ||
                  (vMask != nullptr && !vMask->isEmpty() && vMask->rankOf() >= 2);

  const auto qType = query->dataType();
  // Accept FP32, FP16, and BF16 unconditionally.  The decode path (seqQ=1) uses
  // FlashAttentionHelper → MmulHelper which handles all float types via MKL.
  // The prefill path (seqQ>1) uses OneDNN Graph which JIT-compiles for the type,
  // and executeSDPA4D auto-casts to FP32 if the ISA doesn't support the type natively.
  bool isSupportedType = (qType == DataType::FLOAT32 || qType == DataType::HALF
                          || qType == DataType::BFLOAT16);

  // GQA (mismatched Q/K head counts) only supported for FP32 via MKL batch GEMM path
  auto keys = block.width() > 2 ? INPUT_VARIABLE(2) : value;
  bool isGqa = (query->rankOf() == 4 && keys->rankOf() == 4 &&
                query->sizeAt(2) != keys->sizeAt(2));
  bool gqaUnsupported = isGqa && qType != DataType::FLOAT32;

  const auto rank = query->rankOf();

  Requirements req("ONEDNN GRAPH SDPA");
  req.expectFalse(makeInfoVariable(query->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(value->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectTrue(makeInfoVariable(isSupportedType, TYPE_MSG_INPUT), "Must be FLOAT32, HALF, or BFLOAT16") &&
      req.expectFalse(makeInfoVariable(hasMasks, "HAS_MASKS"), "Custom masks not supported") &&
      req.expectFalse(makeInfoVariable(gqaUnsupported, "IS_GQA_NON_FP32"), "GQA only supported for FP32 (MKL path)") &&
      req.expectEq(makeInfoVariable(dropout, "DROPOUT"), 0.0) &&
      req.expectGreaterEq(makeInfoVariable(rank, RANK_MSG_INPUT0), 2) &&
      req.expectLessEq(makeInfoVariable(rank, RANK_MSG_INPUT0), 4);

  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
