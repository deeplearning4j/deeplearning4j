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

namespace sd {
namespace ops {
namespace platforms {

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
    int dtype;  // 0=f32, 1=f16, 2=bf16
    bool is4D;  // true for 4D [batch, seq, heads, dim], false for 3D [batch, seq, dim]

    bool operator==(const Key& o) const {
      return batch == o.batch && seqQ == o.seqQ && seqKV == o.seqKV &&
             numHeads == o.numHeads && headDim == o.headDim &&
             dtype == o.dtype && is4D == o.is4D;
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
// Get or create compiled SDPA partition
static SDPACache::Entry& getSDPA(int64_t batch, int64_t seqQ, int64_t seqKV,
                                  int64_t numHeads, int64_t headDim, int dtype, bool is4D) {
  auto& cache = SDPACache::instance();
  SDPACache::Key key{batch, seqQ, seqKV, numHeads, headDim, dtype, is4D};

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
    cache.cache[key] = buildSDPAGraph4D(batch, seqQ, seqKV, numHeads, headDim, dt, cache.eng);
  } else {
    // For 3D, headDim is the full dimension, numHeads=1
    cache.cache[key] = buildSDPAGraph3D(batch, seqQ, seqKV, headDim, dt, cache.eng);
  }
  return cache.cache[key];
}

//////////////////////////////////////////////////////////////////////////
// Execute 3D SDPA - minimal overhead path
static void executeSDPA3D(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
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
    THROW_EXCEPTION("SDPA 3D graph compilation failed");
  }

  auto& cache = SDPACache::instance();
  auto& strm = getThreadStream(cache.eng);

  dg::logical_tensor::data_type dt = dg::logical_tensor::data_type::f32;
  if (dtype == 1) dt = dg::logical_tensor::data_type::f16;
  else if (dtype == 2) dt = dg::logical_tensor::data_type::bf16;

  // Create logical tensors with actual strides
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

  // Create tensors with direct buffer pointers
  dg::tensor t_query(query_lt, cache.eng, query->buffer());
  dg::tensor t_key(key_lt, cache.eng, key->buffer());
  dg::tensor t_scale(scale_lt, cache.eng, &scale);
  dg::tensor t_value(value_lt, cache.eng, value->buffer());
  dg::tensor t_output(output_lt, cache.eng, output->buffer());

  entry.cp.execute(strm, {t_query, t_key, t_scale, t_value}, {t_output});
  strm.wait();
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

    // Sum and normalize
    float sum = cblas_sasum(cols, row, 1);
    if (sum > 0.0f) {
      cblas_sscal(cols, 1.0f / sum, row, 1);
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// Execute 4D SDPA using MKL strided batch GEMM
static void executeSDPA4D_MKL(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
                               float scale, LaunchContext* context) {
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

  const MKL_INT qBatchStride = query->strideAt(0);
  const MKL_INT qSeqStride = query->strideAt(1);
  const MKL_INT qHeadStride = query->strideAt(2);

  const MKL_INT kBatchStride = key->strideAt(0);
  const MKL_INT kSeqStride = key->strideAt(1);
  const MKL_INT kHeadStride = key->strideAt(2);

  const MKL_INT vBatchStride = value->strideAt(0);
  const MKL_INT vSeqStride = value->strideAt(1);
  const MKL_INT vHeadStride = value->strideAt(2);

  const MKL_INT outBatchStride = output->strideAt(0);
  const MKL_INT outSeqStride = output->strideAt(1);
  const MKL_INT outHeadStride = output->strideAt(2);

  const MKL_INT scoreSize = seqQ * seqKV;
  const bool isGqa = (numKVHeads != numHeads);
  const MKL_INT headsPerGroup = isGqa ? (numHeads / numKVHeads) : 1;

  auto& scratch = MKLSDPABuffer::instance();
  float* allScores = scratch.ensureCapacity(numHeads * scoreSize);

  for (MKL_INT b = 0; b < batch; b++) {
    float* Q = qPtr + b * qBatchStride;
    float* K = kPtr + b * kBatchStride;
    float* V = vPtr + b * vBatchStride;
    float* O = outPtr + b * outBatchStride;

    if (!isGqa) {
      // Standard MHA: single batched GEMM for all heads
      cblas_sgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasTrans,
                                 seqQ, seqKV, headDim, 1.0f,
                                 Q, qSeqStride, qHeadStride,
                                 K, kSeqStride, kHeadStride,
                                 0.0f, allScores, seqKV, scoreSize, numHeads);
    } else {
      // GQA: each KV head group shares one K head
      // Process headsPerGroup Q heads against the same K head per group
      for (MKL_INT g = 0; g < numKVHeads; g++) {
        float* Qg = Q + g * headsPerGroup * qHeadStride;
        float* Kg = K + g * kHeadStride;
        float* Sg = allScores + g * headsPerGroup * scoreSize;
        // Batched GEMM: headsPerGroup Q heads @ same K head (K stride=0)
        cblas_sgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasTrans,
                                   seqQ, seqKV, headDim, 1.0f,
                                   Qg, qSeqStride, qHeadStride,
                                   Kg, kSeqStride, 0,  // stride_b=0: reuse same K head
                                   0.0f, Sg, seqKV, scoreSize, headsPerGroup);
      }
    }

    // Softmax for all heads
    for (MKL_INT h = 0; h < numHeads; h++) {
      mklSoftmaxInPlace(allScores + h * scoreSize, seqQ, seqKV, scale);
    }

    if (!isGqa) {
      // Standard MHA: single batched GEMM
      cblas_sgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                 seqQ, headDim, seqKV, 1.0f,
                                 allScores, seqKV, scoreSize,
                                 V, vSeqStride, vHeadStride,
                                 0.0f, O, outSeqStride, outHeadStride, numHeads);
    } else {
      // GQA: each KV head group shares one V head
      for (MKL_INT g = 0; g < numKVHeads; g++) {
        float* Sg = allScores + g * headsPerGroup * scoreSize;
        float* Vg = V + g * vHeadStride;
        float* Og = O + g * headsPerGroup * outHeadStride;
        cblas_sgemm_batch_strided(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                                   seqQ, headDim, seqKV, 1.0f,
                                   Sg, seqKV, scoreSize,
                                   Vg, vSeqStride, 0,  // stride_b=0: reuse same V head
                                   0.0f, Og, outSeqStride, outHeadStride, headsPerGroup);
      }
    }
  }
}
#endif

//////////////////////////////////////////////////////////////////////////
// Execute 4D SDPA with [batch, seq, heads, dim] layout
static void executeSDPA4D(NDArray* query, NDArray* key, NDArray* value, NDArray* output,
                          float scale, LaunchContext* context) {
#ifdef HAVE_MKL
  if (query->dataType() == DataType::FLOAT32) {
    executeSDPA4D_MKL(query, key, value, output, scale, context);
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

  // Create logical tensors with strided layouts
  auto query_lt = dg::logical_tensor(0, dt, q_shape, q_strides);
  auto key_lt = dg::logical_tensor(1, dt, k_shape, k_strides);
  auto scale_lt = dg::logical_tensor(2, dg::logical_tensor::data_type::f32, {1}, {1});
  auto value_lt = dg::logical_tensor(3, dt, v_shape, v_strides);
  auto output_lt = dg::logical_tensor(4, dt, out_shape, out_strides);

  // Create tensors pointing directly to input/output buffers - ZERO COPY
  dg::tensor t_query(query_lt, cache.eng, query->buffer());
  dg::tensor t_key(key_lt, cache.eng, key->buffer());
  dg::tensor t_scale(scale_lt, cache.eng, &scale);
  dg::tensor t_value(value_lt, cache.eng, value->buffer());
  dg::tensor t_output(output_lt, cache.eng, output->buffer());

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

  auto rank = queries->rankOf();
  bool isRank4 = (rank == 4);
  REQUIRE_TRUE(rank >= 2 && rank <= 4, 0,
               "dot_product_attention_v2: rank must be 2, 3, or 4, got %i", rank);

  auto scale = block.numT() > 0 ? T_ARG(0) : 0.0;
  if (scale <= 0) {
    scale = 1.0 / std::sqrt(static_cast<double>(queries->sizeAt(-1)));
  }

  // KV cache scatter: write current K/V into cache, then use full cache as K/V
  auto kvCacheK = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
  auto kvCacheV = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;
  auto cachePosInput = block.width() > 7 ? INPUT_VARIABLE(7) : nullptr;
  bool hasKvCache = (kvCacheK != nullptr && !kvCacheK->isEmpty()) &&
                    (kvCacheV != nullptr && !kvCacheV->isEmpty());
  bool useInPlaceKv = hasKvCache && (cachePosInput != nullptr);

  if (useInPlaceKv) {
    LongType cachePosVal = cachePosInput->e<LongType>(0);
    if (isRank4) {
      auto batch = keys->sizeAt(0);
      auto numKvHeads = keys->sizeAt(2);
      auto headDim = keys->sizeAt(3);
      std::vector<LongType> writeIdx = {0, batch, cachePosVal, cachePosVal + 1, 0, numKvHeads, 0, headDim};
      auto* kSlice = (*kvCacheK)(writeIdx);
      auto* vSlice = (*kvCacheV)(writeIdx);
      kSlice->assign(keys);
      vSlice->assign(values);
      delete kSlice;
      delete vSlice;
    } else {
      auto batch = keys->sizeAt(0);
      auto features = keys->sizeAt(2);
      std::vector<LongType> writeIdx = {0, batch, cachePosVal, cachePosVal + 1, 0, features};
      auto* kSlice = (*kvCacheK)(writeIdx);
      auto* vSlice = (*kvCacheV)(writeIdx);
      kSlice->assign(keys);
      vSlice->assign(values);
      delete kSlice;
      delete vSlice;
    }
    // Use full cache as K/V for attention
    keys = kvCacheK;
    values = kvCacheV;
  }

  // Detect attention bias at input 5 (when input 6 is absent, input 5 is bias not KV cache)
  NDArray* attentionBias = nullptr;
  auto extraInput = block.width() > 5 ? INPUT_VARIABLE(5) : nullptr;
  auto extraInput2 = block.width() > 6 ? INPUT_VARIABLE(6) : nullptr;
  if (extraInput != nullptr && !extraInput->isEmpty() &&
      (extraInput2 == nullptr || extraInput2->isEmpty())) {
    // Use working queries for shape (handle rank 2 by using original queries shape)
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

  bool hasAttentionBias = (attentionBias != nullptr && !attentionBias->isEmpty());

  // When attention bias is present, delegate to FlashAttentionHelper which handles it
  if (hasAttentionBias) {
    // Handle rank 2 by reshaping to 3D
    NDArray* q = queries;
    NDArray* k = keys;
    NDArray* v = values;
    NDArray* out = output;
    bool reshapedQ = false;

    if (rank == 2) {
      reshapedQ = true;
      std::vector<sd::LongType> qShape = {1, queries->sizeAt(0), queries->sizeAt(1)};
      std::vector<sd::LongType> vShape = {1, values->sizeAt(0), values->sizeAt(1)};
      q = queries->reshape('c', qShape);
      v = values->reshape('c', vShape);
      if (keys != values) {
        std::vector<sd::LongType> kShape = {1, keys->sizeAt(0), keys->sizeAt(1)};
        k = keys->reshape('c', kShape);
      } else {
        k = v;
      }
      std::vector<sd::LongType> outShape = {1, output->sizeAt(0), output->sizeAt(1)};
      out = output->reshape('c', outShape);
    }

    // Cast attention bias to query dtype if needed
    std::unique_ptr<NDArray> biasCastOwner;
    NDArray* biasForHelper = attentionBias;
    if (attentionBias->dataType() != q->dataType()) {
      biasCastOwner.reset(attentionBias->cast(q->dataType()));
      biasForHelper = biasCastOwner.get();
    }

    FlashAttentionHelper::Config config;
    config.scale = static_cast<float>(scale);
    config.isCausal = block.numB() > 0 ? B_ARG(0) : false;
    config.dropout = 0.0f;
    if (isRank4) {
      config.numHeads = q->sizeAt(2);
      config.numKvHeads = k->sizeAt(2);
    } else {
      config.numHeads = 1;
      config.numKvHeads = 1;
    }

    FlashAttentionHelper::forward(q, k, v, out, config,
                                  nullptr, attentionScores, attentionLogits,
                                  block.launchContext(), biasForHelper);

    if (reshapedQ) {
      delete q;
      delete v;
      if (keys != values) delete k;
      delete out;
    }

    return sd::Status::OK;
  }

  // No bias: use fast fused OneDNN graph path
  if (rank == 4) {
    // 4D path: [batch, seq, heads, dim]
    executeSDPA4D(queries, keys, values, output, static_cast<float>(scale), block.launchContext());
  } else {
    // 2D or 3D path
    NDArray *q3d = nullptr, *k3d = nullptr, *v3d = nullptr, *out3d = nullptr;
    bool needReshape = (rank == 2);

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

    executeSDPA3D(q3d, k3d, v3d, out3d, static_cast<float>(scale), block.launchContext());

    if (!attentionScores->isEmpty()) attentionScores->nullify();
    if (!attentionLogits->isEmpty()) attentionLogits->nullify();

    if (needReshape) {
      delete q3d;
      delete k3d;
      delete v3d;
      delete out3d;
    }
  }

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(dot_product_attention_v2, ENGINE_CPU) {
  auto query = INPUT_VARIABLE(0);
  auto value = INPUT_VARIABLE(1);

  auto dropout = block.numT() > 1 ? T_ARG(1) : 0.0;

  // Check masks
  auto qMask = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
  auto vMask = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;
  bool hasMasks = (qMask != nullptr && !qMask->isEmpty()) || (vMask != nullptr && !vMask->isEmpty());

  const auto qType = query->dataType();
  bool isSupportedType = (qType == DataType::FLOAT32);
  if (qType == DataType::BFLOAT16) {
#if HAVE_ONEDNN
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isSupportedType = (isa >= dnnl_cpu_isa_avx512_core_bf16);
#endif
  } else if (qType == DataType::HALF) {
#if HAVE_ONEDNN
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isSupportedType = (isa >= dnnl_cpu_isa_avx512_core_amx_fp16);
#endif
  }

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
