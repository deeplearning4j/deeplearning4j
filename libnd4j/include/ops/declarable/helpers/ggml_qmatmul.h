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
// ggml_qmatmul.h — Runtime quantized matmul: activations x GGML-packed weights
//
// out[m,n] = sum_k A[m,k] * dequant(W[n,k])  (fp32 accumulation)
//
// Supported quantization formats (v1): Q8_0, Q4_K, Q6_K.
// Additional formats slot in via the dispatch tables in the .cpp/.cu files.
//
// @author generated for deeplearning4j
//

#ifndef LIBND4J_GGML_QMATMUL_H
#define LIBND4J_GGML_QMATMUL_H

#include <ops/declarable/helpers/helpers.h>

namespace sd {
namespace ops {
namespace helpers {

// Block sizes (bytes) matching the GGML canonical layout used in ggml_dequantize.cpp/.cu
//   Q8_0  : 2B FP16 d   + 32B int8 qs            = 34 bytes / 32 elements
//   Q4_K  : 2B FP16 d   + 2B FP16 dmin + 12B scales + 128B nibbles = 144 bytes / 256 elements
//   Q6_K  : 128B ql     + 64B qh + 16B int8 scales + 2B FP16 d     = 210 bytes / 256 elements

static constexpr int GGML_QMM_BLOCK_Q8_0  = 34;
static constexpr int GGML_QMM_ELEMS_Q8_0  = 32;

static constexpr int GGML_QMM_BLOCK_Q4_K  = 144;
static constexpr int GGML_QMM_ELEMS_Q4_K  = 256;

static constexpr int GGML_QMM_BLOCK_Q6_K  = 210;
static constexpr int GGML_QMM_ELEMS_Q6_K  = 256;

// ggmlQMatMul —
//   activations  : FLOAT32 or HALF, rank-2 [M,K] or rank-3 [B,S,K] (treated as [B*S,K])
//   packedWeights: INT8 (raw GGML bytes), rank-1  [N * (K/blockSize) * bytesPerBlock]
//   output       : FLOAT32 or HALF, same rank as activations with K replaced by N
//   quantType    : GgmlQuantType enum value (from ggml_dequantize.h)
//   N            : number of output rows in the weight matrix
//   K            : inner dimension (must be divisible by blockSize for quantType)
//   outputDtype  : 0 = FLOAT32, 1 = HALF
//
SD_LIB_HIDDEN void ggmlQMatMul(sd::LaunchContext* context,
                                sd::NDArray* activations,
                                sd::NDArray* packedWeights,
                                sd::NDArray* output,
                                int quantType,
                                sd::LongType N,
                                sd::LongType K,
                                int outputDtype);

// Returns the number of bytes per block for a given quantType.
// Defined inline (not SD_LIB_HIDDEN) to avoid duplicate-symbol link errors
// between the CPU (.cpp) and CUDA (.cu) translation units.
// Returns -1 for unsupported types.
static SD_INLINE int ggmlQMatMulBytesPerBlock(int quantType) {
    switch (quantType) {
        case 4:  return GGML_QMM_BLOCK_Q8_0;   // Q8_0
        case 8:  return GGML_QMM_BLOCK_Q4_K;   // Q4_K
        case 10: return GGML_QMM_BLOCK_Q6_K;   // Q6_K
        default: return -1;
    }
}

// Returns the number of elements per block for a given quantType.
// Defined inline (not SD_LIB_HIDDEN) to avoid duplicate-symbol link errors.
// Returns -1 for unsupported types.
static SD_INLINE int ggmlQMatMulElemsPerBlock(int quantType) {
    switch (quantType) {
        case 4:  return GGML_QMM_ELEMS_Q8_0;   // Q8_0
        case 8:  return GGML_QMM_ELEMS_Q4_K;   // Q4_K
        case 10: return GGML_QMM_ELEMS_Q6_K;   // Q6_K
        default: return -1;
    }
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_GGML_QMATMUL_H
