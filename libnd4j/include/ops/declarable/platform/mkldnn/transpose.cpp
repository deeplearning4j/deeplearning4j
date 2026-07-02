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
// OneDNN implementation of transpose/permute operations
// Uses reorder primitive for efficient memory layout transformations
//

#include <helpers/MKLDNNStream.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include "mkldnnUtils.h"

using namespace dnnl;

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////
static dnnl::memory::data_type getDnnlDataType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32: return dnnl::memory::data_type::f32;
    case DataType::BFLOAT16: return dnnl::memory::data_type::bf16;
    case DataType::HALF: return dnnl::memory::data_type::f16;
    case DataType::DOUBLE: return dnnl::memory::data_type::f64;
    case DataType::INT8: return dnnl::memory::data_type::s8;
    case DataType::UINT8: return dnnl::memory::data_type::u8;
    case DataType::INT32: return dnnl::memory::data_type::s32;
    default: return dnnl::memory::data_type::f32;
  }
}

static void transposeMKLDNN(NDArray* x, NDArray* z, const std::vector<LongType>& permutation) {
  // The oneDNN reorder path that used to live here mis-materializes non-contiguous permuted
  // VIEWS: for the SmolDocling patch-embedding weight permute [768,3,16,16] -> [16,16,3,768]
  // it produced 0.00411987 where the correct (and CUDA) value is 0.01147461, corrupting every
  // downstream CPU vision feature and yielding degenerate VLM output. The setup looked correct
  // (permuted-stride source desc + contiguous dst) but the strided-source reorder is wrong for
  // this case. Materialize via the reference nd4j path used by the generic transpose op
  // (generic/shape/transpose.cpp): a permuted VIEW + stride-aware assign, which is correct for
  // any x (contiguous or view) and any permutation.
  std::vector<LongType> perm(permutation);
  NDArray* permuted = x->permute(perm, false, false);
  z->assign(permuted);
  delete permuted;
}

PLATFORM_IMPL(transpose, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // DSP view path: output is a view of input with transposed strides already set.
  // The shape info is correct — no data movement needed. Calling transposeMKLDNN
  // would corrupt the shared buffer by reordering in-place.
  if (input->dataBuffer() == output->dataBuffer()) {
    return sd::Status::OK;
  }

  auto xRank = input->rankOf();
  REQUIRE_TRUE(xRank <= 6, 0, "TRANSPOSE_MKLDNN OP: rank must be <= 6, but got rank = %i", xRank);

  // Get permutation
  std::vector<LongType> permutation;
  if (block.width() > 1) {
    auto permArr = INPUT_VARIABLE(1);
    for (LongType i = 0; i < permArr->lengthOf(); i++) {
      permutation.push_back(permArr->e<LongType>(i));
    }
  } else if (block.numI() > 0) {
    for (int i = 0; i < block.numI(); i++) {
      permutation.push_back(INT_ARG(i));
    }
  } else {
    // Default: reverse dimensions
    for (int i = xRank - 1; i >= 0; i--) {
      permutation.push_back(i);
    }
  }

  transposeMKLDNN(input, output, permutation);

  return sd::Status::OK;
}

PLATFORM_CHECK(transpose, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  // OneDNN supports f32, bf16, f16, and integer types for reorder
  // Note: DOUBLE (f64) is NOT supported by oneDNN reorder primitives
  // bf16 requires AVX512_CORE_BF16, f16 requires AVX512_CORE_AMX_FP16
  auto xType = x->dataType();
  bool isSupportedType = (xType == DataType::FLOAT32 ||
                          xType == DataType::INT8 || xType == DataType::UINT8 ||
                          xType == DataType::INT32);
  if (!isSupportedType && xType == DataType::BFLOAT16) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isSupportedType = (isa >= dnnl_cpu_isa_avx512_core_bf16);
  }
  if (!isSupportedType && xType == DataType::HALF) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isSupportedType = (isa >= dnnl_cpu_isa_avx512_core_amx_fp16);
  }

  Requirements req("ONEDNN TRANSPOSE OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectTrue(makeInfoVariable(isSupportedType, TYPE_MSG_INPUT), EXPECTED_TRUE);
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
// PERMUTE - similar to transpose but with explicit permutation order
PLATFORM_IMPL(permute, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  // DSP view path: output is a view of input with permuted strides already set.
  // The shape info is correct — no data movement needed. Calling transposeMKLDNN
  // would corrupt the shared buffer by reordering in-place.
  if (input->dataBuffer() == output->dataBuffer()) {
    return sd::Status::OK;
  }

  auto xRank = input->rankOf();
  REQUIRE_TRUE(xRank <= 6, 0, "PERMUTE_MKLDNN OP: rank must be <= 6, but got rank = %i", xRank);

  // Get permutation from arguments
  std::vector<LongType> permutation;
  if (block.width() > 1) {
    auto permArr = INPUT_VARIABLE(1);
    for (LongType i = 0; i < permArr->lengthOf(); i++) {
      permutation.push_back(permArr->e<LongType>(i));
    }
  } else {
    for (int i = 0; i < block.numI(); i++) {
      permutation.push_back(INT_ARG(i));
    }
  }

  REQUIRE_TRUE(permutation.size() == xRank, 0, "PERMUTE_MKLDNN OP: permutation size must equal rank");

  transposeMKLDNN(input, output, permutation);

  return sd::Status::OK;
}

PLATFORM_CHECK(permute, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  // OneDNN supports f32, bf16, f16, and integer types for reorder
  // Note: DOUBLE (f64) is NOT supported by oneDNN reorder primitives
  // bf16 requires AVX512_CORE_BF16, f16 requires AVX512_CORE_AMX_FP16
  auto xType = x->dataType();
  bool isSupportedType = (xType == DataType::FLOAT32 ||
                          xType == DataType::INT8 || xType == DataType::UINT8 ||
                          xType == DataType::INT32);
  if (!isSupportedType && xType == DataType::BFLOAT16) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isSupportedType = (isa >= dnnl_cpu_isa_avx512_core_bf16);
  }
  if (!isSupportedType && xType == DataType::HALF) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    isSupportedType = (isa >= dnnl_cpu_isa_avx512_core_amx_fp16);
  }
  bool typesMatch = (x->dataType() == z->dataType());

  Requirements req("ONEDNN PERMUTE OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectTrue(makeInfoVariable(isSupportedType, TYPE_MSG_INPUT), EXPECTED_TRUE) &&
      req.expectTrue(makeInfoVariable(typesMatch, "TYPES MATCH"), EXPECTED_TRUE);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
