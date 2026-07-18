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
// OneDNN implementation of exponential function: e^x
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
// Get OneDNN data type from NDArray
static dnnl::memory::data_type getOneDnnDataType(DataType dt) {
  switch (dt) {
    case DataType::FLOAT32:
      return dnnl::memory::data_type::f32;
    case DataType::BFLOAT16:
      return dnnl::memory::data_type::bf16;
    case DataType::HALF:
      return dnnl::memory::data_type::f16;
    default:
      return dnnl::memory::data_type::f32;
  }
}

//////////////////////////////////////////////////////////////////////
// Check if data type is supported by OneDNN eltwise operations (runtime ISA detection)
static bool isSupportedEltwiseType(DataType dt) {
  if (dt == DataType::FLOAT32) return true;
  if (dt == DataType::BFLOAT16) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    return (isa >= dnnl_cpu_isa_avx512_core_bf16);
  }
  if (dt == DataType::HALF) {
    dnnl_cpu_isa_t isa = dnnl_get_effective_cpu_isa();
    return (isa >= dnnl_cpu_isa_avx512_core_amx_fp16);
  }
  return false;
}

//////////////////////////////////////////////////////////////////////
static void expMKLDNN(NDArray* x, NDArray* z) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  // Use actual input data type - OneDNN supports f32, bf16, f16 for eltwise operations
  auto dataType = getOneDnnDataType(x->dataType());

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, dataType, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  z_user_md = z_mkl_md = dnnl::memory::desc(shape, dataType, onednnUtils::getFormat(*z));
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  dnnl::primitive_attr attr;

  // exp: e^x
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, alpha, beta)
  dnnl::eltwise_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      algorithm::eltwise_exp, x_mkl_md, z_mkl_md, 0.f, 0.f);

  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

  dnnl::eltwise_forward(op_prim_desc).execute(stream, args);

  if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

PLATFORM_IMPL(Exp, ENGINE_ONEDNN) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const sd::LongType rank = input->rankOf();
  REQUIRE_TRUE(rank <= 6, 0, "EXP_MKLDNN OP: the rank of input must be less or equal 6, but got rank = %i instead !",
               rank);

  expMKLDNN(input, output);

  return sd::Status::OK;
}

PLATFORM_CHECK(Exp, ENGINE_ONEDNN) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN EXP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectTrue(makeInfoVariable(isSupportedEltwiseType(x->dataType()), TYPE_MSG_INPUT),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectTrue(makeInfoVariable(isSupportedEltwiseType(z->dataType()), TYPE_MSG_OUTPUT),
                     "Must be FLOAT32, BFLOAT16, or HALF") &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT),
                   makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT));
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
