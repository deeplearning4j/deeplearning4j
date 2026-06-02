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
//  @author Oleg Semeniv <oleg.semeniv@gmail.com>
//
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
static void softmaxMKLDNN(NDArray* x, NDArray* z, const sd::LongType axis) {
  dnnl::memory::dims shape = *x->getShapeAsFlatVector();

  const sd::LongType xRank = x->rankOf();

  dnnl::memory::format_tag xFormat = onednnUtils::getFormat(*x);
  dnnl::memory::format_tag zFormat = onednnUtils::getFormat(*z);

  // optimized cases
  if (2 == xRank && 0 == axis) {
    xFormat = dnnl::memory::format_tag::ba;
    zFormat = dnnl::memory::format_tag::ba;
  } else if (4 == xRank && 1 == axis && (x->sizeAt(2) * x->sizeAt(3)) > 1) {
    xFormat = dnnl::memory::format_tag::acdb;
   zFormat = dnnl::memory::format_tag::acdb;
  }

  // Use actual input data type - OneDNN supports f32, bf16, f16 for softmax
  dnnl::memory::data_type xType = getOneDnnDataType(x->dataType());

  dnnl::memory::desc x_mkl_md, x_user_md, z_mkl_md, z_user_md;

  x_user_md = x_mkl_md = dnnl::memory::desc(shape, xType, xFormat);
  onednnUtils::setBlockStrides(*x, x_user_md);

  // z
  z_user_md = z_mkl_md = dnnl::memory::desc(shape, xType, zFormat);
  onednnUtils::setBlockStrides(*z, z_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // operation primitive description
  // OneDNN 3.x API: primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, axis)
  dnnl::softmax_forward::primitive_desc op_prim_desc(engine, dnnl::prop_kind::forward_inference,
                                                      dnnl::algorithm::softmax_accurate, x_mkl_md, z_mkl_md, axis);

  // arguments (memory buffers) necessary for calculations
  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  // provide memory buffers and check whether reorder is required

  // input
  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  // z
  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

  // run calculations
  dnnl::softmax_forward(op_prim_desc).execute(stream, args);

  // reorder outputs if necessary
  if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

PLATFORM_IMPL(softmax, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto output = OUTPUT_VARIABLE(0);

  const int rank = input->rankOf();
  int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  if (dim < 0) {
    dim += rank;
  }

  REQUIRE_TRUE(dim < rank && dim >= 0, 0,
               "SOFTMAX_MKLDNN OP: the value of input integer parameter (dimension) must be less than input array rank "
               "%i, but got dimension = %i instead !",
               rank, dim);

  REQUIRE_TRUE(rank <= 6, 0, "SOFTMAX_MKLDNN OP: the rank of input must be less or qual 6, but got rank = %i instead !",
               rank);

  // mkldnnSoftMax
  softmaxMKLDNN(input, output, dim);

  return sd::Status::OK;
}

//////////////////////////////////////////////////////////////////////
// Check if data type is supported by OneDNN softmax.
// NOTE: HALF (FP16) is excluded because OneDNN FP16 requires AVX-512-FP16
// hardware that is not universally available on CPU. HALF inputs fall back
// to the generic CPU softmax implementation which upcasts to float32 internally.
static bool isSupportedSoftmaxType(DataType dt) {
  return dt == DataType::FLOAT32 || dt == DataType::BFLOAT16;
}

PLATFORM_CHECK(softmax, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto z = OUTPUT_VARIABLE(0);

  Requirements req("ONEDNN SOFTMAX OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 7) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectTrue(makeInfoVariable(isSupportedSoftmaxType(x->dataType()), TYPE_MSG_INPUT),
                     "Must be FLOAT32 or BFLOAT16") &&
      req.expectTrue(makeInfoVariable(isSupportedSoftmaxType(z->dataType()), TYPE_MSG_OUTPUT),
                     "Must be FLOAT32 or BFLOAT16") &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT),
                   makeInfoVariable(z->dataType(), TYPE_MSG_OUTPUT));
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////
static void softmaxBpMKLDNN(NDArray* x, NDArray* dLdz, NDArray* dLdx, const sd::LongType axis) {
  dnnl::memory::desc x_user_md, x_mkl_md, dLdx_mkl_md, dLdx_user_md, dLdz_mkl_md, dLdz_user_md;

  // Use actual input data type - OneDNN supports f32, bf16, f16 for softmax backward
  auto dataType = getOneDnnDataType(x->dataType());

  // x
  x_mkl_md = x_user_md =
      dnnl::memory::desc(*x->getShapeAsFlatVector(), dataType, onednnUtils::getFormat(*x));
  onednnUtils::setBlockStrides(*x, x_user_md);

  // dLdx
  dLdx_mkl_md = dLdx_user_md =
      dnnl::memory::desc(*dLdx->getShapeAsFlatVector(), dataType, onednnUtils::getFormat(*dLdx));
  onednnUtils::setBlockStrides(*dLdx, dLdx_user_md);

  // dLdz
  dLdz_mkl_md = dLdz_user_md =
      dnnl::memory::desc(*dLdz->getShapeAsFlatVector(), dataType, onednnUtils::getFormat(*dLdz));
  onednnUtils::setBlockStrides(*dLdz, dLdz_user_md);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // operation primitive description
  // OneDNN 3.x API: forward primitive_desc(engine, prop_kind, algorithm, src_md, dst_md, axis)
  dnnl::softmax_forward::primitive_desc op_ff_prim_desc(engine, dnnl::prop_kind::forward_training,
                                                         dnnl::algorithm::softmax_accurate, x_mkl_md, x_mkl_md, axis);

  // OneDNN 3.x API: backward primitive_desc(engine, algorithm, diff_src_md, diff_dst_md, dst_md, axis, hint_fwd_pd)
  dnnl::softmax_backward::primitive_desc op_bp_prim_desc(engine, dnnl::algorithm::softmax_accurate,
                                                          dLdx_mkl_md, dLdz_mkl_md, x_mkl_md, axis, op_ff_prim_desc);

  // arguments (memory buffers) necessary for calculations
  std::unordered_map<int, dnnl::memory> argsbp, argsff;

  dnnl::stream stream(engine);

  // provide memory buffers and check whether reorder is required for forward
  // input
  onednnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_ff_prim_desc.src_desc(), argsff[DNNL_ARG_SRC]);

  // dLdz
  onednnUtils::loadDataToMklStream(*dLdz, engine, stream, dLdz_user_md, op_bp_prim_desc.diff_dst_desc(),
                                   argsbp[DNNL_ARG_DIFF_DST]);

  // dLdx
  auto dLdx_user_mem = onednnUtils::loadDataToMklStream(*dLdx, engine, stream, dLdx_user_md, op_ff_prim_desc.src_desc(),
                                                        argsff[DNNL_ARG_DST]);

  // check and arg set for backprob
  argsbp[DNNL_ARG_DIFF_SRC] = argsff[DNNL_ARG_DST];
  argsbp[DNNL_ARG_DST] = argsff[DNNL_ARG_DST];


  // run calculations backward
  dnnl::softmax_backward(op_bp_prim_desc).execute(stream, argsbp);

  // reorder outputs if necessary
  if (op_ff_prim_desc.dst_desc() != dLdx_user_mem.get_desc())
    dnnl::reorder(argsff[DNNL_ARG_DST], dLdx_user_mem).execute(stream, argsff[DNNL_ARG_DST], dLdx_user_mem);

  stream.wait();
}

PLATFORM_IMPL(softmax_bp, ENGINE_CPU) {
  auto input = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto softmaxOutput = INPUT_VARIABLE(2);
  auto dLdx = OUTPUT_VARIABLE(0);
  dLdx->assign(softmaxOutput);

  const sd::LongType rank = input->rankOf();
  const sd::LongType dLdzRank = dLdz->rankOf();
  int dim = block.getIArguments()->size() > 0 ? INT_ARG(0) : rank - 1;

  if (dim < 0) {
    dim += rank;
  }

  REQUIRE_TRUE(dim < rank && dim >= 0, 0,
               "SOFTMAX_MKLDNN_BP OP: the value of input integer parameter (dimension) must be less than input array "
               "rank %i, but got dimension = %i instead !",
               rank, dim);

  REQUIRE_TRUE(rank <= 6 && dLdzRank <= 6, 0,
               "SOFTMAX_MKLDNN_BP OP: the rank of input and dLdz must be less or qual 6, but got input rank = %i and "
               "dLdz rank rank = %i instead !",
               rank, dLdzRank);

  // mkldnnSoftMax
  softmaxBpMKLDNN(input, dLdz, dLdx, dim);

  return sd::Status::OK;
}

PLATFORM_CHECK(softmax_bp, ENGINE_CPU) {
  auto x = INPUT_VARIABLE(0);
  auto dLdz = INPUT_VARIABLE(1);
  auto softmaxOutput = INPUT_VARIABLE(2);
  auto dLdx = OUTPUT_VARIABLE(0);
  dLdx->assign(softmaxOutput);

  Requirements req("ONEDNN SOFTMAX_BP OP");
  req.expectFalse(makeInfoVariable(x->isEmpty(), IS_EMPTY_MSG_INPUT0), EXPECTED_FALSE) &&
      req.expectFalse(makeInfoVariable(dLdz->isEmpty(), IS_EMPTY_MSG_INPUT1), EXPECTED_FALSE) &&
      req.expectLess(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), 7) &&
      req.expectEq(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT0), makeInfoVariable(dLdz->rankOf(), RANK_MSG_INPUT1)) &&
      req.expectGreater(makeInfoVariable(x->rankOf(), RANK_MSG_INPUT), 0) &&
      req.expectTrue(makeInfoVariable(isSupportedSoftmaxType(x->dataType()), TYPE_MSG_INPUT0),
                     "Must be FLOAT32 or BFLOAT16") &&
      req.expectTrue(makeInfoVariable(isSupportedSoftmaxType(dLdz->dataType()), TYPE_MSG_INPUT1),
                     "Must be FLOAT32 or BFLOAT16") &&
      req.expectTrue(makeInfoVariable(isSupportedSoftmaxType(dLdx->dataType()), TYPE_MSG_OUTPUT),
                     "Must be FLOAT32 or BFLOAT16") &&
      req.expectEq(makeInfoVariable(x->dataType(), TYPE_MSG_INPUT0),
                   makeInfoVariable(dLdz->dataType(), TYPE_MSG_INPUT1)) &&
      req.expect(
          makeShapeInfoVariable(x, SHAPE_MSG_INPUT0), makeShapeInfoVariable(dLdz, SHAPE_MSG_INPUT1),
          [](const decltype(x)& l, const decltype(dLdz)& r) {
            for (int i = 0; i < l->rankOf(); i++) {
              if (l->sizeAt(i) != r->sizeAt(i)) {
                return false;
              }
            }
            return true;
          },
          EXPECTED_EQ_MSG);
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
