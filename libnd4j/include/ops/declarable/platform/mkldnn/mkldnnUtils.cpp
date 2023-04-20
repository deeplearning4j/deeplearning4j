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
// @author saudet
// @author Yurii Shyrma (iuriish@yahoo.com)
//
#include "mkldnnUtils.h"

#include <dnnl_types.h>
#include <ops/declarable/helpers/convolutions.h>

using namespace dnnl;

namespace sd {
namespace onednnUtils {

//////////////////////////////////////////////////////////////////////
void getDims(const NDArray* array, const int rank, dnnl::memory::dims& mklDims) {
  std::vector<int64_t> vDims(rank);
  for (auto i = 0; i < rank; i++) {
    vDims[i] = array->sizeAt(i);
  }
  mklDims = dnnl::memory::dims(vDims);
}
//////////////////////////////////////////////////////////////////////
dnnl::memory::format_tag getFormat(const NDArray& arr) {
  dnnl::memory::format_tag result;

  switch (arr.rankOf()) {
    case 1:
      result = dnnl::memory::format_tag::a;
      break;
    case 2:
      result = arr.ordering() == 'c' ? dnnl::memory::format_tag::ab : dnnl::memory::format_tag::ba;
      break;
    case 3:
      result = arr.ordering() == 'c' ? dnnl::memory::format_tag::abc : dnnl::memory::format_tag::cba;
      break;
    case 4:
      result = dnnl::memory::format_tag::abcd;
      break;
    case 5:
      result = dnnl::memory::format_tag::abcde;
      break;
    case 6:
      result = dnnl::memory::format_tag::abcdef;
      break;
    default:
      throw std::invalid_argument("MKLDNN getFormat: do we really want to use arras with rank > 6 ?");
  }

  return result;
}

//////////////////////////////////////////////////////////////////////
void setBlockStrides(const NDArray& array, dnnl::memory::desc& mklMd, const std::vector<int>& permut) {
  if (array.ews() != 1 || (array.rankOf() > 3 && array.ordering() == 'f') || !permut.empty()) {
    mklMd.data.format_kind = dnnl_blocked;  // overrides format

    if (permut.empty())
      for (auto i = 0; i < array.rankOf(); ++i) mklMd.data.format_desc.blocking.strides[i] = array.strideAt(i);
    else {
      if (array.rankOf() != permut.size())
        throw std::invalid_argument("mkldnnUtils::setBlockStrides: size of permut vector is not equal to array rank !");
      for (auto i = 0; i < array.rankOf(); ++i) mklMd.data.format_desc.blocking.strides[i] = array.strideAt(permut[i]);
    }
  }
}
////////////////////////////////////////////////////////////////////////////////////////////////
dnnl::memory loadDataToMklStream(const NDArray& array, const dnnl::engine& engine, const dnnl::stream& stream,
                                 const dnnl::memory::desc& user_md, const dnnl::memory::desc& primitive_md,
                                 dnnl::memory& arg) {
  auto user_mem = dnnl::memory(user_md, engine, const_cast<NDArray&>(array).buffer());
  const bool bReorder = primitive_md != user_mem.get_desc();
  auto mkl_mem = bReorder ? dnnl::memory(primitive_md, engine) : user_mem;
  if (bReorder) dnnl::reorder(user_mem, mkl_mem).execute(stream, user_mem, mkl_mem);
  arg = mkl_mem;
  return user_mem;
}

//////////////////////////////////////////////////////////////////////
void poolingONEDNN(const NDArray* input, NDArray* output, const int kD, const int kH, const int kW, const int sD,
                   const int sH, const int sW, const int pD, const int pH, const int pW, const int isNCHW,
                   const dnnl::algorithm mode) {
  // unfortunately mkl dnn doesn't support any format (dnnl::memory::format_tag::any) for input
  const int rank = input->rankOf();

  int bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;
  dnnl::memory::dims strides, kernel, padding, padding_r, xDims, zDims;
  dnnl::memory::format_tag xzFrmat;

  const auto type = dnnl::memory::data_type::f32;

  if (rank == 4) {  // 2d

    ops::ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                                    indIiH, indWiC, indWoC, indWkH, indOoH);

    strides = {sH, sW};
    kernel = {kH, kW};
    padding = {pH, pW};
    padding_r = {(oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW};
    xDims = {bS, iC, iH, iW};
    zDims = {bS, oC, oH, oW};

    xzFrmat = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
  } else {  // 3d

    ops::ConvolutionUtils::getSizesAndIndexesConv3d(isNCHW, 0, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                                    indIOioC, indIiH, indWiC, indWoC, indWkH);

    strides = {sD, sH, sW};
    kernel = {kD, kH, kW};
    padding = {pD, pH, pW};
    padding_r = {(oD - 1) * sD - iD + kD - pD, (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW};
    xDims = {bS, iC, iD, iH, iW};
    zDims = {bS, oC, oD, oH, oW};

    xzFrmat = isNCHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
  }

  std::vector<int> permut;
  if (!isNCHW) permut = rank == 4 ? std::vector<int>({0, 3, 1, 2}) : std::vector<int>({0, 4, 1, 2, 3});

  // memory descriptors for arrays

  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, type, xzFrmat);
  dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, type, xzFrmat);
  onednnUtils::setBlockStrides(*input, x_user_md, permut);

  // output
  dnnl::memory::desc z_mkl_md = dnnl::memory::desc(zDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc z_user_md = dnnl::memory::desc(zDims, type, xzFrmat);
  onednnUtils::setBlockStrides(*output, z_user_md, permut);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());

  // operation primitive description
  dnnl::pooling_forward::desc op_desc(dnnl::prop_kind::forward_inference, mode, x_mkl_md, z_mkl_md, strides, kernel,
                                      padding, padding_r);
  dnnl::pooling_forward::primitive_desc op_prim_desc(op_desc, engine);

  // arguments (memory buffers) necessary for calculations
  std::unordered_map<int, dnnl::memory> args;

  dnnl::stream stream(engine);

  // provide memory buffers and check whether reorder is required

  // input
  onednnUtils::loadDataToMklStream(*input, engine, stream, x_user_md, op_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

  // output
  auto z_user_mem =
      onednnUtils::loadDataToMklStream(*output, engine, stream, z_user_md, op_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

  // run calculations
  dnnl::pooling_forward(op_prim_desc).execute(stream, args);

  // reorder outputs if necessary
  if (op_prim_desc.dst_desc() != z_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////
void poolingBpONEDNN(const NDArray* input, const NDArray* gradO, NDArray* gradI, const int kD, const int kH,
                     const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW,
                     const int isNCHW, const dnnl::algorithm mode) {
  // unfortunately mkl dnn doesn't support any format (dnnl::memory::format_tag::any) for input

  const int rank = input->rankOf();

  int bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;
  dnnl::memory::dims strides, kernel, padding, padding_r, xDims, zDims;
  dnnl::memory::format_tag xzFrmat;

  const auto type = dnnl::memory::data_type::f32;

  if (rank == 4) {  // 2d

    ops::ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC,
                                                    indIiH, indWiC, indWoC, indWkH, indOoH);

    strides = {sH, sW};
    kernel = {kH, kW};
    padding = {pH, pW};
    padding_r = {(oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW};
    xDims = {bS, iC, iH, iW};
    zDims = {bS, oC, oH, oW};

    xzFrmat = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
  } else {  // 3d

    ops::ConvolutionUtils::getSizesAndIndexesConv3d(isNCHW, 0, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW,
                                                    indIOioC, indIiH, indWiC, indWoC, indWkH);

    strides = {sD, sH, sW};
    kernel = {kD, kH, kW};
    padding = {pD, pH, pW};
    padding_r = {(oD - 1) * sD - iD + kD - pD, (oH - 1) * sH - iH + kH - pH, (oW - 1) * sW - iW + kW - pW};
    xDims = {bS, iC, iD, iH, iW};
    zDims = {bS, oC, oD, oH, oW};

    xzFrmat = isNCHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
  }

  std::vector<int> permut;
  if (!isNCHW) permut = rank == 4 ? std::vector<int>({0, 3, 1, 2}) : std::vector<int>({0, 4, 1, 2, 3});

  // memory descriptors for arrays

  // input
  dnnl::memory::desc x_mkl_md = dnnl::memory::desc(xDims, type, xzFrmat);
  dnnl::memory::desc x_user_md = dnnl::memory::desc(xDims, type, xzFrmat);
  onednnUtils::setBlockStrides(*input, x_user_md, permut);

  // gradO
  dnnl::memory::desc gradO_mkl_md = dnnl::memory::desc(zDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradO_user_md = dnnl::memory::desc(zDims, type, xzFrmat);
  onednnUtils::setBlockStrides(*gradO, gradO_user_md, permut);

  // gradI
  dnnl::memory::desc gradI_mkl_md = dnnl::memory::desc(xDims, type, dnnl::memory::format_tag::any);
  dnnl::memory::desc gradI_user_md = dnnl::memory::desc(xDims, type, xzFrmat);
  onednnUtils::setBlockStrides(*gradI, gradI_user_md, permut);

  auto engine = onednnUtils::getEngine(LaunchContext::defaultContext()->engine());
  dnnl::stream stream(engine);

  // forward primitive description
  dnnl::pooling_forward::desc op_ff_desc(dnnl::prop_kind::forward, mode, x_mkl_md, gradO_mkl_md, strides, kernel,
                                         padding, padding_r);
  dnnl::pooling_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

  // backward primitive description
  dnnl::pooling_backward::desc op_bp_desc(mode, gradI_mkl_md, gradO_mkl_md, strides, kernel, padding, padding_r);
  dnnl::pooling_backward::primitive_desc op_bp_prim_desc(op_bp_desc, engine, op_ff_prim_desc);

  // arguments (memory buffers) necessary for calculations
  std::unordered_map<int, dnnl::memory> args;

  // gradO
  onednnUtils::loadDataToMklStream(*gradO, engine, stream, gradO_user_md, op_bp_prim_desc.diff_dst_desc(),
                                   args[DNNL_ARG_DIFF_DST]);

  // gradI
  auto gradI_user_mem = onednnUtils::loadDataToMklStream(*gradI, engine, stream, gradI_user_md,
                                                         op_bp_prim_desc.diff_src_desc(), args[DNNL_ARG_DIFF_SRC]);

  if (mode == algorithm::pooling_max) {
    // input
    onednnUtils::loadDataToMklStream(*input, engine, stream, x_user_md, op_ff_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

    // z
    auto z_mkl_mem = dnnl::memory(op_ff_prim_desc.dst_desc(), engine);
    args[DNNL_ARG_DST] = z_mkl_mem;

    // auxiliary memory allocation
    auto workspace = dnnl::memory(op_ff_prim_desc.workspace_desc(), engine);
    args[DNNL_ARG_WORKSPACE] = workspace;

    // run forward calculations
    dnnl::pooling_forward(op_ff_prim_desc).execute(stream, args);
  }

  // run backward calculations
  dnnl::pooling_backward(op_bp_prim_desc).execute(stream, args);

  // reorder gradI if necessary
  if (op_bp_prim_desc.diff_src_desc() != gradI_user_mem.get_desc())
    dnnl::reorder(args[DNNL_ARG_DIFF_SRC], gradI_user_mem).execute(stream, args[DNNL_ARG_DIFF_SRC], gradI_user_mem);

  stream.wait();
}

//////////////////////////////////////////////////////////////////////////
void getONEDNNMemoryDescLrn(const NDArray* src, const NDArray* diff_src, const NDArray* dst,
                            dnnl::memory::desc* lrn_src_md, dnnl::memory::desc* lrn_diff_src_md,
                            dnnl::memory::desc* lrn_dst_md, dnnl::memory::desc* user_src_md,
                            dnnl::memory::desc* user_diff_src_md, dnnl::memory::desc* user_dst_md, int axis) {
  const sd::LongType* shape = src->shapeInfo();
  long rank = shape[0];
  long dim1 = axis;  // MKL-DNN supports only 1 axis, which has to be the "channel" one
  long dim2 = axis >= 2 ? 1 : 2;
  long dim3 = axis >= 3 ? 2 : 3;
  dnnl::memory::dims lrn_src_tz = {(int)shape[1], (int)shape[dim1 + 1], rank > 2 ? (int)shape[dim2 + 1] : 1,
                                   rank > 3 ? (int)shape[dim3 + 1] : 1};

  auto type = dnnl::memory::data_type::f32;
  auto format = axis == 1 ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
  auto supposed_to_be_any_format = format;  // doesn't work with "any"

  if (src != nullptr && src->buffer() != nullptr && lrn_src_md != nullptr) {
    *lrn_src_md = dnnl::memory::desc({lrn_src_tz}, type, supposed_to_be_any_format);
    *user_src_md = dnnl::memory::desc({lrn_src_tz}, type, format);
    user_src_md->data.format_kind = dnnl_blocked;
    user_src_md->data.format_desc.blocking.strides[0] = src->stridesOf()[0];
    user_src_md->data.format_desc.blocking.strides[1] = src->stridesOf()[dim1];
    user_src_md->data.format_desc.blocking.strides[2] = rank > 2 ? src->stridesOf()[dim2] : 1;
    user_src_md->data.format_desc.blocking.strides[3] = rank > 3 ? src->stridesOf()[dim3] : 1;
  }

  if (diff_src != nullptr && diff_src->buffer() != nullptr && lrn_diff_src_md != nullptr) {
    *lrn_diff_src_md = dnnl::memory::desc({lrn_src_tz}, type, supposed_to_be_any_format);
    *user_diff_src_md = dnnl::memory::desc({lrn_src_tz}, type, format);
    user_diff_src_md->data.format_kind = dnnl_blocked;
    user_diff_src_md->data.format_desc.blocking.strides[0] = diff_src->stridesOf()[0];
    user_diff_src_md->data.format_desc.blocking.strides[1] = diff_src->stridesOf()[dim1];
    user_diff_src_md->data.format_desc.blocking.strides[2] = rank > 2 ? diff_src->stridesOf()[dim2] : 1;
    user_diff_src_md->data.format_desc.blocking.strides[3] = rank > 3 ? diff_src->stridesOf()[dim3] : 1;
  }

  if (dst != nullptr && dst->buffer() != nullptr && lrn_dst_md != nullptr) {
    *lrn_dst_md = dnnl::memory::desc({lrn_src_tz}, type, supposed_to_be_any_format);
    *user_dst_md = dnnl::memory::desc({lrn_src_tz}, type, format);
    user_dst_md->data.format_kind = dnnl_blocked;
    user_dst_md->data.format_desc.blocking.strides[0] = dst->stridesOf()[0];
    user_dst_md->data.format_desc.blocking.strides[1] = dst->stridesOf()[dim1];
    user_dst_md->data.format_desc.blocking.strides[2] = rank > 2 ? dst->stridesOf()[dim2] : 1;
    user_dst_md->data.format_desc.blocking.strides[3] = rank > 3 ? dst->stridesOf()[dim3] : 1;
  }
}

//////////////////////////////////////////////////////////////////////////
dnnl::engine& getEngine(void* ptr) {
  auto eng = reinterpret_cast<dnnl::engine*>(ptr);
  return *eng;
}

void checkPoolingONEDNN(Requirements& reqs, sd::graph::Context& block, const sd::NDArray& in, const sd::NDArray& out) {
  // replicate OneDNN check that was added since v1.8
  // https://github.com/oneapi-src/oneDNN/blob/master/src/common/pooling.cpp#L108-L110
  // if (str < 1 || dil < 0 || pad_l < 0 || pad_r + str < 0) return invalid_arguments;
  if (in.rankOf() > 4 && block.getIArguments()->size() > 12) {
    // pooling 3D
    int kD = INT_ARG(0);            // filter(kernel) depth
    int kH = INT_ARG(1);            // filter(kernel) height
    int kW = INT_ARG(2);            // filter(kernel) width
    int sD = INT_ARG(3);            // strides depth
    int sH = INT_ARG(4);            // strides height
    int sW = INT_ARG(5);            // strides width
    int pD = INT_ARG(6);            // paddings depth
    int pH = INT_ARG(7);            // paddings height
    int pW = INT_ARG(8);            // paddings width
    int dD = INT_ARG(9);            // dilations depth
    int dH = INT_ARG(10);           // dilations height
    int dW = INT_ARG(11);           // dilations width
    int paddingMode = INT_ARG(12);  // 1-SAME,  0-VALID
    // int extraParam0 = INT_ARG(13); // unnecessary for max case, required only for avg and pnorm cases
    int isNCDHW = block.getIArguments()->size() > 14 ? !INT_ARG(14) : 1;  // 1-NDHWC, 0-NCDHW
    reqs.expectEq(makeInfoVariable(in.rankOf(), RANK_MSG_INPUT0), 5) &&
        // stride >=1
        reqs.expectGreaterEq(makeInfoVariable(sD, "strides#Depth"), 1) &&
        reqs.expectGreaterEq(makeInfoVariable(sH, "strides#Height"), 1) &&
        reqs.expectGreaterEq(makeInfoVariable(sW, "strides#Width"), 1) &&
        // dilation >=0
        reqs.expectGreaterEq(makeInfoVariable(dW, "dilation#Depth"), 0) &&
        reqs.expectGreaterEq(makeInfoVariable(dH, "dilation#Height"), 0) &&
        reqs.expectGreaterEq(makeInfoVariable(dW, "dilation#Width"), 0);
    if (reqs) {
      int bS, iC, iD, iH, iW, oC, oD, oH,
          oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
      int indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
      ops::ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, in, out, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC,
                                                      indIOioD, indWiC, indWoC, indWkD);

      if (paddingMode)  // SAME
        ops::ConvolutionUtils::calcPadding3D(pD, pH, pW, oD, oH, oW, iD, iH, iW, kD, kH, kW, sD, sH, sW, dD, dH, dW);
      // pad_l >=0
      reqs.expectGreaterEq(makeInfoVariable(pD, "padding_l#Depth"), 0) &&
          reqs.expectGreaterEq(makeInfoVariable(pH, "padding_l#Height"), 0) &&
          reqs.expectGreaterEq(makeInfoVariable(pW, "padding_l#Width"), 0) &&
          // pad_r+ stride
          reqs.expectGreaterEq(makeInfoVariable(((oD - 1) * sD - iD + kD - pD) + sD, "padding_r#Depth + stride#Depth"),
                               0) &&
          reqs.expectGreaterEq(
              makeInfoVariable(((oH - 1) * sH - iH + kH - pH) + sH, "padding_r#Height + stride#Height"), 0) &&
          reqs.expectGreaterEq(makeInfoVariable(((oW - 1) * sW - iW + kW - pW) + sW, "padding_r#Width + stride#Width"),
                               0);
    }
  } else if (block.getIArguments()->size() > 8) {
    const int kH = INT_ARG(0);
    const int kW = INT_ARG(1);
    const int sH = INT_ARG(2);
    const int sW = INT_ARG(3);
    int pH = INT_ARG(4);
    int pW = INT_ARG(5);
    const int dH = INT_ARG(6);
    const int dW = INT_ARG(7);
    const int paddingMode = INT_ARG(8);
    const int isNCHW = block.getIArguments()->size() > 10 ? !INT_ARG(10) : 1;  // INT_ARG(10): 1-NHWC, 0-NCHW
    reqs.expectEq(makeInfoVariable(in.rankOf(), RANK_MSG_INPUT0), 4) &&
        // stride >=1
        reqs.expectGreaterEq(makeInfoVariable(sH, "strides#Height"), 1) &&
        reqs.expectGreaterEq(makeInfoVariable(sW, "strides#Width"), 1) &&
        // dilation >=0
        reqs.expectGreaterEq(makeInfoVariable(dH, "dilation#Height"), 0) &&
        reqs.expectGreaterEq(makeInfoVariable(dW, "dilation#Width"), 0);
    if (reqs) {
      int bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;
      ops::ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, in, out, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                                      indWiC, indWoC, indWkH, indOoH);
      if (paddingMode) {
        ops::ConvolutionUtils::calcPadding2D(pH, pW, oH, oW, iH, iW, kH, kW, sH, sW, dH, dW);
      }
      // pad_l >=0
      reqs.expectGreaterEq(makeInfoVariable(pH, "padding_l#Height"), 0) &&
          reqs.expectGreaterEq(makeInfoVariable(pW, "padding_l#Width"), 0) &&
          // pad_r+ stride
          reqs.expectGreaterEq(
              makeInfoVariable(((oH - 1) * sH - iH + kH - pH) + sH, "padding_r#Height + stride#Height"), 0) &&
          reqs.expectGreaterEq(makeInfoVariable(((oW - 1) * sW - iW + kW - pW) + sW, "padding_r#Width + stride#Width"),
                               0);
    }
  }
  return;
}


}  // namespace onednnUtils
}  // namespace sd
