/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
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
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <platform_boilerplate.h>

#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>
#include <NDArrayFactory.h>


namespace nd4j      {
namespace ops       {
namespace platforms {


//////////////////////////////////////////////////////////////////////////
static void batchnormMKLDNN(const NDArray* x, const NDArray* mean, const NDArray* variance, const NDArray* weights, const float epsilon, NDArray* z) {

    // unfortunately mkl dnn doesn't support any format (mkldnn::memory::format_tag::any)
    // also it gives wrong results for formats nhwc and ndhwc

    // x -> 2D:nc, 4D:nchw, 5D:ncdhw
    // mean -> 1D [c]
    // variance -> 1D [c]
    // weights 2D [2, c], weights({0,1, 0,0}) contains gamma and weights({1,2, 0,0}) contains beta
    // z(output) - same shape as x

    const int xRank = x->rankOf();

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // input type
    mkldnn::memory::data_type type = mkldnn::memory::data_type::f32;

    // indicate whether gamma or/and beta are given
    auto flags = mkldnn::normalization_flags::use_global_stats;
    if (weights != nullptr)
        flags |= mkldnn::normalization_flags::use_scale_shift;

    mkldnn::memory::dims dims;
    mkldnn::memory::format_tag format;

    if(xRank == 2) {
        dims = {x->sizeAt(0), x->sizeAt(1)};
        format = mkldnn::memory::format_tag::nc;
    }
    else if(xRank == 4) {
        dims = {x->sizeAt(0), x->sizeAt(1), x->sizeAt(2), x->sizeAt(3)};
        format = mkldnn::memory::format_tag::nchw;
    }
    else {  // xRank = 5
        dims = {x->sizeAt(0), x->sizeAt(1), x->sizeAt(2), x->sizeAt(3), x->sizeAt(4)};
        format = mkldnn::memory::format_tag::ncdhw;
    }

    // memory descriptors for arrays

    // x
    mkldnn::memory::desc x_mkl_md  = mkldnn::memory::desc(dims, type, format);
    mkldnn::memory::desc x_user_md = mkldnn::memory::desc(dims, type, format);
    x_user_md.data.format_kind = mkldnn_blocked;    // overrides format
    x_user_md.data.format_desc.blocking.strides[0] = x->stridesOf()[0];
    x_user_md.data.format_desc.blocking.strides[1] = x->stridesOf()[1];
    if(xRank > 2) {
        x_user_md.data.format_desc.blocking.strides[2] = x->stridesOf()[2];
        x_user_md.data.format_desc.blocking.strides[3] = x->stridesOf()[3];
    }
    if(xRank > 4)
        x_user_md.data.format_desc.blocking.strides[4] = x->stridesOf()[4];

    // z, output
    mkldnn::memory::desc z_mkl_md  = mkldnn::memory::desc(dims, type, format);
    mkldnn::memory::desc z_user_md = mkldnn::memory::desc(dims, type, format);
    z_user_md.data.format_kind = mkldnn_blocked;    // overrides format
    z_user_md.data.format_desc.blocking.strides[0] = z->stridesOf()[0];
    z_user_md.data.format_desc.blocking.strides[1] = z->stridesOf()[1];
    if(xRank > 2) {
        z_user_md.data.format_desc.blocking.strides[2] = z->stridesOf()[2];
        z_user_md.data.format_desc.blocking.strides[3] = z->stridesOf()[3];
    }
    if(xRank > 4)
        z_user_md.data.format_desc.blocking.strides[4] = z->stridesOf()[4];


    // batchnorm forward description
    mkldnn::batch_normalization_forward::desc op_ff_desc(mkldnn::prop_kind::forward_inference, x_mkl_md, epsilon, flags);
    mkldnn::batch_normalization_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

    // arguments (memory buffers) necessary for calculations
    std::unordered_map<int, mkldnn::memory> args;

    mkldnn::stream stream(engine);

    // provide memory and check whether reorder is required

    // x
    auto x_user_mem = mkldnn::memory(x_user_md, engine, x->getBuffer());
    const bool xReorder = op_ff_prim_desc.src_desc() != x_user_mem.get_desc();
    auto x_mkl_mem = xReorder ? mkldnn::memory(op_ff_prim_desc.src_desc(), engine) : x_user_mem;
    if (xReorder)
        mkldnn::reorder(x_user_mem, x_mkl_mem).execute(stream, x_user_mem, x_mkl_mem);
    args[MKLDNN_ARG_SRC] = x_mkl_mem;

    // z
    auto z_user_mem = mkldnn::memory(z_user_md, engine, z->getBuffer());
    const bool zReorder = op_ff_prim_desc.dst_desc() != z_user_mem.get_desc();
    auto z_mkl_mem = zReorder ? mkldnn::memory(op_ff_prim_desc.dst_desc(), engine) : z_user_mem;
    if (zReorder)
        mkldnn::reorder(z_user_mem, z_mkl_mem).execute(stream, z_user_mem, z_mkl_mem);
    args[MKLDNN_ARG_DST] = z_mkl_mem;

    // mean
    auto mean_mkl_mem = mkldnn::memory(op_ff_prim_desc.mean_desc(), engine, mean->getBuffer());
    args[MKLDNN_ARG_MEAN] = mean_mkl_mem;

    // variance
    auto var_mkl_mem = mkldnn::memory(op_ff_prim_desc.variance_desc(), engine, variance->getBuffer());
    args[MKLDNN_ARG_VARIANCE] = var_mkl_mem;

    // gamma and beta (and their gradients) if they are present
    if(weights != nullptr) {

        auto w_mkl_mem = mkldnn::memory(op_ff_prim_desc.weights_desc(), engine, weights->getBuffer());
        args[MKLDNN_ARG_WEIGHTS] = w_mkl_mem;
    }

    // run calculations
    mkldnn::batch_normalization_forward(op_ff_prim_desc).execute(stream, args);

    // reorder outputs if necessary
    if (zReorder)
        mkldnn::reorder(z_mkl_mem, z_user_mem).execute(stream, z_mkl_mem, z_user_mem);

    stream.wait();

    // shape::printArray(z_mkl_mem.map_data<float>(),8);
}


//////////////////////////////////////////////////////////////////////////
static void batchnormBackPropMKLDNN(const NDArray* x, const NDArray* mean, const NDArray* variance, const NDArray* dLdO, const NDArray* weights,
                                    const float epsilon, NDArray* dLdI, NDArray* dLdW) {

    // unfortunately mkl dnn doesn't support any format (mkldnn::memory::format_tag::any)
    // also it gives wrong results for formats nhwc and ndhwc

    // x -> 2D:nc, 4D:nchw, 5D:ncdhw
    // mean -> 1D [c]
    // variance -> 1D [c]
    // dLdO - same shape as x
    // weights 2D [2, c], weights({0,1, 0,0}) contains gamma and weights({1,2, 0,0}) contains beta
    // dLdI - same shape as x
    // dLdW - same shape as weights, dLdW({0,1, 0,0}) contains grad_gamma and dLdW({1,2, 0,0}) contains grad_beta

    const int xRank = x->rankOf();

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // input type
    mkldnn::memory::data_type type = mkldnn::memory::data_type::f32;

    // indicate whether gamma or/and beta are given
    auto flags = mkldnn::normalization_flags::use_global_stats;
    if (weights != nullptr)
        flags |= mkldnn::normalization_flags::use_scale_shift;

    mkldnn::memory::dims dims;
    mkldnn::memory::format_tag format;

    if(xRank == 2) {
        dims = {x->sizeAt(0), x->sizeAt(1)};
        format = mkldnn::memory::format_tag::nc;
    }
    else if(xRank == 4) {
        dims = {x->sizeAt(0), x->sizeAt(1), x->sizeAt(2), x->sizeAt(3)};
        format = mkldnn::memory::format_tag::nchw;
    }
    else {  // xRank = 5
        dims = {x->sizeAt(0), x->sizeAt(1), x->sizeAt(2), x->sizeAt(3), x->sizeAt(4)};
        format = mkldnn::memory::format_tag::ncdhw;
    }

    // memory descriptors for arrays

    // x
    mkldnn::memory::desc x_mkl_md  = mkldnn::memory::desc(dims, type, format);
    mkldnn::memory::desc x_user_md = mkldnn::memory::desc(dims, type, format);
    x_user_md.data.format_kind = mkldnn_blocked;    // overrides format
    x_user_md.data.format_desc.blocking.strides[0] = x->stridesOf()[0];
    x_user_md.data.format_desc.blocking.strides[1] = x->stridesOf()[1];
    if(xRank > 2) {
        x_user_md.data.format_desc.blocking.strides[2] = x->stridesOf()[2];
        x_user_md.data.format_desc.blocking.strides[3] = x->stridesOf()[3];
    }
    if(xRank > 4)
        x_user_md.data.format_desc.blocking.strides[4] = x->stridesOf()[4];

    // dLdO
    mkldnn::memory::desc dLdO_mkl_md  = mkldnn::memory::desc(dims, type, format);
    mkldnn::memory::desc dLdO_user_md = mkldnn::memory::desc(dims, type, format);
    dLdO_user_md.data.format_kind = mkldnn_blocked;    // overrides format
    dLdO_user_md.data.format_desc.blocking.strides[0] = dLdO->stridesOf()[0];
    dLdO_user_md.data.format_desc.blocking.strides[1] = dLdO->stridesOf()[1];
    if(xRank > 2) {
        dLdO_user_md.data.format_desc.blocking.strides[2] = dLdO->stridesOf()[2];
        dLdO_user_md.data.format_desc.blocking.strides[3] = dLdO->stridesOf()[3];
    }
    if(xRank > 4)
        dLdO_user_md.data.format_desc.blocking.strides[4] = dLdO->stridesOf()[4];

    // dLdI
    mkldnn::memory::desc dLdI_mkl_md  = mkldnn::memory::desc(dims, type, format);
    mkldnn::memory::desc dLdI_user_md = mkldnn::memory::desc(dims, type, format);
    dLdI_user_md.data.format_kind = mkldnn_blocked;    // overrides format
    dLdI_user_md.data.format_desc.blocking.strides[0] = dLdI->stridesOf()[0];
    dLdI_user_md.data.format_desc.blocking.strides[1] = dLdI->stridesOf()[1];
    if(xRank > 2) {
        dLdI_user_md.data.format_desc.blocking.strides[2] = dLdI->stridesOf()[2];
        dLdI_user_md.data.format_desc.blocking.strides[3] = dLdI->stridesOf()[3];
    }
    if(xRank > 4)
        dLdI_user_md.data.format_desc.blocking.strides[4] = dLdI->stridesOf()[4];

    // batchnorm forward description
    mkldnn::batch_normalization_forward::desc op_ff_desc(mkldnn::prop_kind::forward_inference, x_mkl_md, epsilon, flags);
    mkldnn::batch_normalization_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

    // batchnorm backprop description
    mkldnn::batch_normalization_backward::desc op_bp_desc(mkldnn::prop_kind::backward, dLdO_mkl_md, x_mkl_md, epsilon, flags);
    mkldnn::batch_normalization_backward::primitive_desc op_bp_prim_desc(op_bp_desc, engine, op_ff_prim_desc);

    // arguments (memory buffers) necessary for calculations
    std::unordered_map<int, mkldnn::memory> args;

    mkldnn::stream stream(engine);

    // provide memory and check whether reorder is required

    // x
    auto x_user_mem = mkldnn::memory(x_user_md, engine, x->getBuffer());
    const bool xReorder = op_bp_prim_desc.src_desc() != x_user_mem.get_desc();
    auto x_mkl_mem = xReorder ? mkldnn::memory(op_bp_prim_desc.src_desc(), engine) : x_user_mem;
    if (xReorder)
        mkldnn::reorder(x_user_mem, x_mkl_mem).execute(stream, x_user_mem, x_mkl_mem);
    args[MKLDNN_ARG_SRC] = x_mkl_mem;

    // dLdO
    auto dLdO_user_mem = mkldnn::memory(dLdO_user_md, engine, dLdO->getBuffer());
    const bool dLdOReorder = op_bp_prim_desc.diff_src_desc() != dLdO_user_mem.get_desc();
    auto dLdO_mkl_mem = dLdOReorder ? mkldnn::memory(op_bp_prim_desc.diff_src_desc(), engine) : dLdO_user_mem;
    if (dLdOReorder)
        mkldnn::reorder(dLdO_user_mem, dLdO_mkl_mem).execute(stream, dLdO_user_mem, dLdO_mkl_mem);
    args[MKLDNN_ARG_DIFF_DST] = dLdO_mkl_mem;

    // mean
    auto mean_mkl_mem = mkldnn::memory(op_bp_prim_desc.mean_desc(), engine, mean->getBuffer());
    args[MKLDNN_ARG_MEAN] = mean_mkl_mem;

    // variance
    auto var_mkl_mem = mkldnn::memory(op_bp_prim_desc.variance_desc(), engine, variance->getBuffer());
    args[MKLDNN_ARG_VARIANCE] = var_mkl_mem;

    // dLdI
    auto dLdI_user_mem = mkldnn::memory(dLdI_user_md, engine, dLdI->getBuffer());
    const bool dLdIReorder = op_bp_prim_desc.diff_dst_desc() != dLdI_user_mem.get_desc();
    auto dLdI_mkl_mem = dLdIReorder ? mkldnn::memory(op_bp_prim_desc.diff_dst_desc(), engine) : dLdI_user_mem;
    args[MKLDNN_ARG_DIFF_SRC] = dLdI_mkl_mem;

    // gamma and beta (and their gradients) if they are present
    if(weights != nullptr) {

        auto w_mkl_mem = mkldnn::memory(op_bp_prim_desc.weights_desc(), engine, weights->getBuffer());
        args[MKLDNN_ARG_WEIGHTS] = w_mkl_mem;

        auto dLdW_mkl_mem = mkldnn::memory(op_bp_prim_desc.weights_desc(), engine, dLdW->getBuffer());
        args[MKLDNN_ARG_DIFF_WEIGHTS] = dLdW_mkl_mem;
    }

    // run calculations
    mkldnn::batch_normalization_backward(op_bp_prim_desc).execute(stream, args);

    // reorder outputs if necessary
    if (dLdIReorder)
        mkldnn::reorder(dLdI_mkl_mem, dLdI_user_mem).execute(stream, dLdI_mkl_mem, dLdI_user_mem);

    stream.wait();

    // shape::printArray(dLdI_mkl_mem.map_data<float>(),8);
}

PLATFORM_IMPL(batchnorm) {

    auto input    = INPUT_VARIABLE(0);  // 2D:nc, 4D:nchw, 5D:ncdhw
    auto mean     = INPUT_VARIABLE(1);  // [c]
    auto variance = INPUT_VARIABLE(2);  // [c]
    NDArray* gamma    = nullptr;        // [c]
    NDArray* beta     = nullptr;        // [c]

    auto output = OUTPUT_VARIABLE(0);   // same shape as input

    const bool   applyScale  = (bool)INT_ARG(0);
    const bool   applyOffset = (bool)INT_ARG(1);
    const double epsilon     = T_ARG(0);

    if(applyScale)
        gamma = INPUT_VARIABLE(3);
    if(applyOffset)
        beta = INPUT_VARIABLE(3 + (int)applyScale);

    const int numOfIntArgs = block.getIArguments()->size();
    const int inRank = input->rankOf();

    // get axes args to normalize input array over
    std::vector<int> axes;
    if(numOfIntArgs > 2)
        for(int i = 2; i < numOfIntArgs; ++i)
            axes.push_back(INT_ARG(i));
    else
        axes.push_back(inRank-1);               // default dimension to reduce along is last dimension

    const int numOfAxes = axes.size();
    REQUIRE_TRUE(numOfAxes == 1, 0, "BATCHNORM_MKLDNN op: mkl dnn library supports only one axis which represents channel dimension, but got %i axes instead!", numOfAxes);
    REQUIRE_TRUE(inRank == 2 || inRank == 4 || inRank == 5, 0, "BATCHNORM_MKLDNN op: possible values for rank of input array are 2, 4 or 5, but got %i instead!", inRank);
    REQUIRE_TRUE(mean->rankOf() == 1 && mean->sizeAt(0) == input->sizeAt(axes[0]), 0, "BATCHNORM_MKLDNN op: wrong shape of mean array, expected is [%lld], but got %s instead !", input->sizeAt(axes[0]), ShapeUtils::shapeAsString(mean).c_str());
    REQUIRE_TRUE(variance->rankOf() == 1 && variance->sizeAt(0) == input->sizeAt(axes[0]), 0, "BATCHNORM_MKLDNN op: wrong shape of variance array, expected is [%lld], but got %s instead !", input->sizeAt(axes[0]), ShapeUtils::shapeAsString(variance).c_str());
    if(gamma != nullptr)
        REQUIRE_TRUE(gamma->rankOf() == 1 && gamma->sizeAt(0) == input->sizeAt(axes[0]), 0, "BATCHNORM_MKLDNN op: wrong shape of gamma array, expected is [%lld], but got %s instead !", input->sizeAt(axes[0]), ShapeUtils::shapeAsString(gamma).c_str());
    if(beta != nullptr)
        REQUIRE_TRUE(beta->rankOf() == 1 && beta->sizeAt(0) == input->sizeAt(axes[0]), 0, "BATCHNORM_MKLDNN op: wrong shape of beta array, expected is [%lld], but got %s instead !", input->sizeAt(axes[0]), ShapeUtils::shapeAsString(beta).c_str());

    // types of all input arrays should be the same (except dLdO)
    for(int i = 1; i < block.width() - 1; ++i)
        REQUIRE_TRUE(INPUT_VARIABLE(0)->dataType() == INPUT_VARIABLE(i)->dataType(), 0, "BATCHNORM_MKLDNN op: types of all input arrays should be the same !");


    NDArray *weights = nullptr;

    if(applyScale || applyOffset) {

        weights = new NDArray(input->ordering(), {2, input->sizeAt(axes[0])}, input->dataType());

        if(applyScale)
            (*weights)({0,1, 0,0}).assign(gamma);
        else
            (*weights)({0,1, 0,0}).assign(1);
        if(applyOffset)
            (*weights)({1,2, 0,0}).assign(beta);
        else
            (*weights)({1,2, 0,0}).assign(0);
    }

    batchnormMKLDNN(input, mean, variance, weights, epsilon, output);

    delete weights;

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(batchnorm) {
    // we don't want to use mkldnn if cpu doesn't support avx/avx2
    // if (::optimalLevel() < 2)
    //     return false;

    auto input    = INPUT_VARIABLE(0);  // 2D:nc, 4D:nchw, 5D:ncdhw
    auto mean     = INPUT_VARIABLE(1);  // [c]
    auto variance = INPUT_VARIABLE(2);  // [c]
    NDArray* gamma    = nullptr;        // [c]
    NDArray* beta     = nullptr;        // [c]

    auto output = OUTPUT_VARIABLE(0);   // same shape as input

    const bool   applyScale  = (bool)INT_ARG(0);
    const bool   applyOffset = (bool)INT_ARG(1);

    if(applyScale)
        gamma = INPUT_VARIABLE(3);
    if(applyOffset)
        beta = INPUT_VARIABLE(3 + (int)applyScale);


    const int numOfIntArgs = block.getIArguments()->size();
    std::vector<int> axes;
    if(numOfIntArgs > 2)
        for(int i = 2; i < numOfIntArgs; ++i)
            axes.push_back(INT_ARG(i));
    else
        axes.push_back(input->rankOf()-1);               // default dimension to reduce along is last dimension

    DataType inputType = input->dataType();
    DataType meanType  = mean->dataType();
    DataType varType   = variance->dataType();
    DataType gammaType = gamma != nullptr ? gamma->dataType() : DataType::FLOAT32;
    DataType betaType  = beta  != nullptr ? beta->dataType()  : DataType::FLOAT32;
    DataType outType   = output->dataType();

    const int inRank = input->rankOf();

    return block.isUseMKLDNN() && axes.size() == 1 && axes[0] == 1 && (inRank == 2 || inRank == 4 || inRank == 5) &&
            (inputType == DataType::FLOAT32 && meanType == DataType::FLOAT32 && varType == DataType::FLOAT32 &&
             gammaType == DataType::FLOAT32 && betaType == DataType::FLOAT32 && outType == DataType::FLOAT32);
}

//////////////////////////////////////////////////////////////////////////
// PLATFORM_IMPL(batchnorm) {

//     auto input = INPUT_VARIABLE(0);
//     auto mean = INPUT_VARIABLE(1);
//     auto variance = INPUT_VARIABLE(2);
//     NDArray *gamma = nullptr;
//     NDArray *beta = nullptr;

//     auto output = OUTPUT_VARIABLE(0);

//     const bool applyScale = (bool) INT_ARG(0);
//     const bool applyOffset = (bool) INT_ARG(1);
//     const double epsilon = T_ARG(0);

//     if (applyScale)
//         gamma = INPUT_VARIABLE(3);
//     if (applyOffset)
//         beta = INPUT_VARIABLE(3 + static_cast<int>(applyScale));

//     std::vector<int> axes;
//     if (block.numI() > 2)
//         for (int i = 2; i < block.numI(); ++i)
//             axes.push_back(INT_ARG(i));
//     else
//         axes.push_back(input->rankOf() - 1);

//     std::vector<Nd4jLong> shape({2, mean->lengthOf()});
//     NDArray weights = NDArrayFactory::create<float>('c', shape, block.launchContext());
//     weights({0, 1, 0, 0}).assign(1.0f);
//     weights({1, 2, 0, 0}).assign(0.0f);

//     mkldnn_memory_desc_t empty;
//     mkldnn::memory::desc batchnorm_src_md(empty), batchnorm_dst_md(empty), user_src_md(empty), user_dst_md(empty);

//     auto flag = mkldnn::normalization_flags::use_global_stats;
//     if (applyScale || applyOffset)
//         flag |= mkldnn::normalization_flags::use_scale_shift;

//     mkldnnUtils::getMKLDNNMemoryDescBatchNorm(input, nullptr, output,
//                                               &batchnorm_src_md, nullptr, &batchnorm_dst_md,
//                                               &user_src_md, nullptr, &user_dst_md, axes[0]);

//     auto batchnorm_desc = mkldnn::batch_normalization_forward::desc(mkldnn::prop_kind::forward_inference, batchnorm_src_md, epsilon, flag);

//     auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
//     mkldnn::stream stream(engine);
//     auto batchnorm_prim_desc = mkldnn::batch_normalization_forward::primitive_desc(batchnorm_desc, engine);
//     auto user_src_memory = mkldnn::memory(user_src_md, engine, input->buffer());
//     auto user_dst_memory = mkldnn::memory(user_dst_md, engine, output->buffer());
//     auto batchnorm_mean_memory = mkldnn::memory(batchnorm_prim_desc.mean_desc(), engine,
//                                                 mean->buffer());
//     auto batchnorm_variance_memory = mkldnn::memory(batchnorm_prim_desc.variance_desc(), engine,
//                                                     variance->buffer());
//     auto batchnorm_src_memory = user_src_memory;
//     mkldnn::memory m(batchnorm_src_md, engine);
//     if (m.get_desc() != user_src_memory.get_desc()) {
//         batchnorm_src_memory = mkldnn::memory(batchnorm_src_md, engine);
//         mkldnn::reorder(user_src_memory, batchnorm_src_memory).execute(stream, user_src_memory,
//                                                                batchnorm_src_memory);
//     }
//     auto batchnorm_dst_memory = user_dst_memory;
//     if (batchnorm_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
//         batchnorm_dst_memory = mkldnn::memory(batchnorm_prim_desc.dst_desc(), engine);
//     }
//     if (applyScale || applyOffset) {
//         if (gamma != nullptr) {
//             weights({0, 1, 0, 0}).assign(gamma);
//         }
//         if (beta != nullptr) {
//             weights({1, 2, 0, 0}).assign(beta);
//         }

//         auto batchnorm_weights_memory = mkldnn::memory(batchnorm_prim_desc.weights_desc(), engine, weights.buffer());
//         mkldnn::batch_normalization_forward(batchnorm_prim_desc).execute(stream,
//                                                                  {{MKLDNN_ARG_SRC,      batchnorm_src_memory},
//                                                                   {MKLDNN_ARG_MEAN,     batchnorm_mean_memory},
//                                                                   {MKLDNN_ARG_VARIANCE, batchnorm_variance_memory},
//                                                                   {MKLDNN_ARG_WEIGHTS,  batchnorm_weights_memory},
//                                                                   {MKLDNN_ARG_DST,      batchnorm_dst_memory}});
//     } else {
//         mkldnn::batch_normalization_forward(batchnorm_prim_desc).execute(stream,
//                                                                  {{MKLDNN_ARG_SRC,      batchnorm_src_memory},
//                                                                   {MKLDNN_ARG_MEAN,     batchnorm_mean_memory},
//                                                                   {MKLDNN_ARG_VARIANCE, batchnorm_variance_memory},
//                                                                   {MKLDNN_ARG_DST,      batchnorm_dst_memory}});
//     }
//     if (batchnorm_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
//         mkldnn::reorder(batchnorm_dst_memory, user_dst_memory).execute(stream, batchnorm_dst_memory,
//                                                                user_dst_memory);
//     }
//     stream.wait();

//     return Status::OK();
// }

//////////////////////////////////////////////////////////////////////////
// PLATFORM_CHECK(batchnorm) {
//     // we don't want to use mkldnn if cpu doesn't support avx/avx2
//     if (::optimalLevel() < 2)
//         return false;

//     auto input = INPUT_VARIABLE(0);
//     auto mean = INPUT_VARIABLE(1);
//     auto variance = INPUT_VARIABLE(2);
//     NDArray *gamma = nullptr;
//     NDArray *beta = nullptr;

//     auto output = OUTPUT_VARIABLE(0);

//     const bool applyScale = (bool) INT_ARG(0);
//     const bool applyOffset = (bool) INT_ARG(1);
//     const double epsilon = T_ARG(0);

//     if (applyScale)
//         gamma = INPUT_VARIABLE(3);
//     if (applyOffset)
//         beta = INPUT_VARIABLE(3 + static_cast<int>(applyScale));

//     std::vector<int> axes;
//     if (block.numI() > 2)
//         for (int i = 2; i < block.numI(); ++i)
//             axes.push_back(INT_ARG(i));
//     else
//         axes.push_back(input->rankOf() - 1);

//     return block.isUseMKLDNN() &&
//            nd4j::MKLDNNStream::isSupported({input, mean, variance, gamma, beta, output}) &&
//            axes.size() == 1;
// }


//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batchnorm_bp) {

    NDArray* input    = INPUT_VARIABLE(0);      // 2D:nc, 4D:nchw, 5D:ncdhw
    NDArray* mean     = INPUT_VARIABLE(1);      // [c]
    NDArray* variance = INPUT_VARIABLE(2);      // [c]
    NDArray* dLdO     = INPUT_VARIABLE(3);      // same as input
    NDArray* gamma    = nullptr;                // [c]
    NDArray* beta     = nullptr;                // [c]

    NDArray* dLdI = OUTPUT_VARIABLE(0);         // same as input
    NDArray* dLdM = OUTPUT_VARIABLE(1);         // [c]
    NDArray* dLdV = OUTPUT_VARIABLE(2);         // [c]
    NDArray* dLdG = nullptr;                    // [c]
    NDArray* dLdB = nullptr;                    // [c]

    const bool  applyScale  = (bool)INT_ARG(0);
    const bool  applyOffset = (bool)INT_ARG(1);
    const float epsilon     = T_ARG(0);

    if(applyScale) {
        gamma = INPUT_VARIABLE(4);
        dLdG  = OUTPUT_VARIABLE(3);
    }
    if(applyOffset) {
        beta = INPUT_VARIABLE(4 + (int)applyScale);
        dLdB = OUTPUT_VARIABLE(3 + (int)applyScale);
    }

    const int numOfIntArgs = block.getIArguments()->size();
    const int inRank = input->rankOf();

    // get axes args to normalize input array over
    std::vector<int> axes;
    if(numOfIntArgs > 2)
        for(int i = 2; i < numOfIntArgs; ++i)
            axes.push_back(INT_ARG(i));
    else
        axes.push_back(inRank-1);               // default dimension to reduce along is last dimension

    const int numOfAxes = axes.size();
    REQUIRE_TRUE(numOfAxes == 1, 0, "BATCHNORM_BP_MKLDNN op: mkl dnn library supports only one axis which represents channel dimension, but got %i axes instead!", numOfAxes);
    REQUIRE_TRUE(inRank == 2 || inRank == 4 || inRank == 5, 0, "BATCHNORM_BP_MKLDNN op: possible values for rank of input array are 2, 4 or 5, but got %i instead!", inRank);
    REQUIRE_TRUE(input->isSameShape(dLdO), 0, "BATCHNORM_BP_MKLDNN op: wrong shape of gradients array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(input).c_str(), ShapeUtils::shapeAsString(dLdO).c_str());
    REQUIRE_TRUE(mean->rankOf() == 1 && mean->sizeAt(0) == input->sizeAt(axes[0]), 0, "BATCHNORM_BP_MKLDNN op: wrong shape of mean array, expected is [%lld], but got %s instead !", input->sizeAt(axes[0]), ShapeUtils::shapeAsString(mean).c_str());
    REQUIRE_TRUE(variance->rankOf() == 1 && variance->sizeAt(0) == input->sizeAt(axes[0]), 0, "BATCHNORM_BP_MKLDNN op: wrong shape of variance array, expected is [%lld], but got %s instead !", input->sizeAt(axes[0]), ShapeUtils::shapeAsString(variance).c_str());
    if(gamma != nullptr)
        REQUIRE_TRUE(gamma->rankOf() == 1 && gamma->sizeAt(0) == input->sizeAt(axes[0]), 0, "BATCHNORM_BP_MKLDNN op: wrong shape of gamma array, expected is [%lld], but got %s instead !", input->sizeAt(axes[0]), ShapeUtils::shapeAsString(gamma).c_str());
    if(beta != nullptr)
        REQUIRE_TRUE(beta->rankOf() == 1 && beta->sizeAt(0) == input->sizeAt(axes[0]), 0, "BATCHNORM_BP_MKLDNN op: wrong shape of beta array, expected is [%lld], but got %s instead !", input->sizeAt(axes[0]), ShapeUtils::shapeAsString(beta).c_str());

    // types of all input arrays should be the same (except dLdO)
    for(int i = 1; i < block.width() - 1; ++i)
        REQUIRE_TRUE(INPUT_VARIABLE(0)->dataType() == INPUT_VARIABLE(i)->dataType(), 0, "BATCHNORM_BP_MKLDNN op: types of all input arrays should be the same !");


    NDArray *weights = nullptr, *dLdW = nullptr;

    if(applyScale || applyOffset) {
        weights = new NDArray(input->ordering(), {2, input->sizeAt(axes[0])}, input->dataType());
        dLdW    = new NDArray(input->ordering(), {2, input->sizeAt(axes[0])}, input->dataType());
        if(applyScale)
            (*weights)({0,1, 0,0}).assign(gamma);
        else
            (*weights)({0,1, 0,0}).assign(1);
        if(applyOffset)
            (*weights)({1,2, 0,0}).assign(beta);
        else
            (*weights)({1,2, 0,0}).assign(0);
    }

    *dLdM = 0;
    *dLdV = 0;

    batchnormBackPropMKLDNN(input, mean, variance, dLdO, weights, epsilon, dLdI, dLdW);

    if(applyScale || applyOffset) {
        if(applyScale)
            dLdG->assign((*dLdW)({0,1, 0,0}));
        if(applyOffset)
            dLdB->assign((*dLdW)({1,2, 0,0}));

        delete weights;
        delete dLdW;
    }

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(batchnorm_bp) {
    // we don't want to use mkldnn if cpu doesn't support avx/avx2
    // if (::optimalLevel() < 2)
    //     return false;

    NDArray* input    = INPUT_VARIABLE(0);      // 2D:nc, 4D:nchw, 5D:ncdhw
    NDArray* mean     = INPUT_VARIABLE(1);      // [c]
    NDArray* variance = INPUT_VARIABLE(2);      // [c]
    NDArray* dLdO     = INPUT_VARIABLE(3);      // same as input
    NDArray* gamma    = nullptr;                // [c]
    NDArray* beta     = nullptr;                // [c]

    NDArray* dLdI = OUTPUT_VARIABLE(0);         // same as input
    NDArray* dLdM = OUTPUT_VARIABLE(1);         // [c]
    NDArray* dLdV = OUTPUT_VARIABLE(2);         // [c]
    NDArray* dLdG = nullptr;                    // [c]
    NDArray* dLdB = nullptr;                    // [c]

    const bool  applyScale  = (bool)INT_ARG(0);
    const bool  applyOffset = (bool)INT_ARG(1);

    if(applyScale) {
        gamma = INPUT_VARIABLE(4);
        dLdG  = OUTPUT_VARIABLE(3);
    }
    if(applyOffset) {
        beta = INPUT_VARIABLE(4 + (int)applyScale);
        dLdB = OUTPUT_VARIABLE(3 + (int)applyScale);
    }

    const int numOfIntArgs = block.getIArguments()->size();
    std::vector<int> axes;
    if(numOfIntArgs > 2)
        for(int i = 2; i < numOfIntArgs; ++i)
            axes.push_back(INT_ARG(i));
    else
        axes.push_back(input->rankOf()-1);               // default dimension to reduce along is last dimension

    DataType inputType = input->dataType();
    DataType meanType  = mean->dataType();
    DataType varType   = variance->dataType();
    DataType dLdOType  = dLdO->dataType();
    DataType gammaType = gamma != nullptr ? gamma->dataType() : DataType::FLOAT32;
    DataType betaType  = beta  != nullptr ? beta->dataType()  : DataType::FLOAT32;

    DataType dLdIType = dLdI->dataType();
    DataType dLdGType = gamma != nullptr ? dLdG->dataType() : DataType::FLOAT32;
    DataType dLdBType = beta  != nullptr ? dLdB->dataType() : DataType::FLOAT32;

    const int inRank = input->rankOf();

    return block.isUseMKLDNN() && axes.size() == 1 && axes[0] == 1 && (inRank == 2 || inRank == 4 || inRank == 5) &&
            (inputType == DataType::FLOAT32 && meanType  == DataType::FLOAT32 && varType  == DataType::FLOAT32 &&
             dLdOType  == DataType::FLOAT32 && gammaType == DataType::FLOAT32 && betaType == DataType::FLOAT32 &&
             dLdIType  == DataType::FLOAT32 && dLdGType  == DataType::FLOAT32 && dLdBType == DataType::FLOAT32);
}

}
}
}