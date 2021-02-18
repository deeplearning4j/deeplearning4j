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
// @author saudet
// @author raver119@gmail.com
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/OpRegistrator.h>
#include <system/platform_boilerplate.h>

#include <helpers/MKLDNNStream.h>
#include "mkldnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>
#include <array/NDArrayFactory.h>


namespace sd      {
namespace ops       {
namespace platforms {


//////////////////////////////////////////////////////////////////////////
static void batchnormMKLDNN(const NDArray* x, const NDArray* mean, const NDArray* variance, const NDArray* weights, NDArray* z,
                            const float epsilon, const bool isNCHW) {

    // unfortunately mkl dnn doesn't support any format (dnnl::memory::format_tag::any) for x

    // x -> 2D:nc, 4D:nchw/nhwc, 5D:ncdhw/ndhwc
    // mean -> 1D [c]
    // variance -> 1D [c]
    // weights 2D [2, c], weights({0,1, 0,0}) contains gamma and weights({1,2, 0,0}) contains beta
    // z(output) - same shape as x

    const int xRank = x->rankOf();

    // input type
    dnnl::memory::data_type type = dnnl::memory::data_type::f32;

    // indicate whether gamma or/and beta are given
    auto flags = dnnl::normalization_flags::use_global_stats;         // don't calculate the mean and variance for each mini-batch
    if (weights != nullptr)
        flags |= dnnl::normalization_flags::use_scale_shift;

    dnnl::memory::dims dims;
    dnnl::memory::format_tag format;

    const int indHW = isNCHW ? 2 : 1;
    const int bS = x->sizeAt(0);
    const int iC = isNCHW ? x->sizeAt(1) : x->sizeAt(-1);

    int iD, iH, iW;

    if(xRank == 2) {
        dims = {bS, iC};
        format = dnnl::memory::format_tag::nc;
    }
    else if(xRank == 4) {
        iH = x->sizeAt(indHW);
        iW = x->sizeAt(indHW + 1);
        dims = {bS, iC, iH, iW};
        format = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
    }
    else {  // xRank = 5
        iD =  x->sizeAt(indHW);
        iH =  x->sizeAt(indHW + 1);
        iW =  x->sizeAt(indHW + 2);
        dims = {bS, iC, iD, iH, iW};
        format = isNCHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
    }

    // memory descriptors for arrays

    // x
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(dims, type, format);
    dnnl::memory::desc x_user_md = dnnl::memory::desc(dims, type, format);
    mkldnnUtils::setBlockStrides(*x, x_user_md);

    // z, output
    dnnl::memory::desc z_mkl_md  = dnnl::memory::desc(dims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc z_user_md = dnnl::memory::desc(dims, type, format);
    mkldnnUtils::setBlockStrides(*z, z_user_md);

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // batchnorm forward description
    dnnl::batch_normalization_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, x_mkl_md, epsilon, flags);
    dnnl::batch_normalization_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

    // arguments (memory buffers) necessary for calculations
    std::unordered_map<int, dnnl::memory> args;

    dnnl::stream stream(engine);

    // provide memory and check whether reorder is required

    // x
    mkldnnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_ff_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

    // z
    auto z_user_mem = mkldnnUtils::loadDataToMklStream(*z, engine, stream, z_user_md, op_ff_prim_desc.dst_desc(), args[DNNL_ARG_DST]);

    // mean
    auto mean_mkl_mem = dnnl::memory(op_ff_prim_desc.mean_desc(), engine, const_cast<void*>(mean->buffer()));
    args[DNNL_ARG_MEAN] = mean_mkl_mem;

    // variance
    auto var_mkl_mem = dnnl::memory(op_ff_prim_desc.variance_desc(), engine, const_cast<void*>(variance->buffer()));
    args[DNNL_ARG_VARIANCE] = var_mkl_mem;

    // gamma and beta (and their gradients) if they are present
    if(weights != nullptr) {

        auto w_mkl_mem = dnnl::memory(op_ff_prim_desc.weights_desc(), engine, const_cast<void*>(weights->buffer()));
        args[DNNL_ARG_WEIGHTS] = w_mkl_mem;
    }

    // run calculations
    dnnl::batch_normalization_forward(op_ff_prim_desc).execute(stream, args);

    // reorder outputs if necessary
    if (op_ff_prim_desc.dst_desc() != z_user_mem.get_desc())
        dnnl::reorder(args[DNNL_ARG_DST], z_user_mem).execute(stream, args[DNNL_ARG_DST], z_user_mem);

    stream.wait();

    // shape::printArray(z_mkl_mem.map_data<float>(),8);
}


//////////////////////////////////////////////////////////////////////////
static void batchnormBpMKLDNN(const NDArray* x, const NDArray* mean, const NDArray* variance, const NDArray &dLdO, const NDArray* weights,
                                    NDArray* dLdI, NDArray* dLdW, const float epsilon, const bool isNCHW) {

    // unfortunately mkl dnn doesn't support any format (dnnl::memory::format_tag::any) for x

    // x -> 2D:nc, 4D:nchw/nhwc, 5D:ncdhw/ndhwc
    // mean -> 1D [c]
    // variance -> 1D [c]
    // dLdO - same shape as x
    // weights 2D [2, c], weights({0,1, 0,0}) contains gamma and weights({1,2, 0,0}) contains beta
    // dLdI - same shape as x
    // dLdW - same shape as weights, dLdW({0,1, 0,0}) contains grad_gamma and dLdW({1,2, 0,0}) contains grad_beta

    const int xRank = x->rankOf();

    // input type
    dnnl::memory::data_type type = dnnl::memory::data_type::f32;

    // indicate whether gamma or/and beta are given
    auto flags = dnnl::normalization_flags::use_global_stats;     // don't calculate the mean and variance for each mini-batch
    if (weights != nullptr)
        flags |= dnnl::normalization_flags::use_scale_shift;

    dnnl::memory::dims dims;
    dnnl::memory::format_tag format;

    const int indHW = isNCHW ? 2 : 1;
    const int bS = x->sizeAt(0);
    const int iC = isNCHW ? x->sizeAt(1) : x->sizeAt(-1);

    int iD, iH, iW;

    if(xRank == 2) {
        dims = {bS, iC};
        format = dnnl::memory::format_tag::nc;
    }
    else if(xRank == 4) {
        iH = x->sizeAt(indHW);
        iW = x->sizeAt(indHW + 1);
        dims = {bS, iC, iH, iW};
        format = isNCHW ? dnnl::memory::format_tag::nchw : dnnl::memory::format_tag::nhwc;
    }
    else {  // xRank = 5
        iD =  x->sizeAt(indHW);
        iH =  x->sizeAt(indHW + 1);
        iW =  x->sizeAt(indHW + 2);
        dims = {bS, iC, iD, iH, iW};
        format = isNCHW ? dnnl::memory::format_tag::ncdhw : dnnl::memory::format_tag::ndhwc;
    }

    // memory descriptors for arrays

    // x
    dnnl::memory::desc x_mkl_md  = dnnl::memory::desc(dims, type, format);
    dnnl::memory::desc x_user_md = dnnl::memory::desc(dims, type, format);
    mkldnnUtils::setBlockStrides(*x, x_user_md);

    // dLdO
    dnnl::memory::desc dLdO_mkl_md  = dnnl::memory::desc(dims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc dLdO_user_md = dnnl::memory::desc(dims, type, format);
    mkldnnUtils::setBlockStrides(dLdO, dLdO_user_md);

    // dLdI
    dnnl::memory::desc dLdI_mkl_md  = dnnl::memory::desc(dims, type, dnnl::memory::format_tag::any);
    dnnl::memory::desc dLdI_user_md = dnnl::memory::desc(dims, type, format);
    mkldnnUtils::setBlockStrides(*dLdI, dLdI_user_md);

    auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());

    // batchnorm forward description
    dnnl::batch_normalization_forward::desc op_ff_desc(dnnl::prop_kind::forward_inference, x_mkl_md, epsilon, flags);
    dnnl::batch_normalization_forward::primitive_desc op_ff_prim_desc(op_ff_desc, engine);

    // batchnorm backprop description
    dnnl::batch_normalization_backward::desc op_bp_desc(dnnl::prop_kind::backward, dLdO_mkl_md, x_mkl_md, epsilon, flags);
    dnnl::batch_normalization_backward::primitive_desc op_bp_prim_desc(op_bp_desc, engine, op_ff_prim_desc);

    // arguments (memory buffers) necessary for calculations
    std::unordered_map<int, dnnl::memory> args;

    dnnl::stream stream(engine);

    // provide memory and check whether reorder is required

    // x
    mkldnnUtils::loadDataToMklStream(*x, engine, stream, x_user_md, op_bp_prim_desc.src_desc(), args[DNNL_ARG_SRC]);

    // dLdO
    mkldnnUtils::loadDataToMklStream(dLdO, engine, stream, dLdO_user_md, op_bp_prim_desc.diff_dst_desc(), args[DNNL_ARG_DIFF_DST]);

    // mean
    auto mean_mkl_mem = dnnl::memory(op_bp_prim_desc.mean_desc(), engine, const_cast<void*>(mean->buffer()));
    args[DNNL_ARG_MEAN] = mean_mkl_mem;

    // variance
    auto var_mkl_mem = dnnl::memory(op_bp_prim_desc.variance_desc(), engine, const_cast<void*>(variance->buffer()));
    args[DNNL_ARG_VARIANCE] = var_mkl_mem;

    // dLdI
    auto dLdI_user_mem = mkldnnUtils::loadDataToMklStream(*dLdI, engine, stream, dLdI_user_md, op_bp_prim_desc.diff_src_desc(), args[DNNL_ARG_DIFF_SRC]);

    // gamma and beta (and their gradients) if they are present
    if(weights != nullptr) {

        auto w_mkl_mem = dnnl::memory(op_bp_prim_desc.weights_desc(), engine, const_cast<void*>(weights->buffer()));
        args[DNNL_ARG_WEIGHTS] = w_mkl_mem;

        auto dLdW_mkl_mem = dnnl::memory(op_bp_prim_desc.weights_desc(), engine, dLdW->buffer());
        args[DNNL_ARG_DIFF_WEIGHTS] = dLdW_mkl_mem;
    }

    // run calculations
    dnnl::batch_normalization_backward(op_bp_prim_desc).execute(stream, args);

    // reorder outputs if necessary
    if (op_bp_prim_desc.diff_src_desc() != dLdI_user_mem.get_desc())
        dnnl::reorder(args[DNNL_ARG_DIFF_SRC], dLdI_user_mem).execute(stream, args[DNNL_ARG_DIFF_SRC], dLdI_user_mem);

    stream.wait();

    // shape::printArray(dLdI_mkl_mem.map_data<float>(),8);

    // notations:
    // f = g * (gamma * ((x - m) / (v + eps)^0.5) + beta) -> means dLdO * ff_output
    // g = dLdO
    // stdInv = 1 / (v + eps)^0.5
    // N - batch size (product of spatial dimensions)

    // formula for full derivative with respect to input (x)
    // dLdI = dfdx + dfdm*dmdx + dfdv*(dvdm*dmdx + dvdx)

    // !!! MKL CALCULATES ONLY FIRST TERM dfdx, SO WE SHOULD CALCULATE TERM (dfdm*dmdx + dfdv*(dvdm*dmdx + dvdx)) BY OURSELF !!!

    // dfdm = -gamma*stdInv*g_sum;
    // dmdx  = 1/N;
    // dvdx  = 2 *  (x - m) / N
    // dvdm  = -2 * [(x - m)]_sum / N
    // dfdv  = -0.5 * [g*(x - m)]_sum * stdInv^3, drop gamma here for calc convenience

    // finally:
    // dLdI = dfdm / N + (2/N) * dfdv * (dvdm/2  + (x - m))
    // dLdI = gamma * (  stdInv * -g_sum/N + (2/N) * dfdv * (dvdm/2  + (x - m))  )

    std::vector<int> axes = isNCHW ? std::vector<int>{1} : std::vector<int>{xRank - 1};
    const auto excludedAxes = ShapeUtils::evalDimsToExclude(x->rankOf(), axes);

    // inversed batch size 1 / N
    const auto Ninv = 1.f * mean->lengthOf() / x->lengthOf();

    // x - mean
    NDArray xMinusMean(x); // empty array with same shape as x
    const_cast<NDArray*>(x)->applyBroadcast(sd::broadcast::Subtract, axes, *mean, xMinusMean);

    // stdInv
    NDArray stdInv = *variance + epsilon;
    stdInv.applyTransform(transform::Reciprocal, stdInv);                           // 1 / (variance + epsilon)
    stdInv.applyTransform(transform::Sqrt, stdInv);                                 // 1 / (variance + epsilon)^0.5

    // dfdm / N
    auto dfdm = dLdO.reduceAlongDimension(sd::reduce::Sum, excludedAxes);
    dfdm *= stdInv;
    dfdm *= -Ninv;

    // dvdm / 2
    NDArray dvdm(mean);                 // empty array with same shape as mean
    xMinusMean.reduceAlongDimension(sd::reduce::Sum, dvdm, excludedAxes);
    dvdm *= -Ninv;

    // (2/N)*dfdv
    NDArray dfdv(variance);                 // empty array with same shape as variance
    (xMinusMean * dLdO).reduceAlongDimension(sd::reduce::Sum, dfdv, excludedAxes);
    dfdv *= stdInv*stdInv*stdInv;
    dfdv *= -Ninv;

    // dvdm/2  + (x - m)
    xMinusMean.applyBroadcast(sd::broadcast::Add, axes, dvdm, xMinusMean);
    // dfdv * (dvdm/2  + (x - m))
    xMinusMean.applyBroadcast(sd::broadcast::Multiply, axes, dfdv, xMinusMean);
    // add dfdm / N
    xMinusMean.applyBroadcast(sd::broadcast::Add, axes, dfdm, xMinusMean);
    // * gamma
    auto gamma = (*weights)({0,1, 0,0});
    xMinusMean.applyBroadcast(sd::broadcast::Multiply, axes, gamma, xMinusMean);

    *dLdI += xMinusMean;
}

PLATFORM_IMPL(batchnorm, ENGINE_CPU) {

    auto input    = INPUT_VARIABLE(0);  // 2D:nc, 4D:nchw/nhwc, 5D:ncdhw/ndhwc
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

    const bool isNCHW = !(axes[0] == inRank - 1 && inRank > 2);

    batchnormMKLDNN(input, mean, variance, weights, output, epsilon, isNCHW);

    delete weights;

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(batchnorm, ENGINE_CPU) {

    auto input    = INPUT_VARIABLE(0);  // 2D:nc, 4D:nchw/nhwc, 5D:ncdhw/ndhwc
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

    return block.isUseMKLDNN() && axes.size() == 1 && (axes[0] == 1 || axes[0] == inRank - 1)  && (inRank == 2 || inRank == 4 || inRank == 5) &&
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
//     dnnl::memory::desc batchnorm_src_md(empty), batchnorm_dst_md(empty), user_src_md(empty), user_dst_md(empty);

//     auto flag = dnnl::normalization_flags::use_global_stats;
//     if (applyScale || applyOffset)
//         flag |= dnnl::normalization_flags::use_scale_shift;

//     mkldnnUtils::getMKLDNNMemoryDescBatchNorm(input, nullptr, output,
//                                               &batchnorm_src_md, nullptr, &batchnorm_dst_md,
//                                               &user_src_md, nullptr, &user_dst_md, axes[0]);

//     auto batchnorm_desc = dnnl::batch_normalization_forward::desc(dnnl::prop_kind::forward_inference, batchnorm_src_md, epsilon, flag);

//     auto engine = mkldnnUtils::getEngine(LaunchContext::defaultContext()->engine());
//     dnnl::stream stream(engine);
//     auto batchnorm_prim_desc = dnnl::batch_normalization_forward::primitive_desc(batchnorm_desc, engine);
//     auto user_src_memory = dnnl::memory(user_src_md, engine, input->buffer());
//     auto user_dst_memory = dnnl::memory(user_dst_md, engine, output->buffer());
//     auto batchnorm_mean_memory = dnnl::memory(batchnorm_prim_desc.mean_desc(), engine,
//                                                 mean->buffer());
//     auto batchnorm_variance_memory = dnnl::memory(batchnorm_prim_desc.variance_desc(), engine,
//                                                     variance->buffer());
//     auto batchnorm_src_memory = user_src_memory;
//     dnnl::memory m(batchnorm_src_md, engine);
//     if (m.get_desc() != user_src_memory.get_desc()) {
//         batchnorm_src_memory = dnnl::memory(batchnorm_src_md, engine);
//         dnnl::reorder(user_src_memory, batchnorm_src_memory).execute(stream, user_src_memory,
//                                                                batchnorm_src_memory);
//     }
//     auto batchnorm_dst_memory = user_dst_memory;
//     if (batchnorm_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
//         batchnorm_dst_memory = dnnl::memory(batchnorm_prim_desc.dst_desc(), engine);
//     }
//     if (applyScale || applyOffset) {
//         if (gamma != nullptr) {
//             weights({0, 1, 0, 0}).assign(gamma);
//         }
//         if (beta != nullptr) {
//             weights({1, 2, 0, 0}).assign(beta);
//         }

//         auto batchnorm_weights_memory = dnnl::memory(batchnorm_prim_desc.weights_desc(), engine, weights.buffer());
//         dnnl::batch_normalization_forward(batchnorm_prim_desc).execute(stream,
//                                                                  {{MKLDNN_ARG_SRC,      batchnorm_src_memory},
//                                                                   {MKLDNN_ARG_MEAN,     batchnorm_mean_memory},
//                                                                   {MKLDNN_ARG_VARIANCE, batchnorm_variance_memory},
//                                                                   {MKLDNN_ARG_WEIGHTS,  batchnorm_weights_memory},
//                                                                   {MKLDNN_ARG_DST,      batchnorm_dst_memory}});
//     } else {
//         dnnl::batch_normalization_forward(batchnorm_prim_desc).execute(stream,
//                                                                  {{MKLDNN_ARG_SRC,      batchnorm_src_memory},
//                                                                   {MKLDNN_ARG_MEAN,     batchnorm_mean_memory},
//                                                                   {MKLDNN_ARG_VARIANCE, batchnorm_variance_memory},
//                                                                   {MKLDNN_ARG_DST,      batchnorm_dst_memory}});
//     }
//     if (batchnorm_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
//         dnnl::reorder(batchnorm_dst_memory, user_dst_memory).execute(stream, batchnorm_dst_memory,
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
//            sd::MKLDNNStream::isSupported({input, mean, variance, gamma, beta, output}) &&
//            axes.size() == 1;
// }


//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batchnorm_bp, ENGINE_CPU) {

    NDArray* input    = INPUT_VARIABLE(0);                  // 2D:nc, 4D:nchw/nhwc, 5D:ncdhw/ndhwc
    NDArray* mean     = INPUT_VARIABLE(1);                  // [c]
    NDArray* variance = INPUT_VARIABLE(2);                  // [c]
    NDArray* gamma    = nullptr;                            // [c]
    NDArray* beta     = nullptr;                            // [c]
    NDArray* dLdO     = INPUT_VARIABLE(block.width() - 1);  // same as input

    NDArray* dLdI = OUTPUT_VARIABLE(0);                     // same as input
    NDArray* dLdM = OUTPUT_VARIABLE(1);                     // [c]
    NDArray* dLdV = OUTPUT_VARIABLE(2);                     // [c]
    NDArray* dLdG = nullptr;                                // [c]
    NDArray* dLdB = nullptr;                                // [c]

    const bool  applyScale  = (bool)INT_ARG(0);
    const bool  applyOffset = (bool)INT_ARG(1);
    const float epsilon     = T_ARG(0);

    if(applyScale) {
        gamma = INPUT_VARIABLE(3);
        dLdG  = OUTPUT_VARIABLE(3);
    }
    if(applyOffset) {
        beta = INPUT_VARIABLE(3 + (int)applyScale);
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

    // types of all input arrays should be the same
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

    const bool isNCHW = !(axes[0] == inRank - 1 && inRank > 2);

    if (shape::strideDescendingCAscendingF(dLdO->shapeInfo()))
        batchnormBpMKLDNN(input, mean, variance, *dLdO, weights, dLdI, dLdW, epsilon, isNCHW);
    else
        batchnormBpMKLDNN(input, mean, variance, dLdO->dup(), weights, dLdI, dLdW, epsilon, isNCHW);

    *dLdM = 0;
    *dLdV = 0;

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
PLATFORM_CHECK(batchnorm_bp, ENGINE_CPU) {

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

    return block.isUseMKLDNN() && axes.size() == 1 && (axes[0] == 1 || axes[0] == inRank - 1)  && (inRank == 2 || inRank == 4 || inRank == 5) &&
            (inputType == DataType::FLOAT32 && meanType  == DataType::FLOAT32 && varType  == DataType::FLOAT32 &&
             dLdOType  == DataType::FLOAT32 && gammaType == DataType::FLOAT32 && betaType == DataType::FLOAT32 &&
             dLdIType  == DataType::FLOAT32 && dLdGType  == DataType::FLOAT32 && dLdBType == DataType::FLOAT32);
}

}
}
}