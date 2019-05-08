/*******************************************************************************
 * Copyright (c) 2015-2019 Skymind, Inc.
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
// @author raver119@gmail.com, created on 29/10/17.
// @author Yurii Shyrma (iuriish@yahoo.com)
//

#include <op_boilerplate.h>
#if NOT_EXCLUDED(OP_batchnorm)

#include <ops/declarable/CustomOperations.h>
#include<ops/declarable/helpers/batchnorm.h>

namespace nd4j {
namespace ops {

#ifdef HAVE_MKLDNN
using namespace mkldnn;

static void getMKLDNNMemoryDescBatchNorm(const NDArray* src, const NDArray* diff_src, const NDArray* dst,
        mkldnn::memory::desc* batchnorm_src_md, mkldnn::memory::desc* batchnorm_diff_src_md, mkldnn::memory::desc* batchnorm_dst_md,
        mkldnn::memory::desc* user_src_md, mkldnn::memory::desc* user_diff_src_md, mkldnn::memory::desc* user_dst_md, int axis) {
    const Nd4jLong* shape = src->getShapeInfo();
    Nd4jLong rank = shape[0];
    Nd4jLong dim1 = axis; // MKL-DNN supports only 1 axis, which has to be the "channel" one
    Nd4jLong dim2 = axis >= 2 ? 1 : 2;
    Nd4jLong dim3 = axis >= 3 ? 2 : 3;
    mkldnn::memory::dims batchnorm_src_tz = { (int)shape[1], (int)shape[dim1 + 1], rank > 2 ? (int)shape[dim2 + 1] : 1, rank > 3 ? (int)shape[dim3 + 1] : 1};

    auto type = mkldnn::memory::data_type::f32;
    auto format = mkldnn::memory::format::nchw;
    auto supposed_to_be_any_format = mkldnn::memory::format::nChw8c; // doesn't work with "any"

    if (src != nullptr && src->getBuffer() != nullptr && batchnorm_src_md != nullptr) {
        *batchnorm_src_md = mkldnn::memory::desc({ batchnorm_src_tz }, type, supposed_to_be_any_format);
        *user_src_md = mkldnn::memory::desc({ batchnorm_src_tz }, type, format);
        user_src_md->data.format = mkldnn_blocked; // overrides format
        user_src_md->data.layout_desc.blocking.strides[0][0] = src->stridesOf()[0];
        user_src_md->data.layout_desc.blocking.strides[0][1] = src->stridesOf()[dim1];
        user_src_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? src->stridesOf()[dim2] : 1;
        user_src_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? src->stridesOf()[dim3] : 1;
    }

    if (diff_src != nullptr && diff_src->getBuffer() != nullptr && batchnorm_diff_src_md != nullptr) {
        *batchnorm_diff_src_md = mkldnn::memory::desc({ batchnorm_src_tz }, type, supposed_to_be_any_format);
        *user_diff_src_md = mkldnn::memory::desc({ batchnorm_src_tz }, type, format);
        user_diff_src_md->data.format = mkldnn_blocked; // overrides format
        user_diff_src_md->data.layout_desc.blocking.strides[0][0] = diff_src->stridesOf()[0];
        user_diff_src_md->data.layout_desc.blocking.strides[0][1] = diff_src->stridesOf()[dim1];
        user_diff_src_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? diff_src->stridesOf()[dim2] : 1;
        user_diff_src_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? diff_src->stridesOf()[dim3] : 1;
    }

    if (dst != nullptr && dst->getBuffer() != nullptr && batchnorm_dst_md != nullptr) {
        *batchnorm_dst_md = mkldnn::memory::desc({ batchnorm_src_tz }, type, supposed_to_be_any_format);
        *user_dst_md = mkldnn::memory::desc({ batchnorm_src_tz }, type, format);
        user_dst_md->data.format = mkldnn_blocked; // overrides format
        user_dst_md->data.layout_desc.blocking.strides[0][0] = dst->stridesOf()[0];
        user_dst_md->data.layout_desc.blocking.strides[0][1] = dst->stridesOf()[dim1];
        user_dst_md->data.layout_desc.blocking.strides[0][2] = rank > 2 ? dst->stridesOf()[dim2] : 1;
        user_dst_md->data.layout_desc.blocking.strides[0][3] = rank > 3 ? dst->stridesOf()[dim3] : 1;
    }
}
#endif

CUSTOM_OP_IMPL(batchnorm, 3, 1, false, 1, 2) {    
    auto input    = INPUT_VARIABLE(0);
    auto mean     = INPUT_VARIABLE(1);
    auto variance = INPUT_VARIABLE(2);
    NDArray *gamma    = nullptr;
    NDArray *beta     = nullptr;

    auto output   = OUTPUT_VARIABLE(0);

    const bool applyScale  = (bool)INT_ARG(0);
    const bool applyOffset = (bool)INT_ARG(1);

    // FIXME: double?
    const double epsilon     = T_ARG(0);

    if(applyScale)
        gamma = INPUT_VARIABLE(3);    
    if(applyOffset)
        beta = INPUT_VARIABLE(3 + static_cast<int>(applyScale));    

    std::vector<const NDArray*> inArrs(block.width());
    for(int i = 0; i < block.width(); ++i)
        inArrs[i] = INPUT_VARIABLE(i);

    // check whether all input shapes are mutually broadcastable
    Nd4jLong* outShapeInfo = nullptr;
    const bool areShapesOk = ShapeUtils::evalCommonBroadcastShapeInfo(inArrs, outShapeInfo, block.getWorkspace());
    REQUIRE_TRUE(areShapesOk, 0, "BATCHNORM op: the shapes of input arrays are not mutually broadcastable !");
    RELEASE(outShapeInfo, block.getWorkspace());

    // normalized output = gamma * ((input - mean) / sqrt(variance + epsilon)) + beta

    auto sigmaInvGam = (*variance + epsilon).transform(transform::RSqrt);
    if(applyScale)
        sigmaInvGam *= *gamma;

    NDArray inputMinusMean;
    if(!input->isSameShape(output) && !mean->isSameShape(output)) {
        auto inputTiled = NDArray(output, false, block.getWorkspace());
        input->tile(inputTiled);
        inputMinusMean = inputTiled - *mean;
    }
    else
        inputMinusMean = *input - *mean;       

    if (applyOffset)
        output->assign(inputMinusMean * sigmaInvGam + *beta);
    else 
        output->assign(inputMinusMean * sigmaInvGam);

    return Status::OK();
}

    DECLARE_TYPES(batchnorm) {
        getOpDescriptor()
                ->setAllowedInputTypes(nd4j::DataType::ANY)
                ->setAllowedOutputTypes({ALL_FLOATS});
    }


//////////////////////////////////////////////////////////////////////////
DECLARE_SHAPE_FN(batchnorm) {        

    std::vector<const NDArray*> inArrs(block.width());
    auto in = inputShape->at(0);
    for(int i = 0; i < block.width(); ++i)
        inArrs[i] = INPUT_VARIABLE(i);

    // check whether all input shapes are mutually broadcastable
    Nd4jLong* outShapeInfo = nullptr;
    const bool areShapesOk = ShapeUtils::evalCommonBroadcastShapeInfo(inArrs, outShapeInfo, block.getWorkspace());
    REQUIRE_TRUE(areShapesOk, 0, "BATCHNORM op: the shapes of input arrays are not mutually broadcastable !");

    ArrayOptions::setDataType(outShapeInfo, DataTypeUtils::pickFloatingType(ArrayOptions::dataType(in)));

    return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(batchnorm_new, 3, 1, false, 1, 2) {

    auto input    = INPUT_VARIABLE(0);
    auto mean     = INPUT_VARIABLE(1);
    auto variance = INPUT_VARIABLE(2);
    NDArray* gamma    = nullptr;
    NDArray* beta     = nullptr;

    auto output   = OUTPUT_VARIABLE(0);

    const bool   applyScale  = (bool)INT_ARG(0);
    const bool   applyOffset = (bool)INT_ARG(1);
    const double epsilon     = T_ARG(0);

    if(applyScale)
        gamma = INPUT_VARIABLE(3);
    if(applyOffset)
        beta = INPUT_VARIABLE(3 + static_cast<int>(applyScale));

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
    REQUIRE_TRUE(numOfAxes <= inRank, 0, "BATCHNORM_NEW op: too big number of input axes to normalize over, expected number should be less or equal to rank of input array, but got %i and %i correspondingly !", numOfAxes, inRank);

    // get, for example, something like {1, inDim1, 1, inDim3, 1} if axes = {1, 3}
    std::vector<Nd4jLong> expShapeWithUnities(inRank, 1);
    for(int i = 0; i < numOfAxes; ++i)
        expShapeWithUnities[axes[i]] = input->sizeAt(axes[i]);

    // evaluate expected shape for mean, variance and gamma. These 3 arrays should have identical shapes
    // for example if input shape is {2,3,4,5,6} and axes = {1,3}, then expected shape would be {1,3,1,5,1}, and if axes = {3}, then expected shape would be {5}
    std::vector<Nd4jLong> expShape = numOfAxes == 1 ? std::vector<Nd4jLong>(1, input->sizeAt(axes[0])) : expShapeWithUnities;
    std::string expShapeStr = ShapeUtils::shapeAsString(expShape);

    REQUIRE_TRUE(ShapeUtils::shapeAsString(mean)     == expShapeStr, 0, "BATCHNORM_NEW op: wrong shape of mean array, expected is %s, but got %s instead !", expShapeStr.c_str(), ShapeUtils::shapeAsString(mean).c_str());
    REQUIRE_TRUE(ShapeUtils::shapeAsString(variance) == expShapeStr, 0, "BATCHNORM_NEW op: wrong shape of variance array, expected is %s, but got %s instead !", expShapeStr.c_str(), ShapeUtils::shapeAsString(variance).c_str());
    if(gamma)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(gamma) == expShapeStr, 0, "BATCHNORM_NEW op: wrong shape of gamma array, expected is %s, but got %s instead !", expShapeStr.c_str(), ShapeUtils::shapeAsString(gamma).c_str());
    if(beta)
        REQUIRE_TRUE(ShapeUtils::shapeAsString(beta) == expShapeStr, 0, "BATCHNORM_NEW op: wrong shape of beta array, expected is %s, but got %s instead !", expShapeStr.c_str(), ShapeUtils::shapeAsString(beta).c_str());

    // types of all input arrays should be the same
    for(int i = 1; i < block.width(); ++i)
        REQUIRE_TRUE(INPUT_VARIABLE(0)->dataType() == INPUT_VARIABLE(i)->dataType(), 0, "BATCHNORM_NEW op: types of all input arrays should be the same !");

#ifdef HAVE_MKLDNN
    if (block.isUseMKLDNN() && nd4j::MKLDNNStream::isSupported({input, mean, variance, gamma, beta, output}) && numOfAxes == 1) {
        std::vector<nd4j::MKLDNNStream>& streams = block.getMKLDNNStreams();
        if (streams.empty()) {
            streams.push_back(MKLDNNStream("batchnorm_new"));
        }

        std::vector<Nd4jLong> shape({2, mean->lengthOf()});
        NDArray weights = NDArrayFactory::create<float>('c', shape, block.getWorkspace());
        weights({0, 1, 0, 0}).assign(1.0f);
        weights({1, 2, 0, 0}).assign(0.0f);

        if (streams[0].checkAndReset({input, mean, variance, gamma, beta}, {output}, {(float)epsilon}, axes)) {
            mkldnn_memory_desc_t empty;
            mkldnn::memory::desc batchnorm_src_md(empty), batchnorm_dst_md(empty), user_src_md(empty), user_dst_md(empty);

            getMKLDNNMemoryDescBatchNorm(input, nullptr, output,
                        &batchnorm_src_md, nullptr, &batchnorm_dst_md,
                        &user_src_md, nullptr, &user_dst_md, axes[0]);

            auto batchnorm_desc = batch_normalization_forward::desc(prop_kind::forward_inference, batchnorm_src_md, epsilon,
                            use_global_stats | (applyScale || applyOffset ? use_scale_shift : 0));

            auto engine = streams[0].getEngine();
            auto batchnorm_prim_desc = batch_normalization_forward::primitive_desc(batchnorm_desc, engine);
            auto user_src_memory = mkldnn::memory({user_src_md, engine}, input->buffer());
            auto user_dst_memory = mkldnn::memory({user_dst_md, engine}, output->buffer());
            auto batchnorm_mean_memory = mkldnn::memory(batchnorm_prim_desc.mean_primitive_desc(), mean->buffer());
            auto batchnorm_variance_memory = mkldnn::memory(batchnorm_prim_desc.variance_primitive_desc(), variance->buffer());

            auto batchnorm_src_memory = user_src_memory;
            streams[0].addMemory(user_src_memory);
            if (mkldnn::memory::primitive_desc({batchnorm_src_md, engine})
                    != user_src_memory.get_primitive_desc()) {
                batchnorm_src_memory = mkldnn::memory({batchnorm_src_md, engine});
                streams[0].addMemory(batchnorm_src_memory);
                streams[0].addOperation(reorder(user_src_memory, batchnorm_src_memory));
            }

            auto batchnorm_dst_memory = user_dst_memory;
            streams[0].addMemory(user_dst_memory);
            if (mkldnn::memory::primitive_desc(batchnorm_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                batchnorm_dst_memory = mkldnn::memory(batchnorm_prim_desc.dst_primitive_desc());
                streams[0].addMemory(batchnorm_dst_memory);
            }

            streams[0].addMemory(batchnorm_mean_memory);
            streams[0].addMemory(batchnorm_variance_memory);

            if (applyScale || applyOffset) {
                auto batchnorm_weights_memory = mkldnn::memory(batchnorm_prim_desc.weights_primitive_desc(), weights.buffer());
                streams[0].addMemory(batchnorm_weights_memory);
                streams[0].addOperation(batch_normalization_forward(batchnorm_prim_desc, (mkldnn::primitive::at)batchnorm_src_memory,
                        (mkldnn::primitive::at)batchnorm_mean_memory, (mkldnn::primitive::at)batchnorm_variance_memory, (mkldnn::primitive::at)batchnorm_weights_memory, batchnorm_dst_memory));
            } else {
                streams[0].addOperation(batch_normalization_forward(batchnorm_prim_desc, (mkldnn::primitive::at)batchnorm_src_memory,
                        (mkldnn::primitive::at)batchnorm_mean_memory, (mkldnn::primitive::at)batchnorm_variance_memory, batchnorm_dst_memory));
            }

            if (mkldnn::memory::primitive_desc(batchnorm_prim_desc.dst_primitive_desc())
                    != user_dst_memory.get_primitive_desc()) {
                streams[0].addOperation(reorder(batchnorm_dst_memory, user_dst_memory));
            }
        }

        if (applyScale || applyOffset) {
            if (gamma != nullptr) {
                weights({0, 1, 0, 0}).assign(gamma);
            }
            if (beta != nullptr) {
                weights({1, 2, 0, 0}).assign(beta);
            }
        }
        streams[0].submitAndWait();
        return Status::OK();
    }
#endif
    nd4j_debug("MKL-DNN is not used for batchnorm_new!\n", 0);

    // formula: output = gamma * ((input - mean) / sqrt(variance + epsilon)) + beta
    helpers::batchnorm(input, mean, variance, gamma, beta, output, axes, epsilon);

    return Status::OK();
}

DECLARE_TYPES(batchnorm_new) {
    getOpDescriptor()->setAllowedInputTypes({ALL_FLOATS})->setSameMode(true);
}

DECLARE_SHAPE_FN(batchnorm_new) {

    auto inShapeInfo = inputShape->at(0);
    DataType outType = DataTypeUtils::pickFloatingType(ArrayOptions::dataType(inShapeInfo));
     
    Nd4jLong* outShapeInfo = ShapeBuilders::copyShapeInfoAndType(inShapeInfo, outType, false, block.getWorkspace());    // output shape is identical to input shape

    return SHAPELIST(outShapeInfo);
}

//////////////////////////////////////////////////////////////////////////
CUSTOM_OP_IMPL(batchnorm_bp, 4, 3, false, 1, 2) {
    auto input    = INPUT_VARIABLE(0);
    auto mean     = INPUT_VARIABLE(1);
    auto variance = INPUT_VARIABLE(2);
    NDArray *gamma    = nullptr;
    NDArray *beta     = nullptr;
    NDArray *dLdO     = nullptr;                 // next epsilon

    auto dLdI = OUTPUT_VARIABLE(0);
    auto dLdM = OUTPUT_VARIABLE(1);
    auto dLdV = OUTPUT_VARIABLE(2);
    NDArray *dLdG = nullptr;
    NDArray *dLdB = nullptr;

    const bool applyScale  = (bool)INT_ARG(0);
    const bool applyOffset = (bool)INT_ARG(1);

    // FIXME: double?
    const double    epsilon     = T_ARG(0);

    const int dLdONum = static_cast<int>(applyScale) + static_cast<int>(applyOffset);

    if(applyScale) {
        gamma = INPUT_VARIABLE(3);
        dLdG  = OUTPUT_VARIABLE(3);
    }
    if(applyOffset) {
        beta = INPUT_VARIABLE(3 + static_cast<int>(applyScale));
        dLdB = OUTPUT_VARIABLE(3 + static_cast<int>(applyScale));
    }
        
    dLdO = INPUT_VARIABLE(3 + dLdONum);
    
    std::vector<const NDArray*> inArrs(block.width());
    for(int i = 0; i < 4 + dLdONum; ++i)
        inArrs[i] = INPUT_VARIABLE(i);

    // check whether all input shapes are mutually broadcastable
    Nd4jLong* outShapeInfo = nullptr;
    const bool areShapesOk = ShapeUtils::evalCommonBroadcastShapeInfo(inArrs, outShapeInfo, block.getWorkspace());
    REQUIRE_TRUE(areShapesOk, 0, "BATCHNORM_BP op: the shapes of input arrays are not mutually broadcastable !");    
    RELEASE(outShapeInfo, block.getWorkspace());

    // ***** calculations ***** //

    auto sigmaInv = (*variance + epsilon).transform(transform::RSqrt);
    
    NDArray sigmaInvGamdLdO = -sigmaInv * *dLdO;
    if(applyScale)
        sigmaInvGamdLdO *= *gamma;

    NDArray inputMinusMean;
    if(!input->isSameShape(dLdO) && !mean->isSameShape(dLdO)) {
        auto inputTiled = NDArray(dLdO, false, block.getWorkspace());
        input->tile(inputTiled);
        inputMinusMean = inputTiled - *mean;
    }
    else
        inputMinusMean = *input - *mean;

    // dLdI
    if(!dLdI->isSameShape(dLdO))
        dLdI->assign( (-sigmaInvGamdLdO).reduceAlongDims(reduce::Sum, ShapeUtils::evalBroadcastBackwardAxis(dLdI->getShapeInfo(), dLdO->getShapeInfo())) );
    else
        dLdI->assign(-sigmaInvGamdLdO);

    // dLdM
    if(!dLdM->isSameShape(dLdO))
        dLdM->assign( sigmaInvGamdLdO.reduceAlongDims(reduce::Sum, ShapeUtils::evalBroadcastBackwardAxis(dLdM->getShapeInfo(), dLdO->getShapeInfo())) );
    else
        dLdM->assign(sigmaInvGamdLdO);

    // dLdV
    if(!dLdV->isSameShape(dLdO)) {
        dLdV->assign( (sigmaInv * sigmaInv * sigmaInvGamdLdO * inputMinusMean * 0.5f).reduceAlongDims(reduce::Sum, ShapeUtils::evalBroadcastBackwardAxis(dLdV->getShapeInfo(), dLdO->getShapeInfo())) );
    }
    else
        dLdV->assign(sigmaInv * sigmaInv * sigmaInvGamdLdO * inputMinusMean * 0.5f);

    // dLdG
    if(applyScale) {
        if(!dLdG->isSameShape(dLdO))
            dLdG->assign( (sigmaInv * inputMinusMean * *dLdO).reduceAlongDims(reduce::Sum, ShapeUtils::evalBroadcastBackwardAxis(dLdG->getShapeInfo(), dLdO->getShapeInfo())) );
        else
            dLdG->assign(sigmaInv * inputMinusMean * *dLdO);
    }

    // dLdB
    if(applyOffset) {
        if(!dLdB->isSameShape(dLdO))
            dLdB->assign(dLdO->reduceAlongDims(reduce::Sum, ShapeUtils::evalBroadcastBackwardAxis(dLdB->getShapeInfo(), dLdO->getShapeInfo())) );
        else
            dLdB->assign(dLdO);
    }

    return Status::OK();
}

        DECLARE_TYPES(batchnorm_bp) {
            getOpDescriptor()
                    ->setAllowedInputTypes(0, nd4j::DataType::ANY)
                    ->setAllowedInputTypes(1, nd4j::DataType::ANY)
                    ->setAllowedInputTypes(2, nd4j::DataType::ANY)
                    ->setAllowedInputTypes(3, nd4j::DataType::ANY)
                    ->setAllowedInputTypes(4, nd4j::DataType::ANY)
                    ->setAllowedInputTypes(5, {ALL_FLOATS})
                    ->setAllowedOutputTypes({ALL_FLOATS});
        }

//////////////////////////////////////////////////////////////////////////

DECLARE_SHAPE_FN(batchnorm_bp) {

    const bool applyScale  = (bool)INT_ARG(0);
    const bool applyOffset = (bool)INT_ARG(1);

    const int dLdONum = static_cast<int>(applyScale) + static_cast<int>(applyOffset);

    std::vector<const NDArray*> inArrs(block.width());
    for(int i = 0; i < 4 + dLdONum; ++i)
        inArrs[i] = INPUT_VARIABLE(i);

    // check whether all input shapes are mutually broadcastable
    Nd4jLong* outShapeInfo = nullptr;
    const bool areShapesOk = ShapeUtils::evalCommonBroadcastShapeInfo(inArrs, outShapeInfo, block.getWorkspace());
    REQUIRE_TRUE(areShapesOk, 0, "BATCHNORM_BP op: the shapes of input arrays are not mutually broadcastable !");    
    RELEASE(outShapeInfo, block.getWorkspace());

    Nd4jLong* dLdIShapeInfo(nullptr), *dLdMShapeInfo(nullptr), *dLdVShapeInfo(nullptr), *dLdGShapeInfo(nullptr), *dLdBShapeInfo(nullptr);
    COPY_SHAPE(inputShape->at(0), dLdIShapeInfo);
    COPY_SHAPE(inputShape->at(1), dLdMShapeInfo);
    COPY_SHAPE(inputShape->at(2), dLdVShapeInfo);

    if(applyScale) {
        COPY_SHAPE(inputShape->at(3), dLdGShapeInfo);
    }
    if(applyOffset){
        COPY_SHAPE(inputShape->at(3 + static_cast<int>(applyScale)), dLdBShapeInfo);
    }

    if(!applyScale && !applyOffset)
        return SHAPELIST(dLdIShapeInfo, dLdMShapeInfo, dLdVShapeInfo);

    if(applyScale && !applyOffset)
        return SHAPELIST(dLdIShapeInfo, dLdMShapeInfo, dLdVShapeInfo, dLdGShapeInfo);

    if(!applyScale && applyOffset)
        return SHAPELIST(dLdIShapeInfo, dLdMShapeInfo, dLdVShapeInfo, dLdBShapeInfo);

    return SHAPELIST(dLdIShapeInfo, dLdMShapeInfo, dLdVShapeInfo, dLdGShapeInfo, dLdBShapeInfo);
}
        // //////////////////////////////////////////////////////////////////////////
        // CONFIGURABLE_OP_IMPL(batchnorm_bp, 5, 1, true, 0, 1) {

        //     NDArray<T>* input = INPUT_VARIABLE(0);
        //     NDArray<T>* epsilon = INPUT_VARIABLE(1);
        //     NDArray<T>* gamma = INPUT_VARIABLE(2);
        //     NDArray<T>* dGlobalMeanView = INPUT_VARIABLE(3);
        //     NDArray<T>* dGlobalVarView = INPUT_VARIABLE(4);
        //     NDArray<T>* outEpsilon = this->getZ(block);
        //     std::vector<int> argI = *(block.getIArguments());
        //     const int bS = epsilon->sizeAt(0);
        //     bool isLockGammaBeta = (bool)argI[0];
        //     const int* epsilonShape = epsilon->getShapeInfo() + 1;
        //     const T eps = (T)1e-5;

        //     int rank = epsilon->rankOf();
        //     std::initializer_list<int> dimensions;
        //     int effectiveBatchSize;
        //     if (rank == 2) {
        //         dimensions = {0};
        //         effectiveBatchSize = bS;
        //     }
        //     else if (rank == 4) {
        //         dimensions = {0, 2, 3};
        //         effectiveBatchSize = input->sizeAt(0)*input->sizeAt(2)*input->sizeAt(3);
        //     }
        //     else
        //         throw "Graph operation batchnorm_bp: the epsilon rank must be equal to 2 or 4 !";

        //     NDArray<T> *mean(nullptr), *var(nullptr), *dBeta(nullptr), *dGamma(nullptr), *dLdVar(nullptr), *dxmu1(nullptr), *dxmu2(nullptr);
        //     mean = input->template reduceAlongDimension<simdOps::Mean<T>>(dimensions);
        //     var = input->template varianceAlongDimension<simdOps::SummaryStatsVariance<T>>(false, dimensions);
        //     var->template applyScalar<simdOps::Add<T>>(eps, nullptr);
        //     auto std = new NDArray<T>(var->getShapeInfo(), block.getWorkspace());
        //     var->template applyTransform<simdOps::Sqrt<T>>(std, nullptr);

        //     auto xMu = new NDArray<T>(input->getShapeInfo(), block.getWorkspace());
        //     auto xHat = new NDArray<T>(input->getShapeInfo(), block.getWorkspace());
        //     auto temp1 = new NDArray<T>(epsilon->getShapeInfo(), block.getWorkspace());
        //     auto temp2 = new NDArray<T>(std->getShapeInfo(), block.getWorkspace());
        //     auto dGammaView = new NDArray<T>('c', {1, epsilonShape[1]}, block.getWorkspace());
        //     auto dBetaView = new NDArray<T>('c', {1, epsilonShape[1]}, block.getWorkspace());
        //     auto dxhat = new NDArray<T>(epsilon->getShapeInfo(), block.getWorkspace());

        //     if (rank == 2) {
        //         input->subRowVector(mean, xMu);
        //         xMu->divRowVector(std, xHat);
        //     }
        //     else {
        //         input->template applyBroadcast<simdOps::Subtract<T>>({1}, mean, xMu, nullptr);
        //         xMu->template applyBroadcast<simdOps::Divide<T>>({1}, std, xHat, nullptr);
        //     }

        //     dBeta = epsilon->sum(dimensions); // dL/dBeta = sum_examples dL/dOut
        //     epsilon->template applyPairwiseTransform<simdOps::Multiply<T>>(xHat, temp1, nullptr);   //dL/dGamma = sum_examples dL/dOut .* xHat
        //     dGamma = temp1->sum(dimensions);  //dL/dGamma = sum_examples dL/dOut .* xHat

        //     if (isLockGammaBeta)
        //         epsilon->template applyPairwiseTransform<simdOps::Multiply<T>>(gamma, dxhat, nullptr);
        //     else {// Standard case
        //         if(rank == 2)
        //             epsilon->mulRowVector(gamma, dxhat); //dL/dxHat = dL/dOut . gamma        Shape: [minibatchSize, nOut]
        //         else
        //             epsilon->template applyBroadcast<simdOps::Multiply<T>>({1}, gamma, dxhat, nullptr);
        //     }

        //     // dLdVar - dL/dVariance, shape: [1, miniBatch]
        //     dxhat->template applyPairwiseTransform<simdOps::Multiply<T>>(xMu, temp1, nullptr);
        //     dLdVar = temp1->sum(dimensions);
        //     dLdVar->template applyScalar<simdOps::Multiply<T>>((T)-0.5, nullptr);
        //     T powParams[] = {(T)(-3.)};
        //     std->template applyTransform<simdOps::Pow<T>>(temp2, powParams);
        //     dLdVar->template applyPairwiseTransform<simdOps::Multiply<T>>(temp2, nullptr);

        //     //dL/dmu
        //     dxmu1 = dxhat->sum(dimensions);
        //     dxmu1->template applyPairwiseTransform<simdOps::Divide<T>>(std, nullptr);
        //     dxmu1->template applyTransform<simdOps::Neg<T>>();
        //     dxmu2 = xMu->sum(dimensions);
        //     dxmu2->template applyScalar<simdOps::Multiply<T>>((T)(-2.)/effectiveBatchSize);
        //     dxmu2->template applyPairwiseTransform<simdOps::Multiply<T>>(dLdVar, nullptr);

        //     dxmu1->template applyPairwiseTransform<simdOps::Add<T>>(dxmu2, nullptr);
        //     NDArray<T>* dLdmu = dxmu1;      //  = dL/dmu Shape: [1, nOut]

        //     //Note the array reuse here: dxhat, xMu, dLdVar, dLdmu - all are invalid after this line (but aren't used later anyway)
        //     NDArray<T>* dLdx = dxhat;
        //     dLdVar->template applyScalar<simdOps::Multiply<T>>((T)(2.)/effectiveBatchSize);
        //     dLdmu->template applyScalar<simdOps::Multiply<T>>((T)(1.)/effectiveBatchSize);
        //     if(rank == 2) {
        //         dLdx->divRowVector(std, dLdx);
        //         xMu->mulRowVector(dLdVar, xMu);
        //     }
        //     else {
        //         dLdx->template applyBroadcast<simdOps::Divide<T>>({1}, std, dLdx, nullptr);
        //         xMu->template applyBroadcast<simdOps::Multiply<T>>({1}, dLdVar, xMu, nullptr);
        //     }
        //     dLdx->template applyPairwiseTransform<simdOps::Add<T>>(xMu, nullptr);
        //     if(rank == 2)
        //         dLdx->addRowVector(dLdmu, dLdx);
        //     else
        //         dLdx->template applyBroadcast<simdOps::Add<T>>({1}, dLdmu, dLdx, nullptr);

        //     *outEpsilon = *dLdx;

        //     //TODO rework this to avoid the assign here
        //     // dGammaView->assign(dGamma);
        //     // dBetaView->assign(dBeta);
        //     // dGlobalMeanView->assign((T)0.);
        //     // dGlobalVarView->assign((T)0.);
        //     // retGradient.setGradientFor(BatchNormalizationParamInitializer.GAMMA, dGammaView);
        //     // retGradient.setGradientFor(BatchNormalizationParamInitializer.BETA, dBetaView);
        //     // retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_MEAN, dGlobalMeanView);
        //     // retGradient.setGradientFor(BatchNormalizationParamInitializer.GLOBAL_VAR, dGlobalVarView);

        //     delete std;
        //     delete xMu;
        //     delete xHat;
        //     delete mean;
        //     delete var;
        //     delete dBeta;
        //     delete dGamma;
        //     delete dLdVar;
        //     delete dxmu1;
        //     delete dxmu2;
        //     delete temp1;
        //     delete temp2;
        //     delete dxhat;
        //     delete dGammaView;
        //     delete dBetaView;

        //     return ND4J_STATUS_OK;
        // }





}
}

#endif
