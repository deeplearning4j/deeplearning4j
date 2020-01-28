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
// @author raver119@gmail.com
//

#include "testlayers.h"
#include <initializer_list>
#include <NDArrayFactory.h>
#include <ops/declarable/PlatformHelper.h>
#include <ops/declarable/CustomOperations.h>
#include <execution/Engine.h>

#ifdef HAVE_CUDNN

#include <ops/declarable/platform/cudnn/cudnnUtils.h>

#endif

using namespace nd4j;

class CuDnnTests : public testing::Test {
public:

};

static void printer(std::initializer_list<nd4j::ops::platforms::PlatformHelper*> helpers) {

    for (auto v:helpers) {
        nd4j_printf("Initialized [%s]\n", v->name().c_str());
    }
}


TEST_F(CuDnnTests, helpers_includer) {
    // we need this block, to make sure all helpers are still available within binary, and not optimized out by linker
#ifdef HAVE_CUDNN
    nd4j::ops::platforms::PLATFORM_conv2d_ENGINE_CUDA conv2d;
    nd4j::ops::platforms::PLATFORM_conv2d_bp_ENGINE_CUDA conv2d_bp;
    nd4j::ops::platforms::PLATFORM_conv3dnew_ENGINE_CUDA conv3dnew;
    nd4j::ops::platforms::PLATFORM_conv3dnew_bp_ENGINE_CUDA conv3dnew_bp;
    nd4j::ops::platforms::PLATFORM_depthwise_conv2d_ENGINE_CUDA depthwise_conv2d;
    nd4j::ops::platforms::PLATFORM_depthwise_conv2d_bp_ENGINE_CUDA depthwise_conv2d_bp;
    nd4j::ops::platforms::PLATFORM_batchnorm_ENGINE_CUDA batchnorm;
    nd4j::ops::platforms::PLATFORM_batchnorm_bp_ENGINE_CUDA batchnorm_bp;
    nd4j::ops::platforms::PLATFORM_avgpool2d_ENGINE_CUDA avgpool2d;
    nd4j::ops::platforms::PLATFORM_avgpool2d_bp_ENGINE_CUDA avgpool2d_bp;
    nd4j::ops::platforms::PLATFORM_maxpool2d_ENGINE_CUDA maxpool2d;
    nd4j::ops::platforms::PLATFORM_maxpool2d_bp_ENGINE_CUDA maxpool2d_bp;
    nd4j::ops::platforms::PLATFORM_avgpool3dnew_ENGINE_CUDA avgpool3dnew;
    nd4j::ops::platforms::PLATFORM_avgpool3dnew_bp_ENGINE_CUDA avgpool3dnew_bp;
    nd4j::ops::platforms::PLATFORM_maxpool3dnew_ENGINE_CUDA maxpool3dnew;
    nd4j::ops::platforms::PLATFORM_maxpool3dnew_bp_ENGINE_CUDA maxpool3dnew_bp;



    printer({&conv2d});
    printer({&conv2d_bp});
    printer({&conv3dnew});
    printer({&conv3dnew_bp});
    printer({&depthwise_conv2d});
    printer({&depthwise_conv2d_bp});
    printer({&batchnorm});
    printer({&batchnorm_bp});
    printer({&avgpool2d});
    printer({&avgpool2d_bp});
    printer({&maxpool2d});
    printer({&maxpool2d_bp});
    printer({&avgpool3dnew});
    printer({&avgpool3dnew_bp});
    printer({&maxpool3dnew});
    printer({&maxpool3dnew_bp});
#endif
}


TEST_F(CuDnnTests, mixed_helpers_test_1) {
#if defined(HAVE_CUDNN) && defined (HAVE_MKLDNN)
    nd4j_printf("Mixed platforms test\n", "");


    int bS=2, iH=4,iW=3,  iC=4,oC=3,  kH=3,kW=2,  sH=1,sW=1,  pH=0,pW=0,  dH=1,dW=1;
    int       oH=2,oW=2;
    int paddingMode = 0;             // 1-SAME, 0-VALID;
    int dataFormat  = 0;             // 1-NHWC, 0-NCHW

    auto input    = NDArrayFactory::create<float>('c', {bS, iC, iH, iW});
    auto weights  = NDArrayFactory::create<float>('c', {oC, iC, kH, kW});
    auto bias     = NDArrayFactory::create<float>('c', {oC}, {1,2,3});

    auto expOutput = NDArrayFactory::create<float>('c', {bS, oC, oH, oW}, {61.f,   61.f,  61.f,   61.f, 177.2f,  177.2f, 177.2f,  177.2f, 293.4f,  293.4f, 293.4f,  293.4f,  61.f,   61.f,  61.f,   61.f, 177.2f,  177.2f, 177.2f,  177.2f, 293.4f,  293.4f, 293.4f,  293.4f});
    auto zCUDA = expOutput.like();
    auto zMKL = expOutput.like();

    input = 2.;
    weights.linspace(0.1, 0.1);
    weights.permutei({2,3,1,0});

    input.syncToHost();
    weights.syncToHost();
    bias.syncToHost();

    nd4j::ops::conv2d op;

    // cuDNN part
    Context cuda(1);
    cuda.setTargetEngine(samediff::Engine::ENGINE_CUDA);
    cuda.setInputArray(0, &input);
    cuda.setInputArray(1, &weights);
    cuda.setInputArray(2, &bias);
    cuda.setOutputArray(0, &zCUDA);
    cuda.setIArguments({kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto statusCUDA = op.execute(&cuda);

    ASSERT_EQ(Status::OK(), statusCUDA);
    ASSERT_EQ(expOutput, zCUDA);

    // MKL-DNN part
    Context mkl(1);
    mkl.setTargetEngine(samediff::Engine::ENGINE_CPU);
    mkl.setInputArray(0, &input);
    mkl.setInputArray(1, &weights);
    mkl.setInputArray(2, &bias);
    mkl.setOutputArray(0, &zMKL);
    mkl.setIArguments({kH,kW,  sH,sW,  pH,pW,  dH,dW, paddingMode, dataFormat});
    auto statusMKL = op.execute(&mkl);

    zMKL.tickWriteHost();

    ASSERT_EQ(Status::OK(), statusMKL);
    ASSERT_EQ(expOutput, zMKL);
#endif
}