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
// @author Yurii Shyrma (iuriish@yahoo.com)
//


#include "cudnnUtils.h"
#include <ops/declarable/helpers/convolutions.h>

namespace sd      {
namespace ops       {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void batchnormCUDNN(const LaunchContext* context,
                            const NDArray* input, const NDArray* mean, const NDArray* variance,
                            const NDArray* gamma, const NDArray* beta,
                                  NDArray* output,
                            const double epsilon, const bool isSpatialMode) {


    // input, output -> 4D:nchw, 5D:ncdhw
    // mean, variance, gamma, beta -> 1xCx1x1 for 4D and 1xCx1x1x1 for 5D for BATCHNORM_MODE_SPATIAL mode
    //                             -> 1xCxHxW for 4D and 1xCxDxHxW for 5D for BATCHNORM_MODE_PER_ACTIVATION mode

    const cudnnDataType_t dataType = cudnnDataType(input->dataType());

    const int xRank = input->rankOf();

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("conv2dCUDNN: can't set stream for cuDNN", err);

    const std::vector<int> xShape = input->getShapeAsVectorInt();               // input and output have same shapes

    std::vector<int> paramsShape, paramsStrides;                                 // mean, variance, gamma and beta have same shapes
    if(isSpatialMode) { // 1xCx1x1
        const int iC = mean->lengthOf();
        const int stride0 = mean->strideAt(0);
        paramsShape   = xRank == 4 ? std::vector<int>({1, iC, 1, 1}) : std::vector<int>({1, iC, 1, 1, 1});
        paramsStrides = xRank == 4 ? std::vector<int>({iC*stride0, stride0, 1, 1}) : std::vector<int>({iC*stride0, stride0, 1, 1, 1});
    }
    else {
        paramsShape = mean->getShapeAsVectorInt();
        paramsStrides = xRank == 4 ? std::vector<int>({(int)mean->strideAt(0), (int)mean->strideAt(1), (int)mean->strideAt(2), (int)mean->strideAt(3)}) : std::vector<int>({(int)mean->strideAt(0), (int)mean->strideAt(1), (int)mean->strideAt(2), (int)mean->strideAt(3), (int)mean->strideAt(4)});
    }

    std::vector<int> xStrides = {(int)input->strideAt(0),  (int)input->strideAt(1),  (int)input->strideAt(2),  (int)input->strideAt(3)};
    std::vector<int> zStrides = {(int)output->strideAt(0), (int)output->strideAt(1), (int)output->strideAt(2), (int)output->strideAt(3)};

    if(xRank > 4) { // 5D
        xStrides.push_back((int)input->strideAt(4));
        zStrides.push_back((int)output->strideAt(4));
    }

    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

     // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(x, format, dataType, xRank, xShape.data());
    else
        err = cudnnSetTensorNdDescriptor(x, dataType, xRank, xShape.data(), xStrides.data());
    if (err != 0) throw sd::cuda_exception::build("batchnormCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for input failed", err);

    // output descriptor
    cudnnTensorDescriptor_t z;
    cudnnCreateTensorDescriptor(&z);
    if(output->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(z, format, dataType, xRank, xShape.data());
    else
        err = cudnnSetTensorNdDescriptor(z, dataType, xRank, xShape.data(), zStrides.data());
    if (err != 0) throw sd::cuda_exception::build("batchnormCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for output failed", err);

    // mean, variance, gamma and beta descriptor, the same descriptor for all of them
    cudnnTensorDescriptor_t params;
    cudnnCreateTensorDescriptor(&params);
    if(mean->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(params, format, dataType, xRank, paramsShape.data());
    else
        err = cudnnSetTensorNdDescriptor(params, dataType, xRank, paramsShape.data(), paramsStrides.data());
    if (err != 0) throw sd::cuda_exception::build("batchnormCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for mean/variance/gamma/beta failed", err);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
    const double alpha64(1), beta64(0);
    const void* ptrAlpha = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* ptrBeta  = output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    NDArray::prepareSpecialUse({output}, {input, mean, variance, gamma, beta});

    // calculations
    err = cudnnBatchNormalizationForwardInference(*handle, isSpatialMode ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                                 ptrAlpha, ptrBeta,
                                                 x, input->specialBuffer(),
                                                 z, output->specialBuffer(),
                                                 params,
                                                 gamma->specialBuffer(), beta->specialBuffer(),
                                                 mean->specialBuffer(), variance->specialBuffer(), epsilon);

    if (err != 0) throw sd::cuda_exception::build("batchnormCUDNN: cudnnBatchNormalizationForwardInference failed", err);

    auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    if (cudaErr != 0)
        throw cuda_exception::build("batchnormCUDNN: cudaStreamSynchronize failed !", cudaErr);

    NDArray::registerSpecialUse({output}, {input, mean, variance, gamma, beta});
}

//////////////////////////////////////////////////////////////////////////
static void batchnormBpCUDNN(const LaunchContext* context,
                            const NDArray* input, const NDArray* mean, const NDArray* variance, const NDArray* gamma, const NDArray* gradO,
                                  NDArray* gradI, NDArray* gradG, NDArray* gradB,
                            const double epsilon, const bool isSpatialMode) {

    // input, gradO, gradI -> 4D:nchw, 5D:ncdhw
    // mean, variance, gamma, beta, gradM, gradV, gradG, gradB -> 1xCx1x1 for 4D and 1xCx1x1x1 for 5D for BATCHNORM_MODE_SPATIAL mode
    //                                                         -> 1xCxHxW for 4D and 1xCxDxHxW for 5D for BATCHNORM_MODE_PER_ACTIVATION mode

    const cudnnDataType_t dataType = cudnnDataType(input->dataType());

    const int xRank = input->rankOf();

    auto handle = reinterpret_cast<cudnnHandle_t *>(context->getCuDnnHandle());
    cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());
    if (err != 0) throw sd::cuda_exception::build("batchnormBpCUDNN: can't set stream for cuDNN", err);

    const std::vector<int> xShape = input->getShapeAsVectorInt();               // input and output have same shapes

    std::vector<int> paramsShape, paramsStrides;                                 // mean, variance, gamma and beta have same shapes
    if(isSpatialMode) { // 1xCx1x1
        const int iC = mean->lengthOf();
        const int stride0 = mean->strideAt(0);
        paramsShape   = xRank == 4 ? std::vector<int>({1, iC, 1, 1}) : std::vector<int>({1, iC, 1, 1, 1});
        paramsStrides = xRank == 4 ? std::vector<int>({iC*stride0, stride0, 1, 1}) : std::vector<int>({iC*stride0, stride0, 1, 1, 1});
    }
    else {
        paramsShape = mean->getShapeAsVectorInt();
        paramsStrides = xRank == 4 ? std::vector<int>({(int)mean->strideAt(0), (int)mean->strideAt(1), (int)mean->strideAt(2), (int)mean->strideAt(3)}) : std::vector<int>({(int)mean->strideAt(0), (int)mean->strideAt(1), (int)mean->strideAt(2), (int)mean->strideAt(3), (int)mean->strideAt(4)});
    }

    std::vector<int> xStrides = {(int)input->strideAt(0),  (int)input->strideAt(1),  (int)input->strideAt(2),  (int)input->strideAt(3)};
    std::vector<int> dxStrides = {(int)gradI->strideAt(0),  (int)gradI->strideAt(1),  (int)gradI->strideAt(2),  (int)gradI->strideAt(3)};
    std::vector<int> dzStrides = {(int)gradO->strideAt(0), (int)gradO->strideAt(1), (int)gradO->strideAt(2), (int)gradO->strideAt(3)};

    if(xRank > 4) { // 5D
        xStrides.push_back((int)input->strideAt(4));
        dxStrides.push_back((int)gradI->strideAt(4));
        dzStrides.push_back((int)gradO->strideAt(4));
    }

    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

     // input descriptor
    cudnnTensorDescriptor_t x;
    cudnnCreateTensorDescriptor(&x);
    if(input->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(x, format, dataType, xRank, xShape.data());
    else
        err = cudnnSetTensorNdDescriptor(x, dataType, xRank, xShape.data(), xStrides.data());
    if (err != 0) throw sd::cuda_exception::build("batchnormBpCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for input failed", err);

    // gradO descriptor
    cudnnTensorDescriptor_t dz;
    cudnnCreateTensorDescriptor(&dz);
    if(gradO->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(dz, format, dataType, xRank, xShape.data());
    else
        err = cudnnSetTensorNdDescriptor(dz, dataType, xRank, xShape.data(), dzStrides.data());
    if (err != 0) throw sd::cuda_exception::build("batchnormBpCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for gradO failed", err);

    // gradI descriptor
    cudnnTensorDescriptor_t dx;
    cudnnCreateTensorDescriptor(&dx);
    if(input->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(dx, format, dataType, xRank, xShape.data());
    else
        err = cudnnSetTensorNdDescriptor(dx, dataType, xRank, xShape.data(), dxStrides.data());
    if (err != 0) throw sd::cuda_exception::build("batchnormBpCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for gradI failed", err);

    // mean, variance, gamma, gradG and gradB descriptor, the same descriptor for all of them
    cudnnTensorDescriptor_t params;
    cudnnCreateTensorDescriptor(&params);
    if(mean->ews() == 1)
        err = cudnnSetTensorNdDescriptorEx(params, format, dataType, xRank, paramsShape.data());
    else
        err = cudnnSetTensorNdDescriptor(params, dataType, xRank, paramsShape.data(), paramsStrides.data());
    if (err != 0) throw sd::cuda_exception::build("batchnormBpCUDNN: cudnnSetTensorNdDescriptor/cudnnSetTensorNdDescriptorEx for mean/variance/gamma/gradG/gradB failed", err);

    // provide scaling parameters
    const float  alpha32(1), beta32(0);
     double alpha64(1), beta64(0);
    const void* ptrAlpha = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
    const void* ptrBeta  = input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32)  : reinterpret_cast<const void*>(&beta64);

    NDArray::prepareSpecialUse({gradI, gradG, gradB}, {input, mean, variance, gamma, gradO});

    // calculations
    // TODO: we can use cache here
    err = cudnnBatchNormalizationBackward(*handle, isSpatialMode ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                            ptrAlpha, ptrBeta, ptrAlpha, ptrBeta,
                                            x, input->specialBuffer(),
                                            dz, gradO->specialBuffer(),
                                            dx, gradI->specialBuffer(),
                                            params,
                                            gamma->specialBuffer(), gradG->specialBuffer(), gradB->specialBuffer(),
                                            epsilon,
                                            nullptr/*mean->specialBuffer()*/, nullptr/*variance->specialBuffer()*/);

    if (err != 0) throw sd::cuda_exception::build("batchnormBpCUDNN: cudnnBatchNormalizationBackward failed", err);

    auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
    if (cudaErr != 0)
        throw cuda_exception::build("batchnormBpCUDNN: cudaStreamSynchronize failed !", cudaErr);

    NDArray::registerSpecialUse({gradI, gradG, gradB}, {input, mean, variance, gamma, gradO});
}


//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batchnorm, ENGINE_CUDA) {

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
    REQUIRE_TRUE(numOfAxes <= inRank, 0, "BATCHNORM CUDNN op: too big number of input axes to normalize over, expected number should be less or equal to rank of input array, but got %i and %i correspondingly !", numOfAxes, inRank);

    // evaluate expected shape for mean, variance and gamma. These 3 arrays should have identical shapes
    // for example if input shape is {2,3,4,5,6} and axes = {1,3}, then expected shape would be {1,3,1,5,1}, and if axes = {3}, then expected shape would be {5}
    std::vector<Nd4jLong> expShape;
    if(numOfAxes == 1)
        expShape.push_back(input->sizeAt(axes[0]));
    else {      // get, for example, something like {1, inputDim1, 1, inputDim3, 1} if axes = {1, 3}
        expShape = std::vector<Nd4jLong>(inRank, 1);
        for(uint i = 0; i < numOfAxes; ++i)
            expShape[axes[i]] = input->sizeAt(axes[i]);
    }

    REQUIRE_TRUE(mean->isSameShape(expShape) , 0, "BATCHNORM CUDNN op: wrong shape of mean array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(mean).c_str());
    REQUIRE_TRUE(variance->isSameShape(expShape), 0, "BATCHNORM CUDNN op: wrong shape of variance array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(variance).c_str());
    if(gamma)
        REQUIRE_TRUE(gamma->isSameShape(expShape), 0, "BATCHNORM CUDNN op: wrong shape of gamma array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(gamma).c_str());
    if(beta)
        REQUIRE_TRUE(beta->isSameShape(expShape), 0, "BATCHNORM CUDNN op: wrong shape of beta array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(beta).c_str());

    // types of all input arrays should be the same
    for(int i = 1; i < block.width(); ++i)
        REQUIRE_TRUE(INPUT_VARIABLE(0)->dataType() == INPUT_VARIABLE(i)->dataType(), 0, "BATCHNORM CUDNN op: types of all input arrays should be the same !");

    // cudnn supports NCHW format only
    const bool needPermut = axes.size() == 1 && mean->lengthOf() == input->sizeAt(-1);

    if(needPermut) {    // if NHWC
        std::vector<int> perm = inRank == 4 ? std::vector<int>({0, 3, 1, 2}) : std::vector<int>({0, 4, 1, 2, 3});           // NHWC -> NCHW
        input  = new NDArray(input->permute(perm));
        output = new NDArray(output->permute(perm));
    }

    // cudnn requires gamma and beta to be non-nullptr
    if(!applyScale) {
        gamma = new NDArray(mean);
        *gamma = 1;
    }
    if(!applyOffset) {
        beta = new NDArray(mean);
        *beta = 0;
    }

    // calculations
    batchnormCUDNN(block.launchContext(), input, mean, variance, gamma, beta, output, epsilon, axes.size() == 1);

    if(needPermut) {
        delete input;
        delete output;
    }

    if(!applyScale)
        delete gamma;

    if(!applyOffset)
        delete beta;

    return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(batchnorm, ENGINE_CUDA) {

    const bool   applyScale  = (bool)INT_ARG(0);
    const bool   applyOffset = (bool)INT_ARG(1);

    NDArray* input     = INPUT_VARIABLE(0);
    NDArray* mean      = INPUT_VARIABLE(1);
    NDArray* variance  = INPUT_VARIABLE(2);
    NDArray* gamma     = applyScale  ? INPUT_VARIABLE(3) : nullptr;
    NDArray* beta      = applyOffset ? INPUT_VARIABLE(3 + (int)applyScale) : nullptr;

    const int numOfIntArgs = block.getIArguments()->size();
    const int xRank = input->rankOf();

    // *********************************** //
    if(xRank != 4 && xRank != 5)
        return false;

    // *********************************** //
    const bool badType = input->dataType() != DataType::DOUBLE && input->dataType() != DataType::FLOAT32 && input->dataType() != DataType::HALF;
    if(badType)
        return false;

    // *********************************** //
    // get axes args to normalize input array over
    std::vector<int> axes;
    if(numOfIntArgs > 2)
        for(int i = 2; i < numOfIntArgs; ++i)
            axes.push_back(INT_ARG(i));
    else
        axes.push_back(xRank-1);               // default dimension to reduce along is last dimension

    if(axes.size() != 1 && axes.size() != 3 && axes.size() != 4)
        return false;

    // *********************************** //
    bool allParamsHaveSameShapeAndStrides = shape::haveSameShapeAndStrides(mean->shapeInfo(), variance->shapeInfo());
    if(gamma)
        allParamsHaveSameShapeAndStrides &= shape::haveSameShapeAndStrides(mean->shapeInfo(), gamma->shapeInfo());
    if(beta)
        allParamsHaveSameShapeAndStrides &= shape::haveSameShapeAndStrides(mean->shapeInfo(), beta->shapeInfo());

    if(!allParamsHaveSameShapeAndStrides)
        return false;

    // *********************************** //
    bool isFormatGood = false;
    if(axes.size() == 1)
        isFormatGood = mean->lengthOf() == input->sizeAt(1) || mean->lengthOf() == input->sizeAt(-1);   // mean [C]
    else {
        auto inputShapeModif = input->getShapeAsVector();     // [dim0,dim1,dim2,dim3] 4D or [dim0,dim1,dim2,dim3,dim4]
        inputShapeModif[0] = 1;
        isFormatGood = mean->isSameShape(inputShapeModif);    // mean [1,dim1,dim2,dim3] 4D or [1,dim1,dim2,dim3,dim4]
    }
    if(!isFormatGood)
        return false;

    return true;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batchnorm_bp, ENGINE_CUDA) {

    NDArray* input    = INPUT_VARIABLE(0);
    NDArray* mean     = INPUT_VARIABLE(1);
    NDArray* variance = INPUT_VARIABLE(2);
    NDArray* gamma    = nullptr;
    NDArray* beta     = nullptr;
    NDArray* gradO     = INPUT_VARIABLE(block.width() - 1);    // next epsilon

    NDArray* gradI = OUTPUT_VARIABLE(0);
    NDArray* gradM = OUTPUT_VARIABLE(1);
    NDArray* gradV = OUTPUT_VARIABLE(2);
    NDArray* gradG = nullptr;
    NDArray* gradB = nullptr;

    const bool   applyScale  = (bool)INT_ARG(0);
    const bool   applyOffset = (bool)INT_ARG(1);
    const float  epsilon     = T_ARG(0);

    if(applyScale) {
        gamma = INPUT_VARIABLE(3);
        gradG  = OUTPUT_VARIABLE(3);
    }
    if(applyOffset) {
        beta = INPUT_VARIABLE(3 + (int)applyScale);
        gradB = OUTPUT_VARIABLE(3 + (int)applyScale);
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
    REQUIRE_TRUE(numOfAxes <= inRank, 0, "BATCHNORM_BP CUDNN op: too big number of input axes to normalize over, expected number should be less or equal to rank of input array, but got %i and %i correspondingly !", numOfAxes, inRank);

    // evaluate expected shape for mean, variance and gamma. These 3 arrays should have identical shapes
    // for example if input shape is {2,3,4,5,6} and axes = {1,3}, then expected shape would be {1,3,1,5,1}, and if axes = {3}, then expected shape would be {5}
    std::vector<Nd4jLong> expShape;
    if(numOfAxes == 1)
        expShape.push_back(input->sizeAt(axes[0]));
    else {      // get, for example, something like {1, inputDim1, 1, inputDim3, 1} if axes = {1, 3}
        expShape = std::vector<Nd4jLong>(inRank, 1);
        for(uint i = 0; i < numOfAxes; ++i)
            expShape[axes[i]] = input->sizeAt(axes[i]);
    }

    REQUIRE_TRUE(mean->isSameShape(expShape), 0, "BATCHNORM_BP CUDNN op: wrong shape of mean array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(mean).c_str());
    REQUIRE_TRUE(variance->isSameShape(expShape), 0, "BATCHNORM_BP CUDNN op: wrong shape of variance array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(variance).c_str());
    if(gamma)
        REQUIRE_TRUE(gamma->isSameShape(expShape), 0, "BATCHNORM_BP CUDNN op: wrong shape of gamma array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(gamma).c_str());
    if(beta)
        REQUIRE_TRUE(beta->isSameShape(expShape), 0, "BATCHNORM_BP CUDNN op: wrong shape of beta array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(beta).c_str());

    REQUIRE_TRUE(input->isSameShape(gradO), 0, "BATCHNORM_BP CUDNN op: wrong shape of output gradients array, expected is %s, but got %s instead !", ShapeUtils::shapeAsString(input).c_str(), ShapeUtils::shapeAsString(gradO).c_str());

    // types of all input arrays should be the same (except gradO)
    for(int i = 1; i < block.width() - 2; ++i)
        REQUIRE_TRUE(INPUT_VARIABLE(0)->dataType() == INPUT_VARIABLE(i)->dataType(), 0, "BATCHNORM_BP CUDNN op: types of arrays (input, mean, variance, gamma, beta) should be the same !");

    // cudnn supports NCHW format only
    const bool needPermut = axes.size() == 1 && mean->lengthOf() != input->sizeAt(1);

    if(needPermut) {    // if NHWC
        std::vector<int> perm = inRank == 4 ? std::vector<int>({0, 3, 1, 2}) : std::vector<int>({0, 4, 1, 2, 3});           // NHWC -> NCHW
        input = new NDArray(input->permute(perm));
        gradO = new NDArray(gradO->permute(perm));
        gradI = new NDArray(gradI->permute(perm));
    }

    // cudnn requires gamma, gradG, gradB to be non-nullptr
    if(!applyScale) {
        gamma = new NDArray(mean);
        gradG = new NDArray(mean);
        *gamma = 1;
    }
    if(!applyOffset)
        gradB = new NDArray(mean);

    // calculations
    batchnormBpCUDNN(block.launchContext(), input, mean, variance, gamma, gradO,   gradI, gradG, gradB, epsilon, axes.size() == 1);

    *gradM = 0;      // put zeros so far
    *gradV = 0;      // put zeros so far

    if(needPermut) {
        delete input;
        delete gradO;
        delete gradI;
    }

    if(!applyScale) {
        delete gamma;
        delete gradG;
    }

    if(!applyOffset)
        delete gradB;

    return Status::OK();

}

PLATFORM_CHECK(batchnorm_bp, ENGINE_CUDA) {

    NDArray* input    = INPUT_VARIABLE(0);
    NDArray* mean     = INPUT_VARIABLE(1);
    NDArray* variance = INPUT_VARIABLE(2);
    NDArray* gamma    = nullptr;
    NDArray* beta     = nullptr;
    NDArray* gradO    = INPUT_VARIABLE(block.width() - 1);    // next epsilon

    NDArray* gradI = OUTPUT_VARIABLE(0);
    NDArray* gradM = OUTPUT_VARIABLE(1);
    NDArray* gradV = OUTPUT_VARIABLE(2);
    NDArray* gradG = nullptr;
    NDArray* gradB = nullptr;

    const int numOfIntArgs = block.getIArguments()->size();
    const int xRank = input->rankOf();

    // *********************************** //
    if(xRank != 4 && xRank != 5)
        return false;

    // *********************************** //
    const bool badType = input->dataType() != DataType::DOUBLE && input->dataType() != DataType::FLOAT32 && input->dataType() != DataType::HALF;
    if(badType)
        return false;

    // *********************************** //
    // get axes args to normalize input array over
    std::vector<int> axes;
    if(numOfIntArgs > 2)
        for(int i = 2; i < numOfIntArgs; ++i)
            axes.push_back(INT_ARG(i));
    else
        axes.push_back(xRank-1);               // default dimension to reduce along is last dimension

    if(axes.size() != 1 && axes.size() != 3 && axes.size() != 4)
        return false;

    // *********************************** //
    bool allParamsHaveSameShapeAndStrides = shape::haveSameShapeAndStrides(mean->shapeInfo(), variance->shapeInfo());
    if(gamma)
        allParamsHaveSameShapeAndStrides &= shape::haveSameShapeAndStrides(mean->shapeInfo(), gamma->shapeInfo());
    if(gradG)
        allParamsHaveSameShapeAndStrides &= shape::haveSameShapeAndStrides(mean->shapeInfo(), gradG->shapeInfo());
    if(gradB)
        allParamsHaveSameShapeAndStrides &= shape::haveSameShapeAndStrides(mean->shapeInfo(), gradB->shapeInfo());

    if(!allParamsHaveSameShapeAndStrides)
        return false;

    // *********************************** //
    bool isFormatGood = false;
    if(axes.size() == 1)
        isFormatGood = mean->lengthOf() == input->sizeAt(1) || mean->lengthOf() == input->sizeAt(-1);   // mean [C]
    else {
        auto inputShapeModif = input->getShapeAsVector();     // [dim0,dim1,dim2,dim3] 4D or [dim0,dim1,dim2,dim3,dim4]
        inputShapeModif[0] = 1;
        isFormatGood = mean->isSameShape(inputShapeModif);    // mean [1,dim1,dim2,dim3] 4D or [1,dim1,dim2,dim3,dim4]
    }
    if(!isFormatGood)
        return false;

    return true;
}


}
}
}
