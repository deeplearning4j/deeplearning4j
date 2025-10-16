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

#include <ops/declarable/helpers/convolutions.h>

#include "cudnnUtils.h"

namespace sd {
namespace ops {
namespace platforms {

//////////////////////////////////////////////////////////////////////////
static void batchnormCUDNN(const LaunchContext* context, NDArray* input, NDArray* mean,
                           NDArray* variance, NDArray* gamma, NDArray* beta, NDArray* output,
                           const double epsilon, const bool isSpatialMode) {
  // input, output -> 4D:nchw, 5D:ncdhw
  // mean, variance, gamma, beta -> 1xCx1x1 for 4D and 1xCx1x1x1 for 5D for BATCHNORM_MODE_SPATIAL mode
  //                             -> 1xCxHxW for 4D and 1xCxDxHxW for 5D for BATCHNORM_MODE_PER_ACTIVATION mode

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  const LongType xRank = input->rankOf();

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE(cudnnSetStream(*handle, *context->getCudaStream()));

  const std::vector<int> xShape = input->getShapeAsVectorInt();  // input and output have same shapes

  std::vector<int> paramsShape, paramsStrides;  // mean, variance, gamma and beta have same shapes
  if (isSpatialMode) {                          // 1xCx1x1
    const int iC = static_cast<int>(mean->lengthOf());
    const int stride0 = static_cast<int>(mean->strideAt(0));
    paramsShape = xRank == 4 ? std::vector<int>({1, iC, 1, 1}) : std::vector<int>({1, iC, 1, 1, 1});
    paramsStrides = xRank == 4 ? std::vector<int>({iC * stride0, stride0, 1, 1})
                               : std::vector<int>({iC * stride0, stride0, 1, 1, 1});
  } else {
    auto* meanShapePtr = mean->getShapeAsVector();
    paramsShape = std::vector<int>(meanShapePtr->begin(), meanShapePtr->end());
    delete meanShapePtr;
    paramsStrides = xRank == 4
                    ? std::vector<int>({static_cast<int>(mean->strideAt(0)), static_cast<int>(mean->strideAt(1)), static_cast<int>(mean->strideAt(2)),
                                        static_cast<int>(mean->strideAt(3))})
                    : std::vector<int>({static_cast<int>(mean->strideAt(0)), static_cast<int>(mean->strideAt(1)), static_cast<int>(mean->strideAt(2)),
                                        static_cast<int>(mean->strideAt(3)), static_cast<int>(mean->strideAt(4))});
  }

  std::vector<int> xStrides = {static_cast<int>(input->strideAt(0)), static_cast<int>(input->strideAt(1)), static_cast<int>(input->strideAt(2)),
                               static_cast<int>(input->strideAt(3))};
  std::vector<int> zStrides = {static_cast<int>(output->strideAt(0)), static_cast<int>(output->strideAt(1)), static_cast<int>(output->strideAt(2)),
                               static_cast<int>(output->strideAt(3))};
  if (xRank > 4) {  // 5D
    xStrides.push_back((LongType)input->strideAt(4));
    zStrides.push_back((LongType)output->strideAt(4));
  }

  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

  // input descriptor
  CudnnTensor x;
    x.set(dataType, xRank, xShape.data(), xStrides.data());

  // output descriptor
  CudnnTensor z;
    z.set(dataType, xRank, xShape.data(), zStrides.data());

  // mean, variance, gamma and beta descriptor, the same descriptor for all of them
  CudnnTensor params;
    params.set(dataType, xRank, paramsShape.data(), paramsStrides.data());

  // provide scaling parameters
  const float alpha32(1), beta32(0);
  const double alpha64(1), beta64(0);
  const void* ptrAlpha =
      output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* ptrBeta =
      output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input, mean, variance, gamma, beta});

  // calculations
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnBatchNormalizationForwardInference),
      cudnnBatchNormalizationForwardInference(
          *handle, isSpatialMode ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION, ptrAlpha, ptrBeta, x,
          input->specialBuffer(), z, output->specialBuffer(), params, gamma->specialBuffer(), beta->specialBuffer(),
          mean->specialBuffer(), variance->specialBuffer(), epsilon));

  auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  if (cudaErr != 0) throw cuda_exception::build("batchnormCUDNN: cudaStreamSynchronize failed !", cudaErr);

  NDArray::registerSpecialUse({output}, {input, mean, variance, gamma, beta});
}

//////////////////////////////////////////////////////////////////////////
static void batchnormBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* mean,
                             NDArray* variance, NDArray* gamma, NDArray* gradO, NDArray* gradI,
                             NDArray* gradG, NDArray* gradB, const double epsilon, const bool isSpatialMode) {
  // input, gradO, gradI -> 4D:nchw, 5D:ncdhw
  // mean, variance, gamma, beta, gradM, gradV, gradG, gradB -> 1xCx1x1 for 4D and 1xCx1x1x1 for 5D for
  // BATCHNORM_MODE_SPATIAL mode
  //                                                         -> 1xCxHxW for 4D and 1xCxDxHxW for 5D for
  //                                                         BATCHNORM_MODE_PER_ACTIVATION mode

  const cudnnDataType_t dataType = cudnnDataType(input->dataType());

  const int xRank = input->rankOf();

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  cudnnStatus_t err = cudnnSetStream(*handle, *context->getCudaStream());

  const std::vector<int> xShape = input->getShapeAsVectorInt();  // input and output have same shapes

  std::vector<int> paramsShape, paramsStrides;  // mean, variance, gamma and beta have same shapes
  if (isSpatialMode) {                          // 1xCx1x1
    const int iC = static_cast<int>(mean->lengthOf());
    const int stride0 = static_cast<int>(mean->strideAt(0));
    paramsShape = xRank == 4 ? std::vector<int>({1, iC, 1, 1}) : std::vector<int>({1, iC, 1, 1, 1});
    paramsStrides = xRank == 4 ? std::vector<int>({iC * stride0, stride0, 1, 1})
                               : std::vector<int>({iC * stride0, stride0, 1, 1, 1});
  } else {
    auto* meanShapePtr = mean->getShapeAsVector();
    paramsShape = std::vector<int>(meanShapePtr->begin(), meanShapePtr->end());
    delete meanShapePtr;
    paramsStrides = xRank == 4
                    ? std::vector<int>({static_cast<int>(mean->strideAt(0)), static_cast<int>(mean->strideAt(1)), static_cast<int>(mean->strideAt(2)),
                                        static_cast<int>(mean->strideAt(3))})
                    : std::vector<int>({static_cast<int>(mean->strideAt(0)), static_cast<int>(mean->strideAt(1)), static_cast<int>(mean->strideAt(2)),
                                        static_cast<int>(mean->strideAt(3)), static_cast<int>(mean->strideAt(4))});
  }

  std::vector<int> xStrides = {static_cast<int>(input->strideAt(0)), static_cast<int>(input->strideAt(1)), static_cast<int>(input->strideAt(2)),
                               static_cast<int>(input->strideAt(3))};
  std::vector<int> dxStrides = {static_cast<int>(gradI->strideAt(0)), static_cast<int>(gradI->strideAt(1)), static_cast<int>(gradI->strideAt(2)),
                                static_cast<int>(gradI->strideAt(3))};
  std::vector<int> dzStrides = {static_cast<int>(gradO->strideAt(0)), static_cast<int>(gradO->strideAt(1)), static_cast<int>(gradO->strideAt(2)),
                                static_cast<int>(gradO->strideAt(3))};

  if (xRank > 4) {  // 5D
    xStrides.push_back(static_cast<int>(input->strideAt(4)));
    dxStrides.push_back(static_cast<int>(gradI->strideAt(4)));
    dzStrides.push_back(static_cast<int>(gradO->strideAt(4)));
  }
  cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;

  // input descriptor
  CudnnTensor x;

    x.set(dataType, xRank, xShape.data(), xStrides.data());

  // gradO descriptor
  CudnnTensor dz;
    dz.set(dataType, xRank, xShape.data(), dzStrides.data());

  // gradI descriptor
  CudnnTensor dx;
    dx.set(dataType, xRank, xShape.data(), dxStrides.data());

  // mean, variance, gamma, gradG and gradB descriptor, the same descriptor for all of them
  CudnnTensor params;
    params.set(dataType, xRank, paramsShape.data(), paramsStrides.data());

  // provide scaling parameters
  const float alpha32(1), beta32(0);
  double alpha64(1), beta64(0);
  const void* ptrAlpha =
      input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* ptrBeta =
      input->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI, gradG, gradB}, {input, mean, variance, gamma, gradO});

  // calculations
  // TODO: we can use cache here
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnBatchNormalizationBackward),
      cudnnBatchNormalizationBackward(*handle, isSpatialMode ? CUDNN_BATCHNORM_SPATIAL : CUDNN_BATCHNORM_PER_ACTIVATION,
                                      ptrAlpha, ptrBeta, ptrAlpha, ptrBeta, x, input->specialBuffer(), dz,
                                      gradO->specialBuffer(), dx, gradI->specialBuffer(), params,
                                      gamma->specialBuffer(), gradG->specialBuffer(), gradB->specialBuffer(), epsilon,
                                      nullptr /*mean->specialBuffer()*/, nullptr /*variance->specialBuffer()*/));

  auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  if (cudaErr != 0) throw cuda_exception::build("batchnormBpCUDNN: cudaStreamSynchronize failed !", cudaErr);

  NDArray::registerSpecialUse({gradI, gradG, gradB}, {input, mean, variance, gamma, gradO});
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batchnorm, ENGINE_CUDA) {
  auto input = INPUT_VARIABLE(0);
  auto mean = INPUT_VARIABLE(1);
  auto variance = INPUT_VARIABLE(2);

  NDArray* gamma = nullptr;
  NDArray* beta = nullptr;

  auto output = OUTPUT_VARIABLE(0);

  const bool applyScale = (bool)INT_ARG(0);
  const bool applyOffset = (bool)INT_ARG(1);
  const double epsilon = T_ARG(0);

  if (applyScale) gamma = INPUT_VARIABLE(3);
  if (applyOffset) beta = INPUT_VARIABLE(3 + (int)applyScale);

  const int numOfIntArgs = block.getIArguments()->size();
  const int inRank = input->rankOf();

  // get axes args to normalize input array over
  std::vector<int> axes;
  if (numOfIntArgs > 2)
    for (int i = 2; i < numOfIntArgs; ++i) axes.push_back(INT_ARG(i));
  else
    axes.push_back(inRank - 1);  // default dimension to reduce along is last dimension

  const int numOfAxes = axes.size();
  REQUIRE_TRUE(numOfAxes <= inRank, 0,
               "BATCHNORM CUDNN op: too big number of input axes to normalize over, expected number should be less or "
               "equal to rank of input array, but got %i and %i correspondingly !",
               numOfAxes, inRank);

  // evaluate expected shape for mean, variance and gamma. These 3 arrays should have identical shapes
  // for example if input shape is {2,3,4,5,6} and axes = {1,3}, then expected shape would be {1,3,1,5,1}, and if axes =
  // {3}, then expected shape would be {5}
  std::vector<LongType> expShape;
  if (numOfAxes == 1)
    expShape.push_back(input->sizeAt(axes[0]));
  else {  // get, for example, something like {1, inputDim1, 1, inputDim3, 1} if axes = {1, 3}
    expShape = std::vector<LongType>(inRank, 1);
    for (LongType i = 0; i < numOfAxes; ++i) expShape[axes[i]] = input->sizeAt(axes[i]);
  }

  REQUIRE_TRUE(mean->isSameShape(expShape), 0,
               "BATCHNORM CUDNN op: wrong shape of mean array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(mean).c_str());
  REQUIRE_TRUE(variance->isSameShape(expShape), 0,
               "BATCHNORM CUDNN op: wrong shape of variance array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(variance).c_str());
  if (gamma)
  REQUIRE_TRUE(gamma->isSameShape(expShape), 0,
               "BATCHNORM CUDNN op: wrong shape of gamma array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(gamma).c_str());
  if (beta)
  REQUIRE_TRUE(beta->isSameShape(expShape), 0,
               "BATCHNORM CUDNN op: wrong shape of beta array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(beta).c_str());

  // types of all input arrays should be the same
  for (int i = 1; i < block.width(); ++i)
  REQUIRE_TRUE(INPUT_VARIABLE(0)->dataType() == INPUT_VARIABLE(i)->dataType(), 0,
               "BATCHNORM CUDNN op: types of all input arrays should be the same !");

  // cudnn supports NCHW format only
  const bool needPermut = axes.size() == 1 && mean->lengthOf() == input->sizeAt(-1);

  std::unique_ptr<NDArray> tmpGamma = {}, tmpBeta = {}, tmpInput = {}, tmpOutput = {};
  if (needPermut) {  // if NHWC
    std::vector<LongType> perm =
        inRank == 4 ? std::vector<LongType>({0, 3, 1, 2}) : std::vector<LongType>({0, 4, 1, 2, 3});  // NHWC -> NCHW
    tmpInput.reset(new NDArray(input->permute(perm)));
    tmpOutput.reset(new NDArray(output->permute(perm)));
    input = tmpInput.get();
    output = tmpOutput.get();
  }

  // cudnn requires gamma and beta to be non-nullptr
  if (!applyScale) {
    tmpGamma.reset(new NDArray(mean));
    gamma = tmpGamma.get();
    *gamma = 1;
  }
  if (!applyOffset) {
    tmpBeta.reset(new NDArray(mean));
    beta = tmpBeta.get();
    *beta = 0;
  }

  // calculations
  batchnormCUDNN(block.launchContext(), input, mean, variance, gamma, beta, output, epsilon, axes.size() == 1);

  return Status::OK;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_CHECK(batchnorm, ENGINE_CUDA) {
  const bool applyScale = (bool)INT_ARG(0);
  const bool applyOffset = (bool)INT_ARG(1);

  NDArray* input = INPUT_VARIABLE(0);
  NDArray* mean = INPUT_VARIABLE(1);
  NDArray* variance = INPUT_VARIABLE(2);
  NDArray* gamma = applyScale ? INPUT_VARIABLE(3) : nullptr;
  NDArray* beta = applyOffset ? INPUT_VARIABLE(3 + (int)applyScale) : nullptr;

  const int numOfIntArgs = block.getIArguments()->size();
  const int xRank = input->rankOf();

  // *********************************** //
  // get axes args to normalize input array over
  std::vector<int> axes;
  if (numOfIntArgs > 2)
    for (int i = 2; i < numOfIntArgs; ++i) axes.push_back(INT_ARG(i));
  else
    axes.push_back(xRank - 1);  // default dimension to reduce along is last dimension

  Requirements req("CUDNN BATCHNORM OP");
  req.expectIn(makeInfoVariable(xRank, RANK_MSG_INPUT0), {4, 5}) &&
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
               {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(axes.size(), "axes.size()"), {1, 3, 4}) &&
  req.expect(
      makeShapeInfoVariable(mean, SHAPE_MSG_INPUT1), makeShapeInfoVariable(variance, SHAPE_MSG_INPUT2),
      [](const decltype(mean)& l, const decltype(variance)& r) {
        return shape::haveSameShapeAndStrides(l->shapeInfo(), r->shapeInfo());
      },
      EXPECTED_EQ_MSG);
  if (gamma) {
    req.expect(
        makeShapeInfoVariable(gamma, SHAPE_MSG_INPUT_ "#gamma"), makeShapeInfoVariable(mean, SHAPE_MSG_INPUT1),
        [](const decltype(gamma)& l, const decltype(mean)& r) {
          return shape::haveSameShapeAndStrides(l->shapeInfo(), r->shapeInfo());
        },
        EXPECTED_EQ_MSG);
  }
  if (beta) {
    req.expect(
        makeShapeInfoVariable(beta, SHAPE_MSG_INPUT_ "#beta"), makeShapeInfoVariable(mean, SHAPE_MSG_INPUT1),
        [](const decltype(beta)& l, const decltype(mean)& r) {
          return shape::haveSameShapeAndStrides(l->shapeInfo(), r->shapeInfo());
        },
        EXPECTED_EQ_MSG);
  }
  if (axes.size() == 1) {
    req.expectIn(makeInfoVariable(mean->lengthOf(), LENGTH_MSG_INPUT1), {-1, 1});
  } else {
    auto* inputShapeModifPtr = input->getShapeAsVector();
    std::vector<LongType> inputShapeModif = *inputShapeModifPtr;
    delete inputShapeModifPtr;
    inputShapeModif[0] = 1;
    // mean [1,dim1,dim2,dim3] 4D or [1,dim1,dim2,dim3,dim4]
    req.expect(
        makeShapeInfoVariable(mean, SHAPE_MSG_INPUT1),
        makeShapeInfoVariable(inputShapeModif, SHAPE_MSG_INPUT_ "#expect"),
        [](const decltype(mean)& l, const decltype(inputShapeModif)& r) { return l->isSameShape(r); }, EXPECTED_EQ_MSG);
  }
  req.logTheSuccess();
  return req;
}

//////////////////////////////////////////////////////////////////////////
PLATFORM_IMPL(batchnorm_bp, ENGINE_CUDA) {
  NDArray* input = INPUT_VARIABLE(0);
  NDArray* mean = INPUT_VARIABLE(1);
  NDArray* variance = INPUT_VARIABLE(2);
  NDArray* gamma = nullptr;
  NDArray* beta = nullptr;
  NDArray* gradO = INPUT_VARIABLE(block.width() - 1);  // next epsilon

  NDArray* gradI = OUTPUT_VARIABLE(0);
  NDArray* gradM = OUTPUT_VARIABLE(1);
  NDArray* gradV = OUTPUT_VARIABLE(2);
  NDArray* gradG = nullptr;
  NDArray* gradB = nullptr;

  const bool applyScale = (bool)INT_ARG(0);
  const bool applyOffset = (bool)INT_ARG(1);
  const float epsilon = T_ARG(0);

  if (applyScale) {
    gamma = INPUT_VARIABLE(3);
    gradG = OUTPUT_VARIABLE(3);
  }
  if (applyOffset) {
    beta = INPUT_VARIABLE(3 + (int)applyScale);
    gradB = OUTPUT_VARIABLE(3 + (int)applyScale);
  }

  const int numOfIntArgs = block.getIArguments()->size();
  const int inRank = input->rankOf();

  // get axes args to normalize input array over
  std::vector<int> axes;
  if (numOfIntArgs > 2)
    for (int i = 2; i < numOfIntArgs; ++i) axes.push_back(INT_ARG(i));
  else
    axes.push_back(inRank - 1);  // default dimension to reduce along is last dimension

  const int numOfAxes = axes.size();
  REQUIRE_TRUE(numOfAxes <= inRank, 0,
               "BATCHNORM_BP CUDNN op: too big number of input axes to normalize over, expected number should be less "
               "or equal to rank of input array, but got %i and %i correspondingly !",
               numOfAxes, inRank);

  // evaluate expected shape for mean, variance and gamma. These 3 arrays should have identical shapes
  // for example if input shape is {2,3,4,5,6} and axes = {1,3}, then expected shape would be {1,3,1,5,1}, and if axes =
  // {3}, then expected shape would be {5}
  std::vector<LongType> expShape;
  if (numOfAxes == 1)
    expShape.push_back(input->sizeAt(axes[0]));
  else {  // get, for example, something like {1, inputDim1, 1, inputDim3, 1} if axes = {1, 3}
    expShape = std::vector<LongType>(inRank, 1);
    for (LongType i = 0; i < numOfAxes; ++i) expShape[axes[i]] = input->sizeAt(axes[i]);
  }

  REQUIRE_TRUE(mean->isSameShape(expShape), 0,
               "BATCHNORM_BP CUDNN op: wrong shape of mean array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(mean).c_str());
  REQUIRE_TRUE(variance->isSameShape(expShape), 0,
               "BATCHNORM_BP CUDNN op: wrong shape of variance array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(variance).c_str());
  if (gamma)
  REQUIRE_TRUE(gamma->isSameShape(expShape), 0,
               "BATCHNORM_BP CUDNN op: wrong shape of gamma array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(gamma).c_str());
  if (beta)
  REQUIRE_TRUE(beta->isSameShape(expShape), 0,
               "BATCHNORM_BP CUDNN op: wrong shape of beta array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(expShape).c_str(), ShapeUtils::shapeAsString(beta).c_str());

  REQUIRE_TRUE(input->isSameShape(gradO), 0,
               "BATCHNORM_BP CUDNN op: wrong shape of output gradients array, expected is %s, but got %s instead !",
               ShapeUtils::shapeAsString(input).c_str(), ShapeUtils::shapeAsString(gradO).c_str());

  // types of all input arrays should be the same (except gradO)
  for (int i = 1; i < block.width() - 2; ++i)
  REQUIRE_TRUE(INPUT_VARIABLE(0)->dataType() == INPUT_VARIABLE(i)->dataType(), 0,
               "BATCHNORM_BP CUDNN op: types of arrays (input, mean, variance, gamma, beta) should be the same !");

  // cudnn supports NCHW format only
  const bool needPermut = axes.size() == 1 && mean->lengthOf() != input->sizeAt(1);
  std::unique_ptr<NDArray> tmpGamma = {}, tmpGradG = {}, tmpGradB = {}, tmpInput = {}, tmpGradI = {}, tmpGradO = {};
  if (needPermut) {  // if NHWC
    std::vector<LongType> perm =
        inRank == 4 ? std::vector<LongType>({0, 3, 1, 2}) : std::vector<LongType>({0, 4, 1, 2, 3});  // NHWC -> NCHW
    tmpInput.reset(new NDArray(input->permute(perm)));
    tmpGradO.reset(new NDArray(gradO->permute(perm)));
    tmpGradI.reset(new NDArray(gradI->permute(perm)));
    input = tmpInput.get();
    gradO = tmpGradO.get();
    gradI = tmpGradI.get();
  }

  // cudnn requires gamma, gradG, gradB to be non-nullptr
  if (!applyScale) {
    tmpGamma.reset(new NDArray(mean));
    tmpGradG.reset(new NDArray(mean));
    gamma = tmpGamma.get();
    gradG = tmpGradG.get();
    *gamma = 1;
  }
  if (!applyOffset) {
    tmpGradB.reset(new NDArray(mean));
    gradB = tmpGradB.get();
  }

  // calculations
  batchnormBpCUDNN(block.launchContext(), input, mean, variance, gamma, gradO, gradI, gradG, gradB, epsilon,
                   axes.size() == 1);

  *gradM = 0;  // put zeros so far
  *gradV = 0;  // put zeros so far

  return Status::OK;
}

PLATFORM_CHECK(batchnorm_bp, ENGINE_CUDA) {
  NDArray* input = INPUT_VARIABLE(0);
  NDArray* mean = INPUT_VARIABLE(1);
  NDArray* variance = INPUT_VARIABLE(2);
  NDArray* gamma = nullptr;
  NDArray* beta = nullptr;
  NDArray* gradO = INPUT_VARIABLE(block.width() - 1);  // next epsilon

  NDArray* gradI = OUTPUT_VARIABLE(0);
  NDArray* gradM = OUTPUT_VARIABLE(1);
  NDArray* gradV = OUTPUT_VARIABLE(2);
  NDArray* gradG = nullptr;
  NDArray* gradB = nullptr;

  const int numOfIntArgs = block.getIArguments()->size();
  const int xRank = input->rankOf();

  // *********************************** //
  // get axes args to normalize input array over
  std::vector<int> axes;
  if (numOfIntArgs > 2)
    for (int i = 2; i < numOfIntArgs; ++i) axes.push_back(INT_ARG(i));
  else
    axes.push_back(xRank - 1);  // default dimension to reduce along is last dimension

  Requirements req("CUDNN BATCHNORM_BP OP");
  req.expectIn(makeInfoVariable(xRank, RANK_MSG_INPUT0), {4, 5}) &&
  req.expectIn(makeInfoVariable(input->dataType(), TYPE_MSG_INPUT0),
               {HALF, FLOAT32, DOUBLE}) &&
  req.expectIn(makeInfoVariable(axes.size(), "axes.size()"), {1, 3, 4}) &&
  req.expect(
      makeShapeInfoVariable(mean, SHAPE_MSG_INPUT1), makeShapeInfoVariable(variance, SHAPE_MSG_INPUT2),
      [](const decltype(mean)& l, const decltype(variance)& r) {
        return shape::haveSameShapeAndStrides(l->shapeInfo(), r->shapeInfo());
      },
      EXPECTED_EQ_MSG);
  if (gamma) {
    req.expect(
        makeShapeInfoVariable(gamma, SHAPE_MSG_INPUT_ "#gamma"), makeShapeInfoVariable(mean, SHAPE_MSG_INPUT1),
        [](const decltype(gamma)& l, const decltype(mean)& r) {
          return shape::haveSameShapeAndStrides(l->shapeInfo(), r->shapeInfo());
        },
        EXPECTED_EQ_MSG);
  }
  if (gradG) {
    req.expect(
        makeShapeInfoVariable(gradG, SHAPE_MSG_INPUT_ "#gradG"), makeShapeInfoVariable(mean, SHAPE_MSG_INPUT1),
        [](const decltype(gradG)& l, const decltype(mean)& r) {
          return shape::haveSameShapeAndStrides(l->shapeInfo(), r->shapeInfo());
        },
        EXPECTED_EQ_MSG);
  }
  if (gradB) {
    req.expect(
        makeShapeInfoVariable(gradB, SHAPE_MSG_INPUT_ "#gradB"), makeShapeInfoVariable(mean, SHAPE_MSG_INPUT1),
        [](const decltype(gradB)& l, const decltype(mean)& r) {
          return shape::haveSameShapeAndStrides(l->shapeInfo(), r->shapeInfo());
        },
        EXPECTED_EQ_MSG);
  }
  if (axes.size() == 1) {
    //     isFormatGood = mean->lengthOf() == input->sizeAt(1) || mean->lengthOf() == input->sizeAt(-1);   // mean [C]
    req.expectIn(makeInfoVariable(mean->lengthOf(), LENGTH_MSG_INPUT1), {-1, 1});
  } else {
    auto* inputShapeModifPtr = input->getShapeAsVector();
    std::vector<LongType> inputShapeModif = *inputShapeModifPtr;
    delete inputShapeModifPtr;
    inputShapeModif[0] = 1;
    //     isFormatGood = mean->isSameShape(inputShapeModif);    // mean [1,dim1,dim2,dim3] 4D or
    //     [1,dim1,dim2,dim3,dim4]
    req.expect(
        makeShapeInfoVariable(mean, SHAPE_MSG_INPUT1),
        makeShapeInfoVariable(inputShapeModif, SHAPE_MSG_INPUT_ "#expect"),
        [](const decltype(mean)& l, const decltype(inputShapeModif)& r) { return l->isSameShape(r); }, EXPECTED_EQ_MSG);
  }
  req.logTheSuccess();
  return req;
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
