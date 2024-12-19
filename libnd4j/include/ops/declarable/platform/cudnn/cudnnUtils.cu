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
std::tuple<std::unique_ptr<NDArray>, std::unique_ptr<NDArray>> checkConv2dCUDNNPadAsymmetric(
    NDArray* input, NDArray* gradI, const int iH, const int iW, const int oH, const int oW, const int kH,
    const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW,
    const bool isNCHW) {
  const auto pHsum = ((oH - 1) * sH + ((kH - 1) * dH + 1) - iH);
  const auto pWsum = ((oW - 1) * sW + ((kW - 1) * dW + 1) - iW);

  const bool isPHasymm = pH != (pHsum - pH);
  const bool isPWasymm = pW != (pWsum - pW);
  std::unique_ptr<NDArray> uNewInput = {}, uNewGradI = {};

  if (!isPHasymm && !isPWasymm) return std::make_tuple(std::move(uNewInput), std::move(uNewGradI));

  std::vector<LongType> newShape = input->getShapeAsVector();

  const int iHposition = isNCHW ? 2 : 1;

  if (isPHasymm) newShape[iHposition] += 1;
  if (isPWasymm) newShape[iHposition + 1] += 1;

  uNewInput.reset(new NDArray(input->ordering(), newShape, input->dataType(), input->getContext()));

  if (isNCHW)
    (*uNewInput)({0, 0, 0, 0, 0, input->sizeAt(2), 0, input->sizeAt(3)}).assign(input);
  else
    (*uNewInput)({0, 0, 0, input->sizeAt(1), 0, input->sizeAt(2), 0, 0}).assign(input);

  if (gradI != nullptr)
    uNewGradI.reset(new NDArray(gradI->ordering(), newShape, gradI->dataType(), gradI->getContext()));
  return std::make_tuple(std::move(uNewInput), std::move(uNewGradI));
}

//////////////////////////////////////////////////////////////////////////
std::tuple<std::unique_ptr<NDArray>, std::unique_ptr<NDArray>> checkConv3dCUDNNPadAsymmetric(
    NDArray* input, NDArray* gradI, const int iD, const int iH, const int iW, const int oD, const int oH,
    const int oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD,
    const int pH, const int pW, const int dD, const int dH, const int dW, const bool isNCDHW) {
  const auto pDsum = ((oD - 1) * sD + ((kD - 1) * dD + 1) - iD);
  const auto pHsum = ((oH - 1) * sH + ((kH - 1) * dH + 1) - iH);
  const auto pWsum = ((oW - 1) * sW + ((kW - 1) * dW + 1) - iW);

  const bool isPDasymm = pD != (pDsum - pD);
  const bool isPHasymm = pH != (pHsum - pH);
  const bool isPWasymm = pW != (pWsum - pW);
  std::unique_ptr<NDArray> uNewInput = {}, uNewGradI = {};
  if (!isPDasymm && !isPHasymm && !isPWasymm) return std::make_tuple(std::move(uNewInput), std::move(uNewGradI));

  std::vector<LongType> newShape = input->getShapeAsVector();

  const int iDposition = isNCDHW ? 2 : 1;

  if (isPDasymm) newShape[iDposition] += 1;
  if (isPHasymm) newShape[iDposition + 1] += 1;
  if (isPWasymm) newShape[iDposition + 2] += 1;

  uNewInput.reset(new NDArray(input->ordering(), newShape, input->dataType(), input->getContext()));

  if (isNCDHW)
    (*uNewInput)({0, 0, 0, 0, 0, input->sizeAt(2), 0, input->sizeAt(3), 0, input->sizeAt(4)}).assign(input);
  else
    (*uNewInput)({0, 0, 0, input->sizeAt(1), 0, input->sizeAt(2), 0, input->sizeAt(3), 0, 0}).assign(input);

  if (gradI != nullptr)
    uNewGradI.reset(new NDArray(gradI->ordering(), newShape, gradI->dataType(), gradI->getContext()));
  return std::make_tuple(std::move(uNewInput), std::move(uNewGradI));
}

//////////////////////////////////////////////////////////////////////////
void pooling2dCUDNN(const LaunchContext* context, NDArray* input, NDArray* output, const int kH, const int kW,
                    const int sH, const int sW, const int pH, const int pW, const int dH, const int dW,
                    const bool isNCHW, const cudnnPoolingMode_t mode) {
  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *output, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  // input descriptor, output descriptor
  CudnnTensor x, z;
  if (input->ordering() == 'c')
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
  else
    x.set4DEx(cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC),
              input->strideAt(indIiH), input->strideAt(indIiH + 1));

  if (output->ordering() == 'c')
    z.set4D(format, cudnnDataType(output->dataType()), bS, oC, oH, oW);
  else
    z.set4DEx(cudnnDataType(output->dataType()), bS, oC, oH, oW, output->strideAt(0), output->strideAt(indIOioC),
              output->strideAt(indOoH), output->strideAt(indOoH + 1));

  // description of pooling
  PoolingDesc pooling;
  pooling.set2D(mode, CUDNN_PROPAGATE_NAN, kH, kW, pH, pW, sH, sW);

  // provide scaling parameters
  const float alpha32(1), beta32(0);
  const double alpha64(1), beta64(0);
  const void* alpha =
      output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta =
      output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  // run calculation
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnPoolingForward),
      cudnnPoolingForward(*handle, pooling, alpha, x, input->specialBuffer(), beta, z, output->specialBuffer()));

  auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  if (cudaErr != 0) throw cuda_exception::build("pooling2dCUDNN: cudaStreamSynchronize failed !", cudaErr);

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
void pooling2dBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* gradO, NDArray* gradI,
                      const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH,
                      const int dW, const bool isNCHW, const cudnnPoolingMode_t mode) {
  LongType bS, iC, iH, iW, oC, oH,
      oW;  // batch size, input channels, input height/width, output channels, output height/width;
  LongType indIOioC, indIiH, indWoC, indWiC, indWkH, indOoH;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv2d(isNCHW, 0, *input, *gradO, bS, iC, iH, iW, oC, oH, oW, indIOioC, indIiH,
                                             indWiC, indWoC, indWkH, indOoH);

  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  cudnnTensorFormat_t format = isNCHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  // input and gradI descriptor
  CudnnTensor x;
  if (input->ordering() == 'c')
    x.set4D(format, cudnnDataType(input->dataType()), bS, iC, iH, iW);
  else
    x.set4DEx(cudnnDataType(input->dataType()), bS, iC, iH, iW, input->strideAt(0), input->strideAt(indIOioC),
              input->strideAt(indIiH), input->strideAt(indIiH + 1));

  // gradO descriptor
  CudnnTensor dz;
  if (gradO->ordering() == 'c')
    dz.set4D(format, cudnnDataType(gradO->dataType()), bS, oC, oH, oW);
  else
    dz.set4DEx(cudnnDataType(gradO->dataType()), bS, oC, oH, oW, gradO->strideAt(0), gradO->strideAt(indIOioC),
               gradO->strideAt(indOoH), gradO->strideAt(indOoH + 1));

  // description of pooling
  PoolingDesc pooling;

  pooling.set2D(mode, CUDNN_PROPAGATE_NAN, kH, kW, pH, pW, sH, sW);

  // provide scaling parameters
  const float alpha32(1), beta32(0);
  const double alpha64(1), beta64(0);
  const void* alpha =
      gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta =
      gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({gradI}, {input, gradO});

  // run calculation for gradI
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnPoolingBackward),
      cudnnPoolingBackward(*handle, pooling, alpha, dz, gradO->specialBuffer(), dz, gradO->specialBuffer(), x,
                           input->specialBuffer(), beta, x, gradI->specialBuffer()));

  auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  if (cudaErr != 0) throw cuda_exception::build("pooling2dBpCUDNN: cudaStreamSynchronize failed !", cudaErr);

  NDArray::registerSpecialUse({gradI}, {input, gradO});
}

//////////////////////////////////////////////////////////////////////////
void pooling3dCUDNN(const LaunchContext* context, NDArray* input, NDArray* output, const int kD, const int kH,
                    const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW,
                    const int dD, const int dH, const int dW, const bool isNCDHW, const cudnnPoolingMode_t mode) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  const int numDims = 5;

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *output, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC,
                                             indIOioD, indWiC, indWoC, indWkD);

  const int pSizes[] = {pD, pH, pW};
  const int sSizes[] = {sD, sH, sW};
  const int kSizes[] = {kD, kH, kW};

  const LongType xShape[] = {bS, iC, iD, iH, iW};
  const LongType zShape[] = {bS, oC, oD, oH, oW};

  const LongType xStrides[] = {(LongType)input->strideAt(0), (LongType)input->strideAt(1), (LongType)input->strideAt(2),
                                   (LongType)input->strideAt(3), (LongType)input->strideAt(4)};
  const LongType zStrides[] = {(LongType)output->strideAt(0), (LongType)output->strideAt(1), (LongType)output->strideAt(2),
                                   (LongType)output->strideAt(3), (LongType)output->strideAt(4)};

  cudnnTensorFormat_t format = isNCDHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  // input descriptor, output descriptor
  CudnnTensor x, z;
  if (input->ordering() == 'c') {
    int newShape[5];
    for(int i = 0; i < 5; i++) {
      newShape[i] = static_cast<int>(xShape[i]);
    }
    x.setEx(format, cudnnDataType(input->dataType()), numDims, newShape);
  } else {
    int newShape[5];
    int newStrides[5];
    for(int i = 0; i < 5; i++) {
      newShape[i] = static_cast<int>(xShape[i]);
    }
    for(int i = 0; i < 5; i++) {
      newStrides[i] = static_cast<int>(xStrides[i]);
    }

    x.set(cudnnDataType(input->dataType()), numDims, newShape, newStrides);
  }


  if (output->ordering() == 'c') {
    int newShape[5];
    int newStrides[5];
    for(int i = 0; i < 5; i++) {
      newShape[i] = static_cast<int>(zShape[i]);
    }
    for(int i = 0; i < 5; i++) {
      newStrides[i] = static_cast<int>(zStrides[i]);
    }
    z.setEx(format, cudnnDataType(output->dataType()), numDims, newShape);
  } else {
    int newShape[5];
    int newStrides[5];
    for(int i = 0; i < 5; i++) {
      newShape[i] = static_cast<int>(zShape[i]);
    }
    for(int i = 0; i < 5; i++) {
      newStrides[i] = static_cast<int>(zStrides[i]);
    }
    z.set(cudnnDataType(output->dataType()), numDims, newShape, newStrides);
  }
  // description of pooling
  PoolingDesc pooling;
  pooling.set(mode, CUDNN_PROPAGATE_NAN, numDims - 2, kSizes, pSizes, sSizes);

  // provide scaling parameters
  const float alpha32(1), beta32(0);
  const double alpha64(1), beta64(0);
  const void* alpha =
      output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta =
      output->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  NDArray::prepareSpecialUse({output}, {input});

  // run calculation
  CHECK_CUDNN_FAILURE_MSG(
      STRINGIZE(cudnnPoolingForward),
      cudnnPoolingForward(*handle, pooling, alpha, x, input->specialBuffer(), beta, z, output->specialBuffer()));

  auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  if (cudaErr != 0) throw cuda_exception::build("pooling3dCUDNN: cudaStreamSynchronize failed !", cudaErr);

  NDArray::registerSpecialUse({output}, {input});
}

//////////////////////////////////////////////////////////////////////////
void pooling3dBpCUDNN(const LaunchContext* context, NDArray* input, NDArray* gradO, NDArray* gradI,
                      const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD,
                      const int pH, const int pW, const int dD, const int dH, const int dW, const bool isNCDHW,
                      const cudnnPoolingMode_t mode) {
  auto handle = reinterpret_cast<cudnnHandle_t*>(context->getCuDnnHandle());
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetStream), cudnnSetStream(*handle, *context->getCudaStream()));

  const int numDims = 5;

  LongType bS, iC, iD, iH, iW, oC, oD, oH,
      oW;  // batch size, input channels, input depth/height/width, output channels, output depth/height/width;
  LongType indIOioC, indIOioD, indWoC, indWiC, indWkD;  // corresponding indexes
  ConvolutionUtils::getSizesAndIndexesConv3d(isNCDHW, 0, *input, *gradO, bS, iC, iD, iH, iW, oC, oD, oH, oW, indIOioC,
                                             indIOioD, indWiC, indWoC, indWkD);

  const int pSizes[] = {pD, pH, pW};
  const int sSizes[] = {sD, sH, sW};
  const int kSizes[] = {kD, kH, kW};

  const int xShape[] = {(int) bS, (int) iC, (int) iD, (int) iH, (int) iW};
  const int dzShape[] = {(int) bS, (int) oC, (int) oD, (int) oH,(int) oW};

  const int xStrides[] = { (int) input->strideAt(0), (int)input->strideAt(1), (int)input->strideAt(2),
                                   (int)input->strideAt(3), (int)input->strideAt(4)};
  const int dzStrides[] = {(int)gradO->strideAt(0), (int)gradO->strideAt(1), (int)gradO->strideAt(2),
                                    (int)gradO->strideAt(3), (int)gradO->strideAt(4)};

  cudnnTensorFormat_t format = isNCDHW ? CUDNN_TENSOR_NCHW : CUDNN_TENSOR_NHWC;

  // input and gradI descriptor
  CudnnTensor x;
  if ( input->ordering() == 'c') {
    x.setEx(format, cudnnDataType(input->dataType()), numDims, xShape);
  } else {
    x.set(cudnnDataType(input->dataType()), numDims, xShape, xStrides);
  }
  // gradO descriptor
  CudnnTensor dz;
  if ( gradO->ordering() == 'c')
    dz.setEx(format, cudnnDataType(gradO->dataType()), numDims, dzShape);
  else
    dz.set(cudnnDataType(gradO->dataType()), numDims, dzShape, dzStrides);

  // description of pooling
  PoolingDesc pooling;
  pooling.set(mode, CUDNN_PROPAGATE_NAN, numDims - 2, kSizes, pSizes, sSizes);

  // provide scaling parameters
  const float alpha32(1), beta32(0);
  const double alpha64(1), beta64(0);
  const void* alpha =
      gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&alpha32) : reinterpret_cast<const void*>(&alpha64);
  const void* beta =
      gradO->sizeOfT() <= 4 ? reinterpret_cast<const void*>(&beta32) : reinterpret_cast<const void*>(&beta64);

  // cudnn maxpool2d_bp api requires ff output as one of input arguments
  if (mode == CUDNN_POOLING_MAX) {
    NDArray temp(gradO);
    NDArray::prepareSpecialUse({gradI}, {input, gradO, &temp});

    // run ff calculation
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnPoolingForward),
        cudnnPoolingForward(*handle, pooling, alpha, x, input->specialBuffer(), beta, dz, temp.specialBuffer()));

    // run bp calculation for gradI
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnPoolingBackward),
        cudnnPoolingBackward(*handle, pooling, alpha, dz, temp.specialBuffer(), dz, gradO->specialBuffer(), x,
                             input->specialBuffer(), beta, x, gradI->specialBuffer()));

    NDArray::registerSpecialUse({gradI}, {input, gradO, &temp});
  } else {
    NDArray::prepareSpecialUse({gradI}, {input, gradO});
    // run bp calculation for gradI
    CHECK_CUDNN_FAILURE_MSG(
        STRINGIZE(cudnnPoolingBackward),
        cudnnPoolingBackward(*handle, pooling, alpha, dz, gradO->specialBuffer(), dz, gradO->specialBuffer(), x,
                             input->specialBuffer(), beta, x, gradI->specialBuffer()));
    NDArray::registerSpecialUse({gradI}, {input, gradO});
  }

  auto cudaErr = cudaStreamSynchronize(*context->getCudaStream());
  if (cudaErr != 0) throw cuda_exception::build("pooling3dBpCUDNN: cudaStreamSynchronize failed !", cudaErr);
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd
