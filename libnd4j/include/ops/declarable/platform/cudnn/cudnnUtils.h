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
// @author raver119@gmail.com
//

#ifndef SD_CUDNNUTILS_H
#define SD_CUDNNUTILS_H
#include <cudnn.h>
#include <exceptions/cuda_exception.h>
#include <exceptions/datatype_exception.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#include <memory>
#include <tuple>
#include <vector>

#define CUDNN_NEW_RNN_API_VER 8001
#define CUDNN_CLIPPING_API_VER 7201

namespace sd {
namespace ops {
namespace platforms {

DECLARE_PLATFORM(conv2d, ENGINE_CUDA);
DECLARE_PLATFORM(conv2d_bp, ENGINE_CUDA);

DECLARE_PLATFORM(conv3dnew, ENGINE_CUDA);
DECLARE_PLATFORM(conv3dnew_bp, ENGINE_CUDA);

DECLARE_PLATFORM(depthwise_conv2d, ENGINE_CUDA);
DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_CUDA);

DECLARE_PLATFORM(batchnorm, ENGINE_CUDA);
DECLARE_PLATFORM(batchnorm_bp, ENGINE_CUDA);

DECLARE_PLATFORM(avgpool2d, ENGINE_CUDA);
DECLARE_PLATFORM(avgpool2d_bp, ENGINE_CUDA);

DECLARE_PLATFORM(maxpool2d, ENGINE_CUDA);
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_CUDA);

DECLARE_PLATFORM(avgpool3dnew, ENGINE_CUDA);
DECLARE_PLATFORM(avgpool3dnew_bp, ENGINE_CUDA);

DECLARE_PLATFORM(maxpool3dnew, ENGINE_CUDA);
DECLARE_PLATFORM(maxpool3dnew_bp, ENGINE_CUDA);

DECLARE_PLATFORM(lstmLayer, ENGINE_CUDA);

DECLARE_PLATFORM(ctc_loss, ENGINE_CUDA);
DECLARE_PLATFORM(ctc_loss_grad, ENGINE_CUDA);

//////////////////////////////////////////////////////////////////////////

inline void throwIfCudnnFailed(cudnnStatus_t result_status,
                               const char* message = "Cudnn error: ", const char* prefix = nullptr) {
  if (result_status != CUDNN_STATUS_SUCCESS) {
    std::string err_message;
    if (prefix) err_message = std::string(prefix) + ": ";
    err_message += std::string(message);
    throw cuda_exception::build(err_message, result_status);
  }
}

#define STRINGIZE(x) STRINGIZE2(x)
#define STRINGIZE2(x) #x
#define CHECK_CUDNN_FAILURE(result_status) throwIfCudnnFailed(result_status, "")
#define CHECK_CUDNN_FAILURE_MSG(custom_message, result_status) \
  throwIfCudnnFailed(result_status, custom_message, __func__)

template <typename T>
SD_INLINE const T* bufferInHost(const NDArray& array) {
  array.syncToHost();
  return reinterpret_cast<const T*>(array.buffer());
}

#define MOVEONLY_DESC_IMPL(DESC)                                                 \
  DESC(const DESC& s) = delete;                                                  \
  DESC& operator=(const DESC& other) = delete;                                   \
  DESC(DESC&& other) noexcept : desc(std::move(other.desc)) { other.desc = {}; } \
  DESC& operator=(DESC&& other) noexcept {                                       \
    if (&other == this) return *this;                                            \
    destroy();                                                                   \
    desc = std::move(other.desc);                                                \
    other.desc = {};                                                             \
    return *this;                                                                \
  }

#define MOVEONLY_DESC_FULL_IMPL(DESC_CLASS, DESC_NAME)                                                         \
  DESC_CLASS() { create(); }                                                                                   \
  DESC_CLASS(cudnn##DESC_NAME##_t created) { desc = created; }                                                 \
  void create() { CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnCreate##DESC_NAME), cudnnCreate##DESC_NAME(&desc)); } \
  void destroy() {                                                                                             \
    if (desc) CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnCreate##DESC_NAME), cudnnDestroy##DESC_NAME(desc));       \
    desc = {};                                                                                                 \
  }                                                                                                            \
  MOVEONLY_DESC_IMPL(DESC_CLASS)                                                                               \
  operator cudnn##DESC_NAME##_t() const { return desc; }                                                       \
  ~DESC_CLASS() { destroy(); }                                                                                 \
  cudnn##DESC_NAME##_t desc;

struct CudnnTensor {
  MOVEONLY_DESC_FULL_IMPL(CudnnTensor, TensorDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetTensorNdDescriptor),
                            cudnnSetTensorNdDescriptor(desc, std::forward<Args>(args)...));
  }

  template <typename... Args>
  void setEx(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetTensorNdDescriptorEx),
                            cudnnSetTensorNdDescriptorEx(desc, std::forward<Args>(args)...));
  }

  template <typename... Args>
  void set4D(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetTensor4dDescriptor),
                            cudnnSetTensor4dDescriptor(desc, std::forward<Args>(args)...));
  }

  template <typename... Args>
  void set4DEx(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetTensor4dDescriptorEx),
                            cudnnSetTensor4dDescriptorEx(desc, std::forward<Args>(args)...));
  }
};

struct CudnnTensorList {
  MOVEONLY_DESC_IMPL(CudnnTensorList)

  CudnnTensorList(int size) {
    desc.resize(size);
    for (int i = 0; i < size; i++) {
      CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnCreateTensorDescriptor), cudnnCreateTensorDescriptor(&desc[i]));
    }
  }

  template <typename... Args>
  void set(int index, Args&&... args) {
    if (index < desc.size()) {
      CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetTensorNdDescriptor),
                              cudnnSetTensorNdDescriptor(desc[index], std::forward<Args>(args)...));
    }
  }

  cudnnTensorDescriptor_t get(int i) const {
    if (i < desc.size()) return desc[i];
    return nullptr;
  }

  const cudnnTensorDescriptor_t* getDescriptors() const { return desc.data(); }

  void destroy() {
    for (auto x : desc) {
      CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnDestroyTensorDescriptor), cudnnDestroyTensorDescriptor(x));
    }
    desc = {};
  }

  ~CudnnTensorList() { destroy(); }

  std::vector<cudnnTensorDescriptor_t> desc;
};

struct FilterDesc {
  MOVEONLY_DESC_FULL_IMPL(FilterDesc, FilterDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetFilterNdDescriptor),
                            cudnnSetFilterNdDescriptor(desc, std::forward<Args>(args)...));
  }

  template <typename... Args>
  void set4D(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetFilter4dDescriptor),
                            cudnnSetFilter4dDescriptor(desc, std::forward<Args>(args)...));
  }
};

struct DropoutDesc {
  MOVEONLY_DESC_FULL_IMPL(DropoutDesc, DropoutDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetDropoutDescriptor),
                            cudnnSetDropoutDescriptor(desc, std::forward<Args>(args)...));
  }
};

#if CUDNN_VERSION > CUDNN_NEW_RNN_API_VER
struct RnnDataDesc {
  MOVEONLY_DESC_FULL_IMPL(RnnDataDesc, RNNDataDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetRNNDataDescriptor),
                            cudnnSetRNNDataDescriptor(desc, std::forward<Args>(args)...));
  }
};
#endif

SD_INLINE void setRnnDescriptorOldApi(cudnnRNNDescriptor_t rnnDesc, cudnnHandle_t handle, cudnnRNNInputMode_t inputMode,
                                      cudnnDirectionMode_t dirMode, cudnnRNNMode_t cellMode, cudnnRNNAlgo_t algo,
                                      cudnnDataType_t mathPrec, int32_t hiddenSize, int32_t numLayers,
                                      cudnnDropoutDescriptor_t dropoutDesc, bool use_tensor_op = false) {
  auto err = cudnnSetRNNDescriptor_v6(handle, rnnDesc, hiddenSize, numLayers, dropoutDesc, inputMode, dirMode, cellMode,
                                      algo, mathPrec);
  CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetRNNDescriptor_v6), err);
#if CUDNN_VERSION >= 7001
  if (cudnnGetVersion() >= 7001) {
    cudnnMathType_t mathType = use_tensor_op ? CUDNN_TENSOR_OP_MATH : CUDNN_DEFAULT_MATH;
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetRNNMatrixMathType), cudnnSetRNNMatrixMathType(rnnDesc, mathType));
  }
#endif
  return;
}

struct RnnDesc {
  MOVEONLY_DESC_FULL_IMPL(RnnDesc, RNNDescriptor)

  template <typename... Args>
  void setUsingOldAPI(Args&&... args) {
    setRnnDescriptorOldApi(desc, std::forward<Args>(args)...);
  }

#if CUDNN_VERSION >= CUDNN_NEW_RNN_API_VER
  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetRNNDescriptor_v8),
                            cudnnSetRNNDescriptor_v8(desc, std::forward<Args>(args)...));
  }
#endif
};

struct CTCLossDesc {
  MOVEONLY_DESC_FULL_IMPL(CTCLossDesc, CTCLossDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetCTCLossDescriptorEx),
                            cudnnSetCTCLossDescriptorEx(desc, std::forward<Args>(args)...));
  }
};
//////////////////////////////////////////////////////////////////////////

struct PoolingDesc {
  MOVEONLY_DESC_FULL_IMPL(PoolingDesc, PoolingDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetPoolingNdDescriptor),
                            cudnnSetPoolingNdDescriptor(desc, std::forward<Args>(args)...));
  }

  template <typename... Args>
  void set2D(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetPooling2dDescriptor),
                            cudnnSetPooling2dDescriptor(desc, std::forward<Args>(args)...));
  }
};

//////////////////////////////////////////////////////////////////////////

struct ConvolutionDesc {
  MOVEONLY_DESC_FULL_IMPL(ConvolutionDesc, ConvolutionDescriptor)

  template <typename... Args>
  void set(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetConvolutionNdDescriptor),
                            cudnnSetConvolutionNdDescriptor(desc, std::forward<Args>(args)...));
  }

  template <typename... Args>
  void set2D(Args&&... args) {
    CHECK_CUDNN_FAILURE_MSG(STRINGIZE(cudnnSetConvolution2dDescriptor),
                            cudnnSetConvolution2dDescriptor(desc, std::forward<Args>(args)...));
  }
};

//////////////////////////////////////////////////////////////////////////
SD_INLINE cudnnDataType_t cudnnDataType(DataType dataType) {
  switch (dataType) {
    case FLOAT32:
      return CUDNN_DATA_FLOAT;
    case DOUBLE:
      return CUDNN_DATA_DOUBLE;
    case HALF:
      return CUDNN_DATA_HALF;
    case INT32:
      return CUDNN_DATA_INT32;
    case INT8:
      return CUDNN_DATA_INT8;
    default:
      throw datatype_exception::build("Unsupported data type", dataType);
  }
}

//////////////////////////////////////////////////////////////////////////
std::tuple<std::unique_ptr<NDArray>, std::unique_ptr<NDArray>> checkConv2dCUDNNPadAsymmetric(
    const NDArray* input, const NDArray* gradI, const int iH, const int iW, const int oH, const int oW, const int kH,
    const int kW, const int sH, const int sW, const int pH, const int pW, const int dH, const int dW,
    const bool isNCHW);

//////////////////////////////////////////////////////////////////////////
std::tuple<std::unique_ptr<NDArray>, std::unique_ptr<NDArray>> checkConv3dCUDNNPadAsymmetric(
    const NDArray* input, const NDArray* gradI, const int iD, const int iH, const int iW, const int oD, const int oH,
    const int oW, const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD,
    const int pH, const int pW, const int dD, const int dH, const int dW, const bool isNCDHW);

//////////////////////////////////////////////////////////////////////////
void pooling2dCUDNN(const LaunchContext* context, const NDArray* input, NDArray* output, const int kH, const int kW,
                    const int sH, const int sW, const int pH, const int pW, const int dH, const int dW,
                    const bool isNCHW, const cudnnPoolingMode_t mode);

//////////////////////////////////////////////////////////////////////////
void pooling2dBpCUDNN(const LaunchContext* context, const NDArray* input, const NDArray* gradO, NDArray* gradI,
                      const int kH, const int kW, const int sH, const int sW, const int pH, const int pW, const int dH,
                      const int dW, const bool isNCHW, const cudnnPoolingMode_t mode);

//////////////////////////////////////////////////////////////////////////
void pooling3dCUDNN(const LaunchContext* context, const NDArray* input, NDArray* output, const int kD, const int kH,
                    const int kW, const int sD, const int sH, const int sW, const int pD, const int pH, const int pW,
                    const int dD, const int dH, const int dW, const bool isNCDHW, const cudnnPoolingMode_t mode);

//////////////////////////////////////////////////////////////////////////
void pooling3dBpCUDNN(const LaunchContext* context, const NDArray* input, const NDArray* gradO, NDArray* gradI,
                      const int kD, const int kH, const int kW, const int sD, const int sH, const int sW, const int pD,
                      const int pH, const int pW, const int dD, const int dH, const int dW, const bool isNCDHW,
                      const cudnnPoolingMode_t mode);

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_CUDNNUTILS_H
