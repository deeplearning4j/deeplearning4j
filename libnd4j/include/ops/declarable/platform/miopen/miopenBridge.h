/* ******************************************************************************
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

#ifndef SD_ZLUDA_MIOPEN_BRIDGE_H
#define SD_ZLUDA_MIOPEN_BRIDGE_H

#include <cstddef>

namespace sd {
namespace ops {
namespace platforms {
namespace miopen_bridge {

// This header is deliberately SDK-neutral. CUDA/ZLUDA translation units include
// it without seeing HIP or MIOpen declarations; miopenBridge.cpp is the only
// translation unit that owns the native ROCm headers.

enum class DataType : int {
  FLOAT32,
  FLOAT16,
  BFLOAT16,
  INT8,
  INT32
};

enum class ActivationMode : int {
  RELU,
  CLIPPED_RELU,
  LOGISTIC,
  TANH,
  ELU,
  SOFT_RELU
};

struct Tensor4D {
  DataType dataType;
  int n;
  int c;
  int h;
  int w;
};

struct Convolution2D {
  int padH;
  int padW;
  int strideH;
  int strideW;
  int dilationH;
  int dilationW;
};

// Returns a thread-local diagnostic for the most recent nonzero bridge status.
const char* lastError();

int activationForward(int deviceId, const Tensor4D& tensor,
                      const void* input, void* output,
                      ActivationMode mode, double alpha,
                      double beta, double gamma);

int batchNormForwardInference(int deviceId, const Tensor4D& tensor,
                              const Tensor4D& parameterTensor,
                              const void* input, void* output,
                              const void* scale, const void* bias,
                              const void* mean, const void* variance,
                              double epsilon);

int batchNormBackward(int deviceId, const Tensor4D& tensor,
                      const Tensor4D& parameterTensor,
                      const void* input, const void* gradOutput,
                      const void* scale, const void* mean,
                      const void* variance, void* gradInput,
                      void* gradScale, void* gradBias,
                      double epsilon);

int softmaxForward(int deviceId, const Tensor4D& tensor,
                   const void* input, void* output, bool logSoftmax);

int softmaxBackward(int deviceId, const Tensor4D& tensor,
                    const void* input, const void* gradOutput,
                    void* gradInput, bool logSoftmax);

int convolutionForward(int deviceId,
                       const Tensor4D& inputTensor,
                       const Tensor4D& weightsTensor,
                       const Tensor4D& outputTensor,
                       const Convolution2D& convolution,
                       const void* input, const void* weights,
                       void* output, const Tensor4D* biasTensor,
                       const void* bias);

int convolutionBackward(int deviceId,
                        const Tensor4D& inputTensor,
                        const Tensor4D& weightsTensor,
                        const Tensor4D& gradOutputTensor,
                        const Convolution2D& convolution,
                        const void* input, const void* weights,
                        const void* gradOutput, void* gradInput,
                        void* gradWeights, const Tensor4D* gradBiasTensor,
                        void* gradBias);

}  // namespace miopen_bridge
}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // SD_ZLUDA_MIOPEN_BRIDGE_H
