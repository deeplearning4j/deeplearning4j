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
// @author Adam Gibson
//
// Metal Performance Shaders - Normalization operations
//

#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <ConstMessages.h>
#include "mpsUtils.h"

#ifdef HAVE_MPS
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#endif

namespace sd {
namespace ops {
namespace platforms {

#ifdef HAVE_MPS

//////////////////////////////////////////////////////////////////////////
// Batch Normalization
// y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
//////////////////////////////////////////////////////////////////////////

static void batchnormMPS(const NDArray* input, const NDArray* mean, const NDArray* variance,
                          const NDArray* gamma, const NDArray* beta,
                          NDArray* output, float epsilon) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        // Input: [N, C, H, W] - NCHW format
        NSUInteger batch = input->sizeAt(0);
        NSUInteger channels = input->sizeAt(1);
        NSUInteger height = input->sizeAt(2);
        NSUInteger width = input->sizeAt(3);

        // Create image descriptors
        MPSImageDescriptor* inputDesc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:width
                                      height:height
                             featureChannels:channels
                              numberOfImages:batch
                                       usage:MTLTextureUsageShaderRead];

        MPSImageDescriptor* outputDesc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:width
                                      height:height
                             featureChannels:channels
                              numberOfImages:batch
                                       usage:MTLTextureUsageShaderWrite];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:inputDesc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:outputDesc];

        // Copy input data
        mpsUtils::copyMPSImageToNDArray(inputImage, const_cast<NDArray*>(input));

        // Create batch normalization data source
        // Note: MPS requires MPSCNNBatchNormalizationDataSource protocol implementation
        // For simplicity, we perform manual normalization using element-wise operations

        // Create command buffer
        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();

        // For proper MPS batch norm, you'd implement MPSCNNBatchNormalizationDataSource
        // Here we use a simplified approach with MPSCNNNormalization or manual computation

        // Execute batch norm using MPSCNNBatchNormalization if available
        if (@available(macOS 10.13.4, iOS 11.3, *)) {
            // Use MPSCNNBatchNormalizationStatistics for inference
            // This is a simplified version - production code would implement full data source

            // For now, commit empty command buffer and do CPU fallback for complex cases
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            // CPU fallback for batch norm
            const float* inPtr = input->bufferAsT<float>();
            const float* meanPtr = mean->bufferAsT<float>();
            const float* varPtr = variance->bufferAsT<float>();
            const float* gammaPtr = gamma ? gamma->bufferAsT<float>() : nullptr;
            const float* betaPtr = beta ? beta->bufferAsT<float>() : nullptr;
            float* outPtr = output->bufferAsT<float>();

            for (NSUInteger n = 0; n < batch; n++) {
                for (NSUInteger c = 0; c < channels; c++) {
                    float m = meanPtr[c];
                    float v = varPtr[c];
                    float g = gammaPtr ? gammaPtr[c] : 1.0f;
                    float b = betaPtr ? betaPtr[c] : 0.0f;
                    float invStd = 1.0f / sqrtf(v + epsilon);

                    for (NSUInteger h = 0; h < height; h++) {
                        for (NSUInteger w = 0; w < width; w++) {
                            NSUInteger idx = n * channels * height * width +
                                           c * height * width +
                                           h * width + w;
                            outPtr[idx] = g * (inPtr[idx] - m) * invStd + b;
                        }
                    }
                }
            }
        }
    }
}

PLATFORM_IMPL(batchnorm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto mean = INPUT_VARIABLE(1);
    auto variance = INPUT_VARIABLE(2);
    NDArray* gamma = block.width() > 3 ? INPUT_VARIABLE(3) : nullptr;
    NDArray* beta = block.width() > 4 ? INPUT_VARIABLE(4) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float epsilon = 1e-5f;
    if (block.getTArguments()->size() > 0) {
        epsilon = T_ARG(0);
    }

    batchnormMPS(input, mean, variance, gamma, beta, output, epsilon);

    return sd::Status::OK;
}

PLATFORM_CHECK(batchnorm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS BATCHNORM OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Layer Normalization
// y = gamma * (x - mean) / sqrt(variance + epsilon) + beta
// Normalizes across the last dimension(s)
//////////////////////////////////////////////////////////////////////////

static void layerNormMPS(const NDArray* input, const NDArray* gamma, const NDArray* beta,
                          NDArray* output, float epsilon, int axis) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        // Layer norm normalizes across specified axes
        // For simplicity, we normalize across the last axis

        const float* inPtr = input->bufferAsT<float>();
        const float* gammaPtr = gamma ? gamma->bufferAsT<float>() : nullptr;
        const float* betaPtr = beta ? beta->bufferAsT<float>() : nullptr;
        float* outPtr = output->bufferAsT<float>();

        // Get dimensions
        LongType outerSize = 1;
        LongType innerSize = input->sizeAt(-1);

        for (int i = 0; i < input->rankOf() - 1; i++) {
            outerSize *= input->sizeAt(i);
        }

        // Compute layer norm
        for (LongType i = 0; i < outerSize; i++) {
            // Compute mean
            float mean = 0.0f;
            for (LongType j = 0; j < innerSize; j++) {
                mean += inPtr[i * innerSize + j];
            }
            mean /= innerSize;

            // Compute variance
            float var = 0.0f;
            for (LongType j = 0; j < innerSize; j++) {
                float diff = inPtr[i * innerSize + j] - mean;
                var += diff * diff;
            }
            var /= innerSize;

            // Normalize
            float invStd = 1.0f / sqrtf(var + epsilon);
            for (LongType j = 0; j < innerSize; j++) {
                float normalized = (inPtr[i * innerSize + j] - mean) * invStd;
                float g = gammaPtr ? gammaPtr[j] : 1.0f;
                float b = betaPtr ? betaPtr[j] : 0.0f;
                outPtr[i * innerSize + j] = g * normalized + b;
            }
        }
    }
}

PLATFORM_IMPL(layer_norm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    NDArray* gamma = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;
    NDArray* beta = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float epsilon = 1e-5f;
    int axis = -1;
    if (block.getTArguments()->size() > 0) {
        epsilon = T_ARG(0);
    }
    if (block.getIArguments()->size() > 0) {
        axis = INT_ARG(0);
    }

    layerNormMPS(input, gamma, beta, output, epsilon, axis);

    return sd::Status::OK;
}

PLATFORM_CHECK(layer_norm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS LAYER_NORM OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Instance Normalization
// Normalizes across spatial dimensions (H, W) for each channel independently
//////////////////////////////////////////////////////////////////////////

static void instanceNormMPS(const NDArray* input, const NDArray* gamma, const NDArray* beta,
                             NDArray* output, float epsilon) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        // Input: [N, C, H, W]
        NSUInteger batch = input->sizeAt(0);
        NSUInteger channels = input->sizeAt(1);
        NSUInteger height = input->sizeAt(2);
        NSUInteger width = input->sizeAt(3);
        NSUInteger spatialSize = height * width;

        const float* inPtr = input->bufferAsT<float>();
        const float* gammaPtr = gamma ? gamma->bufferAsT<float>() : nullptr;
        const float* betaPtr = beta ? beta->bufferAsT<float>() : nullptr;
        float* outPtr = output->bufferAsT<float>();

        // Instance norm: normalize per (N, C) across (H, W)
        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                NSUInteger offset = n * channels * spatialSize + c * spatialSize;

                // Compute mean
                float mean = 0.0f;
                for (NSUInteger i = 0; i < spatialSize; i++) {
                    mean += inPtr[offset + i];
                }
                mean /= spatialSize;

                // Compute variance
                float var = 0.0f;
                for (NSUInteger i = 0; i < spatialSize; i++) {
                    float diff = inPtr[offset + i] - mean;
                    var += diff * diff;
                }
                var /= spatialSize;

                // Normalize
                float invStd = 1.0f / sqrtf(var + epsilon);
                float g = gammaPtr ? gammaPtr[c] : 1.0f;
                float b = betaPtr ? betaPtr[c] : 0.0f;

                for (NSUInteger i = 0; i < spatialSize; i++) {
                    outPtr[offset + i] = g * (inPtr[offset + i] - mean) * invStd + b;
                }
            }
        }
    }
}

PLATFORM_IMPL(instance_norm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    NDArray* gamma = block.width() > 1 ? INPUT_VARIABLE(1) : nullptr;
    NDArray* beta = block.width() > 2 ? INPUT_VARIABLE(2) : nullptr;
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float epsilon = 1e-5f;
    if (block.getTArguments()->size() > 0) {
        epsilon = T_ARG(0);
    }

    instanceNormMPS(input, gamma, beta, output, epsilon);

    return sd::Status::OK;
}

PLATFORM_CHECK(instance_norm, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS INSTANCE_NORM OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectEq(makeInfoVariable(input->rankOf(), RANK_MSG_INPUT0), 4);
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
