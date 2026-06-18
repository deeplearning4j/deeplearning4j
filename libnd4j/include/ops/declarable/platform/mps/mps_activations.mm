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
// Metal Performance Shaders - Activation function operations
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
// ReLU Activation
//////////////////////////////////////////////////////////////////////////

static void reluMPS(const NDArray* input, NDArray* output) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger length = input->lengthOf();
        size_t bufferSize = length * sizeof(float);

        // Create Metal buffers
        id<MTLBuffer> inputBuffer = [device newBufferWithBytes:input->buffer()
                                                        length:bufferSize
                                                       options:MTLResourceStorageModeShared];
        id<MTLBuffer> outputBuffer = [device newBufferWithLength:bufferSize
                                                         options:MTLResourceStorageModeShared];

        // For element-wise operations, we need to use MPSNNGraph or custom compute shaders
        // MPS provides MPSCNNNeuronReLU for image-based operations
        // For general tensors, we can use MPSNDArray operations

        // Create image descriptors (treating as 1D image)
        MPSImageDescriptor* desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:length
                                      height:1
                             featureChannels:1];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];

        // Copy input data
        MTLRegion region = MTLRegionMake3D(0, 0, 0, length, 1, 1);
        [inputImage.texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:input->buffer()
                              bytesPerRow:bufferSize];

        // Create ReLU kernel
        MPSCNNNeuronReLU* relu = [[MPSCNNNeuronReLU alloc] initWithDevice:device a:0.0f];

        // Execute
        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();
        [relu encodeToCommandBuffer:commandBuffer
                        sourceImage:inputImage
                   destinationImage:outputImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy output
        [outputImage.texture getBytes:output->buffer()
                          bytesPerRow:bufferSize
                           fromRegion:region
                          mipmapLevel:0];
    }
}

PLATFORM_IMPL(relu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    reluMPS(input, output);

    return sd::Status::OK;
}

PLATFORM_CHECK(relu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS RELU OP");

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
// Leaky ReLU
//////////////////////////////////////////////////////////////////////////

static void leakyReluMPS(const NDArray* input, NDArray* output, float alpha) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger length = input->lengthOf();
        size_t bufferSize = length * sizeof(float);

        MPSImageDescriptor* desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:length
                                      height:1
                             featureChannels:1];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];

        MTLRegion region = MTLRegionMake3D(0, 0, 0, length, 1, 1);
        [inputImage.texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:input->buffer()
                              bytesPerRow:bufferSize];

        // Create Leaky ReLU kernel (ReLUN with negative slope)
        MPSCNNNeuronReLU* leakyRelu = [[MPSCNNNeuronReLU alloc] initWithDevice:device a:alpha];

        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();
        [leakyRelu encodeToCommandBuffer:commandBuffer
                             sourceImage:inputImage
                        destinationImage:outputImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        [outputImage.texture getBytes:output->buffer()
                          bytesPerRow:bufferSize
                           fromRegion:region
                          mipmapLevel:0];
    }
}

PLATFORM_IMPL(leaky_relu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float alpha = 0.01f;  // Default leaky ReLU slope
    if (block.getTArguments()->size() > 0) {
        alpha = T_ARG(0);
    }

    leakyReluMPS(input, output, alpha);

    return sd::Status::OK;
}

PLATFORM_CHECK(leaky_relu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS LEAKY_RELU OP");

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
// Sigmoid
//////////////////////////////////////////////////////////////////////////

static void sigmoidMPS(const NDArray* input, NDArray* output) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger length = input->lengthOf();
        size_t bufferSize = length * sizeof(float);

        MPSImageDescriptor* desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:length
                                      height:1
                             featureChannels:1];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];

        MTLRegion region = MTLRegionMake3D(0, 0, 0, length, 1, 1);
        [inputImage.texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:input->buffer()
                              bytesPerRow:bufferSize];

        // Create Sigmoid kernel
        MPSCNNNeuronSigmoid* sigmoid = [[MPSCNNNeuronSigmoid alloc] initWithDevice:device];

        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();
        [sigmoid encodeToCommandBuffer:commandBuffer
                           sourceImage:inputImage
                      destinationImage:outputImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        [outputImage.texture getBytes:output->buffer()
                          bytesPerRow:bufferSize
                           fromRegion:region
                          mipmapLevel:0];
    }
}

PLATFORM_IMPL(sigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    sigmoidMPS(input, output);

    return sd::Status::OK;
}

PLATFORM_CHECK(sigmoid, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS SIGMOID OP");

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
// Tanh
//////////////////////////////////////////////////////////////////////////

static void tanhMPS(const NDArray* input, NDArray* output) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger length = input->lengthOf();
        size_t bufferSize = length * sizeof(float);

        MPSImageDescriptor* desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:length
                                      height:1
                             featureChannels:1];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];

        MTLRegion region = MTLRegionMake3D(0, 0, 0, length, 1, 1);
        [inputImage.texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:input->buffer()
                              bytesPerRow:bufferSize];

        // Create Tanh kernel
        MPSCNNNeuronTanH* tanhKernel = [[MPSCNNNeuronTanH alloc] initWithDevice:device a:1.0f b:1.0f];

        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();
        [tanhKernel encodeToCommandBuffer:commandBuffer
                              sourceImage:inputImage
                         destinationImage:outputImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        [outputImage.texture getBytes:output->buffer()
                          bytesPerRow:bufferSize
                           fromRegion:region
                          mipmapLevel:0];
    }
}

PLATFORM_IMPL(tanh, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    tanhMPS(input, output);

    return sd::Status::OK;
}

PLATFORM_CHECK(tanh, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS TANH OP");

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
// ELU
//////////////////////////////////////////////////////////////////////////

static void eluMPS(const NDArray* input, NDArray* output, float alpha) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger length = input->lengthOf();
        size_t bufferSize = length * sizeof(float);

        MPSImageDescriptor* desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:length
                                      height:1
                             featureChannels:1];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];

        MTLRegion region = MTLRegionMake3D(0, 0, 0, length, 1, 1);
        [inputImage.texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:input->buffer()
                              bytesPerRow:bufferSize];

        // Create ELU kernel
        MPSCNNNeuronELU* elu = [[MPSCNNNeuronELU alloc] initWithDevice:device a:alpha];

        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();
        [elu encodeToCommandBuffer:commandBuffer
                       sourceImage:inputImage
                  destinationImage:outputImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        [outputImage.texture getBytes:output->buffer()
                          bytesPerRow:bufferSize
                           fromRegion:region
                          mipmapLevel:0];
    }
}

PLATFORM_IMPL(elu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    float alpha = 1.0f;  // Default ELU alpha
    if (block.getTArguments()->size() > 0) {
        alpha = T_ARG(0);
    }

    eluMPS(input, output, alpha);

    return sd::Status::OK;
}

PLATFORM_CHECK(elu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS ELU OP");

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
// Softmax
//////////////////////////////////////////////////////////////////////////

static void softmaxMPS(const NDArray* input, NDArray* output) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        // For 2D input: [batch, features]
        // For higher dims: softmax along last dimension
        NSUInteger batch = 1;
        NSUInteger features = input->lengthOf();

        if (input->rankOf() >= 2) {
            batch = input->lengthOf() / input->sizeAt(-1);
            features = input->sizeAt(-1);
        }

        size_t bufferSize = input->lengthOf() * sizeof(float);

        MPSImageDescriptor* desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:features
                                      height:batch
                             featureChannels:1];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];

        MTLRegion region = MTLRegionMake3D(0, 0, 0, features, batch, 1);
        size_t bytesPerRow = features * sizeof(float);
        [inputImage.texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:input->buffer()
                              bytesPerRow:bytesPerRow];

        // Create Softmax kernel
        MPSCNNSoftMax* softmax = [[MPSCNNSoftMax alloc] initWithDevice:device];

        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();
        [softmax encodeToCommandBuffer:commandBuffer
                           sourceImage:inputImage
                      destinationImage:outputImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        [outputImage.texture getBytes:output->buffer()
                          bytesPerRow:bytesPerRow
                           fromRegion:region
                          mipmapLevel:0];
    }
}

PLATFORM_IMPL(softmax, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    softmaxMPS(input, output);

    return sd::Status::OK;
}

PLATFORM_CHECK(softmax, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS SOFTMAX OP");

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
// GELU (Gaussian Error Linear Unit)
// GELU(x) = x * Φ(x) where Φ is the standard Gaussian CDF
// Approximation: 0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x^3)))
//////////////////////////////////////////////////////////////////////////

static void geluMPS(const NDArray* input, NDArray* output) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger length = input->lengthOf();
        size_t bufferSize = length * sizeof(float);

        MPSImageDescriptor* desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:length
                                      height:1
                             featureChannels:1];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
        MPSImage* outputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];

        MTLRegion region = MTLRegionMake3D(0, 0, 0, length, 1, 1);
        [inputImage.texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:input->buffer()
                              bytesPerRow:bufferSize];

        // Create GELU kernel (available in newer MPS versions)
        // For compatibility, we can use MPSCNNNeuronGeLU if available
        if (@available(macOS 11.0, iOS 14.0, *)) {
            MPSCNNNeuronGeLU* gelu = [[MPSCNNNeuronGeLU alloc] initWithDevice:device];

            id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();
            [gelu encodeToCommandBuffer:commandBuffer
                            sourceImage:inputImage
                       destinationImage:outputImage];
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            [outputImage.texture getBytes:output->buffer()
                              bytesPerRow:bufferSize
                               fromRegion:region
                              mipmapLevel:0];
        } else {
            // Fallback: compute GELU on CPU
            const float* inPtr = input->bufferAsT<float>();
            float* outPtr = output->bufferAsT<float>();
            const float sqrt2OverPi = 0.7978845608f;  // sqrt(2/π)
            const float coeff = 0.044715f;

            for (NSUInteger i = 0; i < length; i++) {
                float x = inPtr[i];
                float inner = sqrt2OverPi * (x + coeff * x * x * x);
                outPtr[i] = 0.5f * x * (1.0f + tanhf(inner));
            }
        }
    }
}

PLATFORM_IMPL(gelu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    geluMPS(input, output);

    return sd::Status::OK;
}

PLATFORM_CHECK(gelu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS GELU OP");

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
// SiLU (Sigmoid Linear Unit) / Swish
// SiLU(x) = x * sigmoid(x)
//////////////////////////////////////////////////////////////////////////

static void siluMPS(const NDArray* input, NDArray* output) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger length = input->lengthOf();
        size_t bufferSize = length * sizeof(float);

        // For SiLU, we compute x * sigmoid(x)
        // First compute sigmoid, then multiply by input

        MPSImageDescriptor* desc = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:length
                                      height:1
                             featureChannels:1];

        MPSImage* inputImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];
        MPSImage* sigmoidImage = [[MPSImage alloc] initWithDevice:device imageDescriptor:desc];

        MTLRegion region = MTLRegionMake3D(0, 0, 0, length, 1, 1);
        [inputImage.texture replaceRegion:region
                              mipmapLevel:0
                                withBytes:input->buffer()
                              bytesPerRow:bufferSize];

        // Compute sigmoid
        MPSCNNNeuronSigmoid* sigmoid = [[MPSCNNNeuronSigmoid alloc] initWithDevice:device];

        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();
        [sigmoid encodeToCommandBuffer:commandBuffer
                           sourceImage:inputImage
                      destinationImage:sigmoidImage];
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Get sigmoid result and multiply by input
        float* sigmoidData = (float*)malloc(bufferSize);
        [sigmoidImage.texture getBytes:sigmoidData
                           bytesPerRow:bufferSize
                            fromRegion:region
                           mipmapLevel:0];

        const float* inPtr = input->bufferAsT<float>();
        float* outPtr = output->bufferAsT<float>();
        for (NSUInteger i = 0; i < length; i++) {
            outPtr[i] = inPtr[i] * sigmoidData[i];
        }

        free(sigmoidData);
    }
}

PLATFORM_IMPL(silu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);
    auto output = OUTPUT_VARIABLE(0);

    if (input->isEmpty()) return sd::Status::OK;

    siluMPS(input, output);

    return sd::Status::OK;
}

PLATFORM_CHECK(silu, ENGINE_CPU) {
    auto input = INPUT_VARIABLE(0);

    Requirements req("MPS SILU OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(input->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*input), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
