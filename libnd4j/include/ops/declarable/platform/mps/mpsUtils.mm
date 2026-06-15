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
// Metal Performance Shaders (MPS) helper utilities implementation
//

#include "mpsUtils.h"
#include <ConstMessages.h>
#include <system/Environment.h>

namespace sd {
namespace ops {
namespace mpsUtils {

bool hasMPSSupport() {
#ifdef HAVE_MPS
    return MPSDeviceManager::getInstance().isAvailable();
#else
    return false;
#endif
}

#ifdef HAVE_MPS

// ============================================================================
// MPSDeviceManager Implementation
// ============================================================================

MPSDeviceManager::MPSDeviceManager()
    : _device(nil), _commandQueue(nil), _initialized(false) {
}

MPSDeviceManager::~MPSDeviceManager() {
    shutdown();
}

MPSDeviceManager& MPSDeviceManager::getInstance() {
    static MPSDeviceManager instance;
    if (!instance._initialized) {
        instance.initialize();
    }
    return instance;
}

bool MPSDeviceManager::initialize() {
    if (_initialized) {
        return _device != nil;
    }

    @autoreleasepool {
        // Get the default Metal device (usually the discrete GPU if available)
        _device = MTLCreateSystemDefaultDevice();

        if (_device == nil) {
            _initialized = true;
            return false;
        }

        // Check if the device supports MPS
        if (![_device supportsFamily:MTLGPUFamilyApple1] &&
            ![_device supportsFamily:MTLGPUFamilyMac1]) {
            _device = nil;
            _initialized = true;
            return false;
        }

        // Create command queue
        _commandQueue = [_device newCommandQueue];

        if (_commandQueue == nil) {
            _device = nil;
            _initialized = true;
            return false;
        }

        _initialized = true;
        return true;
    }
}

void MPSDeviceManager::shutdown() {
    @autoreleasepool {
        _commandQueue = nil;
        _device = nil;
        _initialized = false;
    }
}

id<MTLCommandBuffer> MPSDeviceManager::createCommandBuffer() {
    if (_commandQueue == nil) {
        return nil;
    }
    return [_commandQueue commandBuffer];
}

std::string MPSDeviceManager::getDeviceName() const {
    if (_device == nil) {
        return "No Metal device";
    }
    return std::string([[_device name] UTF8String]);
}

size_t MPSDeviceManager::getMaxMemory() const {
    if (_device == nil) {
        return 0;
    }
    return [_device recommendedMaxWorkingSetSize];
}

bool MPSDeviceManager::supportsFamily(MTLGPUFamily family) const {
    if (_device == nil) {
        return false;
    }
    return [_device supportsFamily:family];
}

// ============================================================================
// Utility Functions
// ============================================================================

bool isMPSSupported(sd::DataType dtype) {
    switch (dtype) {
        case sd::DataType::FLOAT32:
        case sd::DataType::FLOAT16:
        case sd::DataType::INT32:
        case sd::DataType::INT16:
        case sd::DataType::INT8:
        case sd::DataType::UINT8:
            return true;
        default:
            return false;
    }
}

bool isContiguous(const NDArray& arr) {
    auto rank = arr.rankOf();
    auto shape = arr.shapeOf();
    auto strides = arr.stridesOf();
    sd::LongType expected = 1;
    for (int i = rank - 1; i >= 0; --i) {
      if (shape[i] == 1) continue;
      if (strides[i] != expected) return false;
      expected *= shape[i];
    }
    return true;
}

bool isMPSFriendly(const NDArray& arr) {
    if (!isContiguous(arr)) {
        return false;
    }
    if (!isMPSSupported(arr.dataType())) {
        return false;
    }
    if (arr.isEmpty()) {
        return false;
    }
    return true;
}

MPSDataType getMPSDataType(sd::DataType dtype) {
    switch (dtype) {
        case sd::DataType::FLOAT32:
            return MPSDataTypeFloat32;
        case sd::DataType::FLOAT16:
            return MPSDataTypeFloat16;
        case sd::DataType::INT32:
            return MPSDataTypeInt32;
        case sd::DataType::INT16:
            return MPSDataTypeInt16;
        case sd::DataType::INT8:
            return MPSDataTypeInt8;
        case sd::DataType::UINT8:
            return MPSDataTypeUInt8;
        default:
            return MPSDataTypeFloat32;
    }
}

MPSMatrix* createMPSMatrix(const NDArray* arr, id<MTLDevice> device) {
    if (arr == nullptr || device == nil) {
        return nil;
    }

    @autoreleasepool {
        // Get matrix dimensions
        // For 2D: [rows, cols]
        // For higher dimensions: flatten to 2D
        NSUInteger rows, cols;
        if (arr->rankOf() == 1) {
            rows = 1;
            cols = arr->lengthOf();
        } else if (arr->rankOf() == 2) {
            rows = arr->sizeAt(0);
            cols = arr->sizeAt(1);
        } else {
            // Flatten higher dimensions into rows
            rows = 1;
            for (int i = 0; i < arr->rankOf() - 1; i++) {
                rows *= arr->sizeAt(i);
            }
            cols = arr->sizeAt(arr->rankOf() - 1);
        }

        // Calculate row bytes (must be aligned to 16 bytes for MPS)
        size_t elementSize = arr->sizeOfT();
        size_t rowBytes = cols * elementSize;
        rowBytes = (rowBytes + 15) & ~15;  // Align to 16 bytes

        // Create Metal buffer from array data
        size_t bufferSize = rows * rowBytes;
        id<MTLBuffer> buffer = [device newBufferWithBytes:arr->buffer()
                                                   length:bufferSize
                                                  options:MTLResourceStorageModeShared];

        if (buffer == nil) {
            return nil;
        }

        // Create matrix descriptor
        MPSMatrixDescriptor* descriptor = [MPSMatrixDescriptor
            matrixDescriptorWithRows:rows
                             columns:cols
                            rowBytes:rowBytes
                            dataType:getMPSDataType(arr->dataType())];

        // Create MPS matrix
        MPSMatrix* matrix = [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];

        return matrix;
    }
}

MPSImage* createMPSImage(const NDArray* arr, id<MTLDevice> device) {
    if (arr == nullptr || device == nil || arr->rankOf() != 4) {
        return nil;
    }

    @autoreleasepool {
        // Assuming NCHW format: [batch, channels, height, width]
        NSUInteger batch = arr->sizeAt(0);
        NSUInteger channels = arr->sizeAt(1);
        NSUInteger height = arr->sizeAt(2);
        NSUInteger width = arr->sizeAt(3);

        // Create image descriptor
        MPSImageDescriptor* descriptor = [MPSImageDescriptor
            imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                       width:width
                                      height:height
                             featureChannels:channels
                              numberOfImages:batch
                                       usage:MTLTextureUsageShaderRead | MTLTextureUsageShaderWrite];

        // Create MPS image
        MPSImage* image = [[MPSImage alloc] initWithDevice:device imageDescriptor:descriptor];

        // Copy data to image
        // Note: This is a simplified version - actual implementation needs proper data layout handling
        MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, 1);
        size_t bytesPerRow = width * sizeof(float);

        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                const float* srcPtr = arr->bufferAsT<float>() +
                                      n * channels * height * width +
                                      c * height * width;
                [image.texture replaceRegion:region
                                 mipmapLevel:0
                                       slice:n * channels + c
                                   withBytes:srcPtr
                                 bytesPerRow:bytesPerRow
                               bytesPerImage:0];
            }
        }

        return image;
    }
}

void copyMPSMatrixToNDArray(MPSMatrix* matrix, NDArray* arr) {
    if (matrix == nil || arr == nullptr) {
        return;
    }

    @autoreleasepool {
        // Synchronize to ensure GPU operations are complete
        id<MTLCommandBuffer> commandBuffer = MPSDeviceManager::getInstance().createCommandBuffer();
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy data from Metal buffer to NDArray
        void* bufferContents = [matrix.data contents];
        size_t copySize = std::min(static_cast<size_t>(matrix.rows * matrix.rowBytes),
                                   arr->lengthOf() * arr->sizeOfT());
        memcpy(arr->buffer(), bufferContents, copySize);
    }
}

void copyMPSImageToNDArray(MPSImage* image, NDArray* arr) {
    if (image == nil || arr == nullptr || arr->rankOf() != 4) {
        return;
    }

    @autoreleasepool {
        NSUInteger batch = arr->sizeAt(0);
        NSUInteger channels = arr->sizeAt(1);
        NSUInteger height = arr->sizeAt(2);
        NSUInteger width = arr->sizeAt(3);

        MTLRegion region = MTLRegionMake3D(0, 0, 0, width, height, 1);
        size_t bytesPerRow = width * sizeof(float);

        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                float* dstPtr = arr->bufferAsT<float>() +
                                n * channels * height * width +
                                c * height * width;
                [image.texture getBytes:dstPtr
                            bytesPerRow:bytesPerRow
                          bytesPerImage:0
                             fromRegion:region
                            mipmapLevel:0
                                  slice:n * channels + c];
            }
        }
    }
}

void synchronize() {
    @autoreleasepool {
        id<MTLCommandBuffer> commandBuffer = MPSDeviceManager::getInstance().createCommandBuffer();
        if (commandBuffer != nil) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }
    }
}

// ============================================================================
// MPSCommandBufferGuard Implementation
// ============================================================================

MPSCommandBufferGuard::MPSCommandBufferGuard()
    : _commandBuffer(nil), _committed(false) {
    _commandBuffer = MPSDeviceManager::getInstance().createCommandBuffer();
}

MPSCommandBufferGuard::~MPSCommandBufferGuard() {
    if (!_committed && _commandBuffer != nil) {
        commitAndWait();
    }
}

void MPSCommandBufferGuard::commitAndWait() {
    if (_commandBuffer != nil && !_committed) {
        [_commandBuffer commit];
        [_commandBuffer waitUntilCompleted];
        _committed = true;
    }
}

void MPSCommandBufferGuard::commit() {
    if (_commandBuffer != nil && !_committed) {
        [_commandBuffer commit];
        _committed = true;
    }
}

void checkMPSRequirements(Requirements& reqs, sd::graph::Context& block,
                           const NDArray* input, const NDArray* output) {
    reqs.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    reqs.expectTrue(MPSDeviceManager::getInstance().isAvailable(), "MPS device must be available");

    if (input != nullptr) {
        reqs.expectTrue(isMPSFriendly(*input), "Input must be contiguous with MPS-supported data type");
    }

    if (output != nullptr) {
        reqs.expectTrue(isMPSFriendly(*output), "Output must be contiguous with MPS-supported data type");
    }
}

#endif  // HAVE_MPS

}  // namespace mpsUtils
}  // namespace ops
}  // namespace sd
