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
#include <helpers/shape.h>

#include <system/BackendNamespace.h>

namespace sd {
namespace ops {
SD_BACKEND_OPS_INLINE_NAMESPACE_BEGIN
namespace mpsUtils {

// ---------------------------------------------------------------------------
// hasMPSSupport / isContiguous — always compiled, no #ifdef guards
// ---------------------------------------------------------------------------

bool hasMPSSupport() {
    return MPSDeviceManager::getInstance().isAvailable();
}

bool isContiguous(const sd::NDArray& arr) {
    // Use the canonical contiguity check — never ews().
    return shape::strideDescendingCAscendingF(
        const_cast<sd::LongType*>(arr.shapeInfo()));
}

// ---------------------------------------------------------------------------
// MPSDeviceManager — constructor/destructor always compiled
// ---------------------------------------------------------------------------

MPSDeviceManager::MPSDeviceManager()
    : _initialized(false), _available(false)
#ifdef HAVE_MPS
    , _device(nil), _commandQueue(nil)
#endif
{}

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
    if (_initialized) return _available;

#ifdef HAVE_MPS
    @autoreleasepool {
        _device = MTLCreateSystemDefaultDevice();
        if (_device == nil) {
            _initialized = true;
            _available   = false;
            return false;
        }

        // Require at least Apple1 (iOS) or Mac1 (macOS) GPU family for MPS.
        if (![_device supportsFamily:MTLGPUFamilyApple1] &&
            ![_device supportsFamily:MTLGPUFamilyMac1]) {
            _device      = nil;
            _initialized = true;
            _available   = false;
            return false;
        }

        _commandQueue = [_device newCommandQueue];
        if (_commandQueue == nil) {
            _device      = nil;
            _initialized = true;
            _available   = false;
            return false;
        }

        _available   = true;
        _initialized = true;
        return true;
    }
#else
    _initialized = true;
    _available   = false;
    return false;
#endif
}

void MPSDeviceManager::shutdown() {
#ifdef HAVE_MPS
    @autoreleasepool {
        _commandQueue = nil;
        _device       = nil;
    }
#endif
    _initialized = false;
    _available   = false;
}

// ---------------------------------------------------------------------------
// Methods that reference Metal objects: only compiled under HAVE_MPS.
// On Apple platforms without MPS this header-declared section is omitted.
// ---------------------------------------------------------------------------

#ifdef HAVE_MPS

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
    return static_cast<size_t>([_device recommendedMaxWorkingSetSize]);
}

bool MPSDeviceManager::supportsFamily(MTLGPUFamily family) const {
    if (_device == nil) {
        return false;
    }
    return [_device supportsFamily:family];
}

// ============================================================================
// MPS-only utility functions
// ============================================================================

bool isMPSSupported(sd::DataType dtype) {
    switch (dtype) {
        case sd::DataType::FLOAT32:
        case sd::DataType::HALF:    // float16
        case sd::DataType::INT32:
        case sd::DataType::INT16:
        case sd::DataType::INT8:
        case sd::DataType::UINT8:
            return true;
        default:
            return false;
    }
}

bool isMPSFriendly(const sd::NDArray& arr) {
    return isContiguous(arr) && isMPSSupported(arr.dataType()) && !arr.isEmpty();
}

MPSDataType getMPSDataType(sd::DataType dtype) {
    switch (dtype) {
        case sd::DataType::FLOAT32:  return MPSDataTypeFloat32;
        case sd::DataType::HALF:     return MPSDataTypeFloat16;  // float16
        case sd::DataType::INT32:    return MPSDataTypeInt32;
        case sd::DataType::INT16:    return MPSDataTypeInt16;
        case sd::DataType::INT8:     return MPSDataTypeInt8;
        case sd::DataType::UINT8:    return MPSDataTypeUInt8;
        default:                     return MPSDataTypeFloat32;
    }
}

MPSMatrix* createMPSMatrix(const sd::NDArray* arr, id<MTLDevice> device) {
    if (arr == nullptr || device == nil) {
        return nil;
    }

    @autoreleasepool {
        // Determine matrix dimensions.
        // 1-D → [1, length], 2-D → [rows, cols], N-D → [prod(dims[0..N-2]), dims[N-1]]
        NSUInteger rows, cols;
        if (arr->rankOf() == 1) {
            rows = 1;
            cols = static_cast<NSUInteger>(arr->lengthOf());
        } else if (arr->rankOf() == 2) {
            rows = static_cast<NSUInteger>(arr->sizeAt(0));
            cols = static_cast<NSUInteger>(arr->sizeAt(1));
        } else {
            rows = 1;
            for (int i = 0; i < arr->rankOf() - 1; i++) {
                rows *= static_cast<NSUInteger>(arr->sizeAt(i));
            }
            cols = static_cast<NSUInteger>(arr->sizeAt(arr->rankOf() - 1));
        }

        // Row stride must be aligned to 16 bytes for MPS.
        size_t elementSize = arr->sizeOfT();
        size_t rowBytes = cols * elementSize;
        rowBytes = (rowBytes + 15) & ~static_cast<size_t>(15);

        // Wrap the host buffer in a shared MTLBuffer (zero-copy on unified memory).
        size_t bufferSize = rows * rowBytes;
        id<MTLBuffer> buffer = [device newBufferWithBytes:arr->buffer()
                                                   length:bufferSize
                                                  options:MTLResourceStorageModeShared];
        if (buffer == nil) {
            return nil;
        }

        MPSMatrixDescriptor* descriptor =
            [MPSMatrixDescriptor matrixDescriptorWithRows:rows
                                                  columns:cols
                                                 rowBytes:rowBytes
                                                 dataType:getMPSDataType(arr->dataType())];

        return [[MPSMatrix alloc] initWithBuffer:buffer descriptor:descriptor];
    }
}

MPSImage* createMPSImage(const sd::NDArray* arr, id<MTLDevice> device) {
    if (arr == nullptr || device == nil || arr->rankOf() != 4) {
        return nil;
    }

    @autoreleasepool {
        // Expecting NCHW layout: [batch, channels, height, width]
        NSUInteger batch    = static_cast<NSUInteger>(arr->sizeAt(0));
        NSUInteger channels = static_cast<NSUInteger>(arr->sizeAt(1));
        NSUInteger height   = static_cast<NSUInteger>(arr->sizeAt(2));
        NSUInteger width    = static_cast<NSUInteger>(arr->sizeAt(3));

        MPSImageDescriptor* descriptor =
            [MPSImageDescriptor imageDescriptorWithChannelFormat:MPSImageFeatureChannelFormatFloat32
                                                           width:width
                                                          height:height
                                                 featureChannels:channels
                                                  numberOfImages:batch
                                                           usage:MTLTextureUsageShaderRead |
                                                                 MTLTextureUsageShaderWrite];

        MPSImage* image = [[MPSImage alloc] initWithDevice:device imageDescriptor:descriptor];

        // Upload each slice (one per channel per batch element).
        MTLRegion region        = MTLRegionMake3D(0, 0, 0, width, height, 1);
        size_t    bytesPerRow   = width * sizeof(float);

        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                const float* srcPtr = arr->bufferAsT<float>() +
                                      (n * channels + c) * height * width;
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

void copyMPSMatrixToNDArray(MPSMatrix* matrix, sd::NDArray* arr) {
    if (matrix == nil || arr == nullptr) {
        return;
    }

    @autoreleasepool {
        // Flush any pending GPU work before reading back.
        id<MTLCommandBuffer> commandBuffer =
            MPSDeviceManager::getInstance().createCommandBuffer();
        if (commandBuffer != nil) {
            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];
        }

        void*  bufferContents = [matrix.data contents];
        size_t copySize = std::min(
            static_cast<size_t>(matrix.rows * matrix.rowBytes),
            static_cast<size_t>(arr->lengthOf()) * arr->sizeOfT());
        memcpy(arr->buffer(), bufferContents, copySize);
    }
}

void copyMPSImageToNDArray(MPSImage* image, sd::NDArray* arr) {
    if (image == nil || arr == nullptr || arr->rankOf() != 4) {
        return;
    }

    @autoreleasepool {
        NSUInteger batch    = static_cast<NSUInteger>(arr->sizeAt(0));
        NSUInteger channels = static_cast<NSUInteger>(arr->sizeAt(1));
        NSUInteger height   = static_cast<NSUInteger>(arr->sizeAt(2));
        NSUInteger width    = static_cast<NSUInteger>(arr->sizeAt(3));

        MTLRegion region      = MTLRegionMake3D(0, 0, 0, width, height, 1);
        size_t    bytesPerRow = width * sizeof(float);

        for (NSUInteger n = 0; n < batch; n++) {
            for (NSUInteger c = 0; c < channels; c++) {
                float* dstPtr = arr->bufferAsT<float>() +
                                (n * channels + c) * height * width;
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
        id<MTLCommandBuffer> commandBuffer =
            MPSDeviceManager::getInstance().createCommandBuffer();
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

// ============================================================================
// checkMPSRequirements
// ============================================================================

void checkMPSRequirements(sd::Requirements& reqs, sd::graph::Context& block,
                           const sd::NDArray* input, const sd::NDArray* output) {
    reqs.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    reqs.expectTrue(MPSDeviceManager::getInstance().isAvailable(),
                    "MPS device must be available");

    if (input != nullptr) {
        reqs.expectTrue(isMPSFriendly(*input),
                        "Input must be contiguous with an MPS-supported data type");
    }

    if (output != nullptr) {
        reqs.expectTrue(isMPSFriendly(*output),
                        "Output must be contiguous with an MPS-supported data type");
    }
}

#endif  // HAVE_MPS

}  // namespace mpsUtils
SD_BACKEND_OPS_INLINE_NAMESPACE_END
}  // namespace ops
}  // namespace sd
