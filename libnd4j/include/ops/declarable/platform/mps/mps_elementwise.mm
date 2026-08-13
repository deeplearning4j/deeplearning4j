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
// Metal Performance Shaders - Element-wise operations
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
// Element-wise Addition using MPS
//////////////////////////////////////////////////////////////////////////

static void addMPS(NDArray* x, NDArray* y, NDArray* z) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger length = z->lengthOf();
        size_t bufferSize = length * sizeof(float);

        // Create Metal buffers
        id<MTLBuffer> bufferX = [device newBufferWithBytes:x->buffer()
                                                    length:x->lengthOf() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferY = [device newBufferWithBytes:y->buffer()
                                                    length:y->lengthOf() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferZ = [device newBufferWithLength:bufferSize
                                                    options:MTLResourceStorageModeShared];

        // Use MPSNDArray for element-wise operations (macOS 10.15+, iOS 13+)
        if (@available(macOS 10.15, iOS 13.0, *)) {
            // Create matrix descriptors for element-wise operation
            // Treating as 1D vectors for simplicity
            MPSMatrixDescriptor* descX = [MPSMatrixDescriptor
                matrixDescriptorWithRows:1
                                 columns:x->lengthOf()
                                rowBytes:x->lengthOf() * sizeof(float)
                                dataType:MPSDataTypeFloat32];

            MPSMatrixDescriptor* descY = [MPSMatrixDescriptor
                matrixDescriptorWithRows:1
                                 columns:y->lengthOf()
                                rowBytes:y->lengthOf() * sizeof(float)
                                dataType:MPSDataTypeFloat32];

            MPSMatrixDescriptor* descZ = [MPSMatrixDescriptor
                matrixDescriptorWithRows:1
                                 columns:length
                                rowBytes:length * sizeof(float)
                                dataType:MPSDataTypeFloat32];

            MPSMatrix* matrixX = [[MPSMatrix alloc] initWithBuffer:bufferX descriptor:descX];
            MPSMatrix* matrixY = [[MPSMatrix alloc] initWithBuffer:bufferY descriptor:descY];
            MPSMatrix* matrixZ = [[MPSMatrix alloc] initWithBuffer:bufferZ descriptor:descZ];

            // Use MPSMatrixSum for addition (alpha * A + beta * B)
            MPSMatrixSum* matrixSum = [[MPSMatrixSum alloc]
                initWithDevice:device
                         count:2
                          rows:1
                       columns:length
                     transpose:NO];

            id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();

            // MPSMatrixSum adds multiple matrices: result = sum of all inputs
            NSArray<MPSMatrix*>* matrices = @[matrixX, matrixY];
            [matrixSum encodeToCommandBuffer:commandBuffer
                              sourceMatrices:matrices
                              resultMatrix:matrixZ
                                     scale:nil
                             offsetVector:nil
                             biasVector:nil
                               startIndex:0];

            [commandBuffer commit];
            [commandBuffer waitUntilCompleted];

            memcpy(z->buffer(), [bufferZ contents], bufferSize);
        } else {
            // Fallback for older systems - use CPU
            const float* xPtr = x->bufferAsT<float>();
            const float* yPtr = y->bufferAsT<float>();
            float* zPtr = z->bufferAsT<float>();

            // Handle broadcasting
            if (x->lengthOf() == y->lengthOf()) {
                for (NSUInteger i = 0; i < length; i++) {
                    zPtr[i] = xPtr[i] + yPtr[i];
                }
            } else if (y->lengthOf() == 1) {
                float scalar = yPtr[0];
                for (NSUInteger i = 0; i < length; i++) {
                    zPtr[i] = xPtr[i] + scalar;
                }
            } else if (x->lengthOf() == 1) {
                float scalar = xPtr[0];
                for (NSUInteger i = 0; i < length; i++) {
                    zPtr[i] = scalar + yPtr[i];
                }
            }
        }
    }
}

PLATFORM_IMPL(add, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    addMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(add, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS ADD OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(x->dataType() == y->dataType(),
                   "Input arrays must have the same data type");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Element-wise Multiplication using MPS
//////////////////////////////////////////////////////////////////////////

static void multiplyMPS(NDArray* x, NDArray* y, NDArray* z) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) return;

        NSUInteger length = z->lengthOf();
        size_t bufferSize = length * sizeof(float);

        // Create Metal buffers
        id<MTLBuffer> bufferX = [device newBufferWithBytes:x->buffer()
                                                    length:x->lengthOf() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferY = [device newBufferWithBytes:y->buffer()
                                                    length:y->lengthOf() * sizeof(float)
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferZ = [device newBufferWithLength:bufferSize
                                                    options:MTLResourceStorageModeShared];

        // For element-wise multiplication, we can use Hadamard product
        // MPS doesn't have a direct element-wise multiply, so we use compute shader approach
        // or fall back to CPU for now

        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();

        // Handle broadcasting
        if (x->lengthOf() == y->lengthOf()) {
            // Use vDSP for vectorized multiplication (Accelerate framework)
            // vDSP_vmul(xPtr, 1, yPtr, 1, zPtr, 1, length);
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] * yPtr[i];
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] * scalar;
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = scalar * yPtr[i];
            }
        } else {
            // General broadcasting - more complex
            // For now, element-wise with same shape
            NSUInteger minLen = std::min(x->lengthOf(), y->lengthOf());
            for (NSUInteger i = 0; i < minLen; i++) {
                zPtr[i] = xPtr[i % x->lengthOf()] * yPtr[i % y->lengthOf()];
            }
        }
    }
}

PLATFORM_IMPL(multiply, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    multiplyMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(multiply, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS MULTIPLY OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(x->dataType() == y->dataType(),
                   "Input arrays must have the same data type");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Element-wise Subtraction
//////////////////////////////////////////////////////////////////////////

static void subtractMPS(NDArray* x, NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        NSUInteger length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] - yPtr[i];
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] - scalar;
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = scalar - yPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(subtract, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    subtractMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(subtract, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS SUBTRACT OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Element-wise Division
//////////////////////////////////////////////////////////////////////////

static void divideMPS(NDArray* x, NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        NSUInteger length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] / yPtr[i];
            }
        } else if (y->lengthOf() == 1) {
            float scalar = yPtr[0];
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = xPtr[i] / scalar;
            }
        } else if (x->lengthOf() == 1) {
            float scalar = xPtr[0];
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = scalar / yPtr[i];
            }
        }
    }
}

PLATFORM_IMPL(divide, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    divideMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(divide, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS DIVIDE OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Element-wise Square Root
//////////////////////////////////////////////////////////////////////////

static void sqrtMPS(NDArray* x, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        NSUInteger length = z->lengthOf();

        for (NSUInteger i = 0; i < length; i++) {
            zPtr[i] = sqrtf(xPtr[i]);
        }
    }
}

PLATFORM_IMPL(sqrt, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    sqrtMPS(x, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(sqrt, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS SQRT OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Element-wise Exponential
//////////////////////////////////////////////////////////////////////////

static void expMPS(NDArray* x, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        NSUInteger length = z->lengthOf();

        for (NSUInteger i = 0; i < length; i++) {
            zPtr[i] = expf(xPtr[i]);
        }
    }
}

PLATFORM_IMPL(exp, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    expMPS(x, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(exp, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS EXP OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Element-wise Natural Logarithm
//////////////////////////////////////////////////////////////////////////

static void logMPS(NDArray* x, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        NSUInteger length = z->lengthOf();

        for (NSUInteger i = 0; i < length; i++) {
            zPtr[i] = logf(xPtr[i]);
        }
    }
}

PLATFORM_IMPL(log, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    logMPS(x, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(log, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS LOG OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Element-wise Power
//////////////////////////////////////////////////////////////////////////

static void powMPS(NDArray* x, NDArray* y, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        const float* yPtr = y->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        NSUInteger length = z->lengthOf();

        if (x->lengthOf() == y->lengthOf()) {
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = powf(xPtr[i], yPtr[i]);
            }
        } else if (y->lengthOf() == 1) {
            float exponent = yPtr[0];
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = powf(xPtr[i], exponent);
            }
        } else if (x->lengthOf() == 1) {
            float base = xPtr[0];
            for (NSUInteger i = 0; i < length; i++) {
                zPtr[i] = powf(base, yPtr[i]);
            }
        }
    }
}

PLATFORM_IMPL(pow, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty() || y->isEmpty()) return sd::Status::OK;

    powMPS(x, y, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(pow, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto y = INPUT_VARIABLE(1);

    Requirements req("MPS POW OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input X must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*y), "Input Y must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Element-wise Absolute Value
//////////////////////////////////////////////////////////////////////////

static void absMPS(NDArray* x, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        NSUInteger length = z->lengthOf();

        for (NSUInteger i = 0; i < length; i++) {
            zPtr[i] = fabsf(xPtr[i]);
        }
    }
}

PLATFORM_IMPL(abs, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    absMPS(x, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(abs, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS ABS OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

//////////////////////////////////////////////////////////////////////////
// Element-wise Negative
//////////////////////////////////////////////////////////////////////////

static void negMPS(NDArray* x, NDArray* z) {
    @autoreleasepool {
        const float* xPtr = x->bufferAsT<float>();
        float* zPtr = z->bufferAsT<float>();
        NSUInteger length = z->lengthOf();

        for (NSUInteger i = 0; i < length; i++) {
            zPtr[i] = -xPtr[i];
        }
    }
}

PLATFORM_IMPL(neg, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);
    auto z = OUTPUT_VARIABLE(0);

    if (x->isEmpty()) return sd::Status::OK;

    negMPS(x, z);

    return sd::Status::OK;
}

PLATFORM_CHECK(neg, ENGINE_CPU) {
    auto x = INPUT_VARIABLE(0);

    Requirements req("MPS NEG OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(x->dataType() == DataType::FLOAT32,
                   "Only float32 is supported");
    req.expectTrue(mpsUtils::isContiguous(*x), "Input must be contiguous");

    req.logTheSuccess();
    return req;
}

#else  // !HAVE_MPS — register no-op stubs so the helper entries exist on all builds

PLATFORM_IMPL(add, ENGINE_CPU)      { return sd::Status::OK; }
PLATFORM_CHECK(add, ENGINE_CPU)     { return false; }

PLATFORM_IMPL(subtract, ENGINE_CPU) { return sd::Status::OK; }
PLATFORM_CHECK(subtract, ENGINE_CPU){ return false; }

PLATFORM_IMPL(multiply, ENGINE_CPU) { return sd::Status::OK; }
PLATFORM_CHECK(multiply, ENGINE_CPU){ return false; }

PLATFORM_IMPL(divide, ENGINE_CPU)   { return sd::Status::OK; }
PLATFORM_CHECK(divide, ENGINE_CPU)  { return false; }

PLATFORM_IMPL(pow, ENGINE_CPU)      { return sd::Status::OK; }
PLATFORM_CHECK(pow, ENGINE_CPU)     { return false; }

PLATFORM_IMPL(sqrt, ENGINE_CPU)     { return sd::Status::OK; }
PLATFORM_CHECK(sqrt, ENGINE_CPU)    { return false; }

PLATFORM_IMPL(exp, ENGINE_CPU)      { return sd::Status::OK; }
PLATFORM_CHECK(exp, ENGINE_CPU)     { return false; }

PLATFORM_IMPL(log, ENGINE_CPU)      { return sd::Status::OK; }
PLATFORM_CHECK(log, ENGINE_CPU)     { return false; }

PLATFORM_IMPL(abs, ENGINE_CPU)      { return sd::Status::OK; }
PLATFORM_CHECK(abs, ENGINE_CPU)     { return false; }

PLATFORM_IMPL(neg, ENGINE_CPU)      { return sd::Status::OK; }
PLATFORM_CHECK(neg, ENGINE_CPU)     { return false; }

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
