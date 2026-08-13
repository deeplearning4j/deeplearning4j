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
// Metal Performance Shaders - BLAS operations (matrix multiplication)
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

/**
 * Matrix multiplication using MPS
 * C = alpha * op(A) * op(B) + beta * C
 */
static void matmulMPS(NDArray* a, NDArray* b, NDArray* c,
                       bool transA, bool transB, double alpha, double beta) {
    @autoreleasepool {
        auto& manager = mpsUtils::MPSDeviceManager::getInstance();
        id<MTLDevice> device = manager.getDevice();

        if (device == nil) {
            return;
        }

        // Get dimensions
        NSUInteger M = transA ? a->sizeAt(1) : a->sizeAt(0);
        NSUInteger K = transA ? a->sizeAt(0) : a->sizeAt(1);
        NSUInteger N = transB ? b->sizeAt(0) : b->sizeAt(1);

        // Create Metal buffers
        size_t aSize = a->lengthOf() * a->sizeOfT();
        size_t bSize = b->lengthOf() * b->sizeOfT();
        size_t cSize = c->lengthOf() * c->sizeOfT();

        id<MTLBuffer> bufferA = [device newBufferWithBytes:a->buffer()
                                                    length:aSize
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferB = [device newBufferWithBytes:b->buffer()
                                                    length:bSize
                                                   options:MTLResourceStorageModeShared];
        id<MTLBuffer> bufferC = [device newBufferWithBytes:c->buffer()
                                                    length:cSize
                                                   options:MTLResourceStorageModeShared];

        // Row bytes (aligned to 16 bytes)
        NSUInteger rowBytesA = ((transA ? M : K) * sizeof(float) + 15) & ~15;
        NSUInteger rowBytesB = ((transB ? K : N) * sizeof(float) + 15) & ~15;
        NSUInteger rowBytesC = (N * sizeof(float) + 15) & ~15;

        // Create matrix descriptors
        MPSMatrixDescriptor* descA = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(transA ? K : M)
                             columns:(transA ? M : K)
                            rowBytes:rowBytesA
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descB = [MPSMatrixDescriptor
            matrixDescriptorWithRows:(transB ? N : K)
                             columns:(transB ? K : N)
                            rowBytes:rowBytesB
                            dataType:MPSDataTypeFloat32];

        MPSMatrixDescriptor* descC = [MPSMatrixDescriptor
            matrixDescriptorWithRows:M
                             columns:N
                            rowBytes:rowBytesC
                            dataType:MPSDataTypeFloat32];

        // Create MPS matrices
        MPSMatrix* matrixA = [[MPSMatrix alloc] initWithBuffer:bufferA descriptor:descA];
        MPSMatrix* matrixB = [[MPSMatrix alloc] initWithBuffer:bufferB descriptor:descB];
        MPSMatrix* matrixC = [[MPSMatrix alloc] initWithBuffer:bufferC descriptor:descC];

        // Create matrix multiplication kernel
        MPSMatrixMultiplication* matmul = [[MPSMatrixMultiplication alloc]
            initWithDevice:device
               transposeLeft:transA
              transposeRight:transB
                  resultRows:M
               resultColumns:N
             interiorColumns:K
                       alpha:alpha
                        beta:beta];

        // Create command buffer and encode
        id<MTLCommandBuffer> commandBuffer = manager.createCommandBuffer();
        [matmul encodeToCommandBuffer:commandBuffer
                           leftMatrix:matrixA
                          rightMatrix:matrixB
                         resultMatrix:matrixC];

        // Execute and wait
        [commandBuffer commit];
        [commandBuffer waitUntilCompleted];

        // Copy result back to NDArray
        memcpy(c->buffer(), [bufferC contents], cSize);
    }
}

PLATFORM_IMPL(matmul, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);
    auto c = OUTPUT_VARIABLE(0);

    if (a->isEmpty() || b->isEmpty()) return sd::Status::OK;

    // Get transpose flags
    int transA = 0, transB = 0;
    if (block.getIArguments()->size() > 0) transA = INT_ARG(0);
    if (block.getIArguments()->size() > 1) transB = INT_ARG(1);

    // Get alpha/beta
    double alpha = 1.0, beta = 0.0;
    if (block.getTArguments()->size() > 0) alpha = T_ARG(0);
    if (block.getTArguments()->size() > 1) beta = T_ARG(1);

    // Flatten all leading dimensions into one batch dimension, then use the
    // current pointer-returning NDArray view API for each rank-2 matrix.
    if (a->rankOf() > 2) {
        std::vector<LongType> aShape = {-1, a->sizeAt(-2), a->sizeAt(-1)};
        std::vector<LongType> bShape = {-1, b->sizeAt(-2), b->sizeAt(-1)};
        std::vector<LongType> cShape = {-1, c->sizeAt(-2), c->sizeAt(-1)};
        NDArray* aReshape = a->reshape(a->ordering(), aShape);
        NDArray* bReshape = b->reshape(b->ordering(), bShape);
        NDArray* cReshape = c->reshape(c->ordering(), cShape, false);

        try {
            LongType batchSize = aReshape->sizeAt(0);
            for (LongType i = 0; i < batchSize; i++) {
                NDArray* aSub = (*aReshape)(i, {0});
                NDArray* bSub = (*bReshape)(i, {0});
                NDArray* cSub = (*cReshape)(i, {0});
                try {
                    matmulMPS(aSub, bSub, cSub, transA != 0, transB != 0, alpha, beta);
                } catch (...) {
                    delete aSub;
                    delete bSub;
                    delete cSub;
                    throw;
                }
                delete aSub;
                delete bSub;
                delete cSub;
            }
        } catch (...) {
            delete aReshape;
            delete bReshape;
            delete cReshape;
            throw;
        }

        delete aReshape;
        delete bReshape;
        delete cReshape;
    } else {
        matmulMPS(a, b, c, transA != 0, transB != 0, alpha, beta);
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(matmul, ENGINE_CPU) {
    auto a = INPUT_VARIABLE(0);
    auto b = INPUT_VARIABLE(1);

    Requirements req("MPS MATMUL OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(a->dataType() == DataType::FLOAT32,
                   "Only float32 is currently supported by MPS matmul");
    req.expectTrue(a->dataType() == b->dataType(),
                   "Input arrays must have the same data type");
    req.expectTrue(mpsUtils::isContiguous(*a), "Input A must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*b), "Input B must be contiguous");
    req.expectFalse(a->isEmpty(), "Input A must not be empty");
    req.expectFalse(b->isEmpty(), "Input B must not be empty");

    req.logTheSuccess();
    return req;
}

// ===========================================================================
// batched_gemm
// Loop over the batch dimension, applying matmulMPS for each slice.
// Input shapes: A[bS, M, K], B[bS, K, N], output C[bS, M, N].
// iArgs: 0=transA, 1=transB
// tArgs: 0=alpha,  1=beta
// ===========================================================================

PLATFORM_IMPL(batched_gemm, ENGINE_CPU) {
    auto A = INPUT_VARIABLE(0);
    auto B = INPUT_VARIABLE(1);
    auto C = OUTPUT_VARIABLE(0);

    if (A->isEmpty() || B->isEmpty()) return sd::Status::OK;

    bool transA = block.getIArguments()->size() > 0 ? (bool)INT_ARG(0) : false;
    bool transB = block.getIArguments()->size() > 1 ? (bool)INT_ARG(1) : false;
    double alpha = block.getTArguments()->size() > 0 ? T_ARG(0) : 1.0;
    double beta  = block.getTArguments()->size() > 1 ? T_ARG(1) : 0.0;

    sd::LongType rank = A->rankOf();
    sd::LongType bS   = (rank >= 3) ? A->sizeAt(0) : 1;
    sd::LongType M    = transA ? A->sizeAt(-1) : A->sizeAt(-2);
    sd::LongType K    = transA ? A->sizeAt(-2) : A->sizeAt(-1);
    sd::LongType N    = transB ? B->sizeAt(-2) : B->sizeAt(-1);

    if (rank == 2) {
        matmulMPS(A, B, C, transA, transB, alpha, beta);
    } else {
        for (sd::LongType i = 0; i < bS; i++) {
            NDArray* aSlice = (*A)(i, {0});
            NDArray* bSlice = (*B)(i, {0});
            NDArray* cSlice = (*C)(i, {0});
            try {
                matmulMPS(aSlice, bSlice, cSlice, transA, transB, alpha, beta);
            } catch (...) {
                delete aSlice;
                delete bSlice;
                delete cSlice;
                throw;
            }
            delete aSlice;
            delete bSlice;
            delete cSlice;
        }
    }

    return sd::Status::OK;
}

PLATFORM_CHECK(batched_gemm, ENGINE_CPU) {
    auto A = INPUT_VARIABLE(0);
    auto B = INPUT_VARIABLE(1);
    auto C = OUTPUT_VARIABLE(0);

    Requirements req("MPS BATCHED_GEMM OP");

    req.expectTrue(block.isUseMPS(), IS_USE_MPS_MSG);
    req.expectTrue(mpsUtils::MPSDeviceManager::getInstance().isAvailable(),
                   "MPS device must be available");
    req.expectTrue(A->dataType() == DataType::FLOAT32,
                   "Only float32 is currently supported by MPS batched_gemm");
    req.expectTrue(A->dataType() == B->dataType(),
                   "Input arrays must have the same data type");
    req.expectTrue(mpsUtils::isContiguous(*A), "Input A must be contiguous");
    req.expectTrue(mpsUtils::isContiguous(*B), "Input B must be contiguous");
    req.expectFalse(A->isEmpty(), "Input A must not be empty");
    req.expectFalse(B->isEmpty(), "Input B must not be empty");
    req.expectTrue(A->rankOf() >= 2 && A->rankOf() <= 3,
                   "batched_gemm expects rank 2 or 3 inputs");

    req.logTheSuccess();
    return req;
}

#endif  // HAVE_MPS

}  // namespace platforms
}  // namespace ops
}  // namespace sd
