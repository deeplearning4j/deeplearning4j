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
// MIOpen Utilities for ZLUDA AMD GPU Support
// This provides cuDNN-equivalent functionality for AMD GPUs via MIOpen
//

#ifndef SD_MIOPEN_UTILS_H
#define SD_MIOPEN_UTILS_H

#include <system/common.h>

#if defined(HAVE_MIOPEN)

#include <miopen/miopen.h>
#include <hip/hip_runtime.h>

#include <array/DataType.h>
#include <ops/declarable/OpRegistrator.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>
#include <stdexcept>
#include <string>

namespace sd {
namespace ops {
namespace platforms {

// ============================================================================
// Platform Helper Declarations for ENGINE_ZLUDA_AMD
// These mirror the cuDNN platform helpers but use MIOpen
// ============================================================================

// Convolution operations
DECLARE_PLATFORM(conv2d, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(conv2d_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(conv3dnew, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(conv3dnew_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(depthwise_conv2d, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(depthwise_conv2d_bp, ENGINE_ZLUDA_AMD);

// Pooling operations
DECLARE_PLATFORM(avgpool2d, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(avgpool2d_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(avgpool3dnew, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(avgpool3dnew_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(maxpool2d, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(maxpool2d_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(maxpool3dnew, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(maxpool3dnew_bp, ENGINE_ZLUDA_AMD);

// Normalization operations
DECLARE_PLATFORM(batchnorm, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(batchnorm_bp, ENGINE_ZLUDA_AMD);

// Activation operations
DECLARE_PLATFORM(relu, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(relu6, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(sigmoid, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(tanh, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(elu, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(softplus, ENGINE_ZLUDA_AMD);

// Softmax operations
DECLARE_PLATFORM(softmax, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(softmax_bp, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(log_softmax, ENGINE_ZLUDA_AMD);
DECLARE_PLATFORM(log_softmax_bp, ENGINE_ZLUDA_AMD);

// ============================================================================
// Error Handling Utilities
// ============================================================================

/**
 * Convert MIOpen status to string for error reporting
 */
inline const char* miopenStatusToString(miopenStatus_t status) {
    switch (status) {
        case miopenStatusSuccess: return "miopenStatusSuccess";
        case miopenStatusNotInitialized: return "miopenStatusNotInitialized";
        case miopenStatusInvalidValue: return "miopenStatusInvalidValue";
        case miopenStatusBadParm: return "miopenStatusBadParm";
        case miopenStatusAllocFailed: return "miopenStatusAllocFailed";
        case miopenStatusInternalError: return "miopenStatusInternalError";
        case miopenStatusNotImplemented: return "miopenStatusNotImplemented";
        case miopenStatusUnknownError: return "miopenStatusUnknownError";
        default: return "Unknown MIOpen error";
    }
}

/**
 * Throw exception if MIOpen call fails
 */
inline void throwIfMIOpenFailed(miopenStatus_t status, const char* message = "",
                                 const char* func = __builtin_FUNCTION()) {
    if (status != miopenStatusSuccess) {
        std::string errorMsg = std::string("MIOpen error in ") + func + ": " +
                               miopenStatusToString(status);
        if (message && message[0] != '\0') {
            errorMsg += " - " + std::string(message);
        }
        throw std::runtime_error(errorMsg);
    }
}

#define CHECK_MIOPEN(status) throwIfMIOpenFailed(status, #status)
#define CHECK_MIOPEN_MSG(msg, status) throwIfMIOpenFailed(status, msg)

/**
 * Check HIP errors
 */
inline void throwIfHipFailed(hipError_t error, const char* message = "",
                              const char* func = __builtin_FUNCTION()) {
    if (error != hipSuccess) {
        std::string errorMsg = std::string("HIP error in ") + func + ": " +
                               hipGetErrorString(error);
        if (message && message[0] != '\0') {
            errorMsg += " - " + std::string(message);
        }
        throw std::runtime_error(errorMsg);
    }
}

#define CHECK_HIP(error) throwIfHipFailed(error, #error)

// ============================================================================
// Data Type Conversion Utilities
// ============================================================================

/**
 * Convert ND4J DataType to MIOpen data type
 */
inline miopenDataType_t miopenDataType(DataType dataType) {
    switch (dataType) {
        case FLOAT32: return miopenFloat;
        case FLOAT16: return miopenHalf;
        case BFLOAT16: return miopenBFloat16;
        case INT8: return miopenInt8;
        case INT32: return miopenInt32;
        default:
            throw std::runtime_error("Unsupported data type for MIOpen: " +
                                      std::to_string(static_cast<int>(dataType)));
    }
}

/**
 * Check if data type is supported by MIOpen
 */
inline bool isMIOpenSupportedType(DataType dataType) {
    switch (dataType) {
        case FLOAT32:
        case FLOAT16:
        case BFLOAT16:
        case INT8:
        case INT32:
            return true;
        default:
            return false;
    }
}

// ============================================================================
// RAII Descriptor Wrappers
// ============================================================================

/**
 * RAII wrapper for MIOpen tensor descriptor
 */
class MIOpenTensor {
public:
    miopenTensorDescriptor_t desc;

    MIOpenTensor() : desc(nullptr) {
        CHECK_MIOPEN(miopenCreateTensorDescriptor(&desc));
    }

    ~MIOpenTensor() {
        if (desc != nullptr) {
            miopenDestroyTensorDescriptor(desc);
        }
    }

    // Non-copyable
    MIOpenTensor(const MIOpenTensor&) = delete;
    MIOpenTensor& operator=(const MIOpenTensor&) = delete;

    // Moveable
    MIOpenTensor(MIOpenTensor&& other) noexcept : desc(other.desc) {
        other.desc = nullptr;
    }

    MIOpenTensor& operator=(MIOpenTensor&& other) noexcept {
        if (this != &other) {
            if (desc != nullptr) {
                miopenDestroyTensorDescriptor(desc);
            }
            desc = other.desc;
            other.desc = nullptr;
        }
        return *this;
    }

    void set4D(miopenDataType_t dataType, int n, int c, int h, int w) {
        CHECK_MIOPEN(miopenSet4dTensorDescriptor(desc, dataType, n, c, h, w));
    }

    void setND(miopenDataType_t dataType, int nbDims, const int* dims, const int* strides) {
        CHECK_MIOPEN(miopenSetTensorDescriptor(desc, dataType, nbDims,
                     const_cast<int*>(dims), const_cast<int*>(strides)));
    }

    operator miopenTensorDescriptor_t() const { return desc; }
};

/**
 * RAII wrapper for MIOpen convolution descriptor
 */
class MIOpenConvolution {
public:
    miopenConvolutionDescriptor_t desc;

    MIOpenConvolution() : desc(nullptr) {
        CHECK_MIOPEN(miopenCreateConvolutionDescriptor(&desc));
    }

    ~MIOpenConvolution() {
        if (desc != nullptr) {
            miopenDestroyConvolutionDescriptor(desc);
        }
    }

    // Non-copyable
    MIOpenConvolution(const MIOpenConvolution&) = delete;
    MIOpenConvolution& operator=(const MIOpenConvolution&) = delete;

    // Moveable
    MIOpenConvolution(MIOpenConvolution&& other) noexcept : desc(other.desc) {
        other.desc = nullptr;
    }

    void set2D(int padH, int padW, int strideH, int strideW, int dilH, int dilW,
               miopenConvolutionMode_t mode = miopenConvolution) {
        CHECK_MIOPEN(miopenInitConvolutionDescriptor(desc, mode,
                     padH, padW, strideH, strideW, dilH, dilW));
    }

    void setGroupCount(int groupCount) {
        CHECK_MIOPEN(miopenSetConvolutionGroupCount(desc, groupCount));
    }

    operator miopenConvolutionDescriptor_t() const { return desc; }
};

/**
 * RAII wrapper for MIOpen pooling descriptor
 */
class MIOpenPooling {
public:
    miopenPoolingDescriptor_t desc;

    MIOpenPooling() : desc(nullptr) {
        CHECK_MIOPEN(miopenCreatePoolingDescriptor(&desc));
    }

    ~MIOpenPooling() {
        if (desc != nullptr) {
            miopenDestroyPoolingDescriptor(desc);
        }
    }

    // Non-copyable
    MIOpenPooling(const MIOpenPooling&) = delete;
    MIOpenPooling& operator=(const MIOpenPooling&) = delete;

    void set2D(miopenPoolingMode_t mode, int windowH, int windowW,
               int padH, int padW, int strideH, int strideW) {
        CHECK_MIOPEN(miopenSet2dPoolingDescriptor(desc, mode,
                     windowH, windowW, padH, padW, strideH, strideW));
    }

    operator miopenPoolingDescriptor_t() const { return desc; }
};

/**
 * RAII wrapper for MIOpen activation descriptor
 */
class MIOpenActivation {
public:
    miopenActivationDescriptor_t desc;

    MIOpenActivation() : desc(nullptr) {
        CHECK_MIOPEN(miopenCreateActivationDescriptor(&desc));
    }

    ~MIOpenActivation() {
        if (desc != nullptr) {
            miopenDestroyActivationDescriptor(desc);
        }
    }

    // Non-copyable
    MIOpenActivation(const MIOpenActivation&) = delete;
    MIOpenActivation& operator=(const MIOpenActivation&) = delete;

    void set(miopenActivationMode_t mode, double alpha = 0.0, double beta = 0.0, double gamma = 0.0) {
        CHECK_MIOPEN(miopenSetActivationDescriptor(desc, mode, alpha, beta, gamma));
    }

    operator miopenActivationDescriptor_t() const { return desc; }
};

/**
 * RAII wrapper for MIOpen LRN descriptor
 */
class MIOpenLRN {
public:
    miopenLRNDescriptor_t desc;

    MIOpenLRN() : desc(nullptr) {
        CHECK_MIOPEN(miopenCreateLRNDescriptor(&desc));
    }

    ~MIOpenLRN() {
        if (desc != nullptr) {
            miopenDestroyLRNDescriptor(desc);
        }
    }

    // Non-copyable
    MIOpenLRN(const MIOpenLRN&) = delete;
    MIOpenLRN& operator=(const MIOpenLRN&) = delete;

    void set(miopenLRNMode_t mode, unsigned int n, double alpha, double beta, double k) {
        CHECK_MIOPEN(miopenSetLRNDescriptor(desc, mode, n, alpha, beta, k));
    }

    operator miopenLRNDescriptor_t() const { return desc; }
};

// ============================================================================
// Workspace Management
// ============================================================================

/**
 * RAII wrapper for HIP device memory allocation
 */
class HipWorkspace {
public:
    void* ptr;
    size_t size;

    HipWorkspace() : ptr(nullptr), size(0) {}

    explicit HipWorkspace(size_t bytes) : ptr(nullptr), size(bytes) {
        if (bytes > 0) {
            CHECK_HIP(hipMalloc(&ptr, bytes));
        }
    }

    ~HipWorkspace() {
        if (ptr != nullptr) {
            hipFree(ptr);
        }
    }

    // Non-copyable
    HipWorkspace(const HipWorkspace&) = delete;
    HipWorkspace& operator=(const HipWorkspace&) = delete;

    // Moveable
    HipWorkspace(HipWorkspace&& other) noexcept : ptr(other.ptr), size(other.size) {
        other.ptr = nullptr;
        other.size = 0;
    }

    HipWorkspace& operator=(HipWorkspace&& other) noexcept {
        if (this != &other) {
            if (ptr != nullptr) {
                hipFree(ptr);
            }
            ptr = other.ptr;
            size = other.size;
            other.ptr = nullptr;
            other.size = 0;
        }
        return *this;
    }

    void resize(size_t bytes) {
        if (bytes > size) {
            if (ptr != nullptr) {
                hipFree(ptr);
            }
            CHECK_HIP(hipMalloc(&ptr, bytes));
            size = bytes;
        }
    }

    operator void*() const { return ptr; }
};

// ============================================================================
// Handle Management
// ============================================================================

/**
 * Get MIOpen handle from LaunchContext
 * Note: This requires LaunchContext to be extended to store MIOpen handle
 */
inline miopenHandle_t getMIOpenHandle(const LaunchContext* context) {
    // For now, return the handle stored in the context
    // The LaunchContext needs to be extended to support this
    void* handle = context->getCuDnnHandle();  // Reuse cuDNN handle storage
    return reinterpret_cast<miopenHandle_t>(handle);
}

}  // namespace platforms
}  // namespace ops
}  // namespace sd

#endif  // HAVE_MIOPEN

#endif  // SD_MIOPEN_UTILS_H
