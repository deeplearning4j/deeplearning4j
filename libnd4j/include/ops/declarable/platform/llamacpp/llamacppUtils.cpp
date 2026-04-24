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

#include "llamacppUtils.h"

#if HAVE_LLAMACPP

#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>
#include <system/env_functions.h>

namespace sd {
namespace llamacppUtils {

ggml_type toGgmlType(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32:
            return GGML_TYPE_F32;
        case DataType::FLOAT16:
        case DataType::HALF:
            return GGML_TYPE_F16;
        case DataType::BFLOAT16:
            return GGML_TYPE_BF16;
        case DataType::INT32:
            return GGML_TYPE_I32;
        case DataType::INT16:
            return GGML_TYPE_I16;
        case DataType::INT8:
            return GGML_TYPE_I8;
        default:
            return GGML_TYPE_F32;  // Default fallback
    }
}

DataType fromGgmlType(ggml_type gt) {
    switch (gt) {
        case GGML_TYPE_F32:
            return DataType::FLOAT32;
        case GGML_TYPE_F16:
            return DataType::FLOAT16;
        case GGML_TYPE_BF16:
            return DataType::BFLOAT16;
        case GGML_TYPE_I32:
            return DataType::INT32;
        case GGML_TYPE_I16:
            return DataType::INT16;
        case GGML_TYPE_I8:
            return DataType::INT8;
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
        case GGML_TYPE_Q8_1:
            return DataType::INT8;  // Quantized types map to INT8 for compatibility
        default:
            return DataType::FLOAT32;
    }
}

bool isSupportedType(DataType dt) {
    switch (dt) {
        case DataType::FLOAT32:
        case DataType::FLOAT16:
        case DataType::HALF:
        case DataType::BFLOAT16:
        case DataType::INT32:
        case DataType::INT16:
        case DataType::INT8:
            return true;
        default:
            return false;
    }
}

bool hasQuantizedSupport() {
    // GGML always has quantized support built-in
    return true;
}

static thread_local struct ggml_context* tls_ggml_ctx = nullptr;
static thread_local std::vector<uint8_t> tls_ggml_buffer;

struct ggml_context* getGgmlContext() {
    if (tls_ggml_ctx == nullptr) {
        // Default context size: 256MB
        const size_t ctx_size = 256 * 1024 * 1024;
        tls_ggml_buffer.resize(ctx_size);

        struct ggml_init_params params = {
            .mem_size   = ctx_size,
            .mem_buffer = tls_ggml_buffer.data(),
            .no_alloc   = false,
        };

        tls_ggml_ctx = ggml_init(params);
    }
    return tls_ggml_ctx;
}

void releaseGgmlContext(struct ggml_context* ctx) {
    if (ctx != nullptr && ctx == tls_ggml_ctx) {
        ggml_free(ctx);
        tls_ggml_ctx = nullptr;
        tls_ggml_buffer.clear();
    }
}

struct ggml_tensor* createGgmlTensor(struct ggml_context* ctx, const NDArray* array, const char* name) {
    const auto rank = array->rankOf();
    const auto shape = array->shapeOf();
    const auto type = toGgmlType(array->dataType());

    struct ggml_tensor* tensor = nullptr;

    // GGML supports up to 4 dimensions
    switch (rank) {
        case 1:
            tensor = ggml_new_tensor_1d(ctx, type, shape[0]);
            break;
        case 2:
            tensor = ggml_new_tensor_2d(ctx, type, shape[1], shape[0]);
            break;
        case 3:
            tensor = ggml_new_tensor_3d(ctx, type, shape[2], shape[1], shape[0]);
            break;
        case 4:
        default:
            tensor = ggml_new_tensor_4d(ctx, type,
                rank >= 4 ? shape[3] : 1,
                rank >= 3 ? shape[2] : 1,
                rank >= 2 ? shape[1] : 1,
                shape[0]);
            break;
    }

    if (tensor != nullptr) {
        // Copy data from NDArray to GGML tensor
        memcpy(tensor->data, array->buffer(), array->lengthOf() * array->sizeOfT());

        if (name != nullptr) {
            ggml_set_name(tensor, name);
        }
    }

    return tensor;
}

void copyGgmlToNDArray(const struct ggml_tensor* tensor, NDArray* array) {
    if (tensor == nullptr || array == nullptr) return;

    const size_t bytes = ggml_nbytes(tensor);
    const size_t arrayBytes = array->lengthOf() * array->sizeOfT();

    if (bytes <= arrayBytes) {
        memcpy(array->buffer(), tensor->data, bytes);
    }
}

void executeGgmlGraph(struct ggml_context* ctx, struct ggml_cgraph* graph, int numThreads) {
    if (numThreads <= 0) {
        // Use default thread count
        numThreads = sd::env_maxMasterThreads();
    }

    // Create a new compute plan
    struct ggml_cplan plan = ggml_graph_plan(graph, numThreads);

    // Allocate work buffer if needed
    std::vector<uint8_t> work_buffer;
    if (plan.work_size > 0) {
        work_buffer.resize(plan.work_size);
        plan.work_data = work_buffer.data();
    }

    // Execute the graph
    ggml_graph_compute(graph, &plan);
}

int getAvailableBackends() {
    int backends = GGML_BACKEND_CPU;  // CPU is always available

#ifdef GGML_USE_CUDA
    backends |= GGML_BACKEND_CUDA;
#endif

#ifdef GGML_USE_METAL
    backends |= GGML_BACKEND_METAL;
#endif

#ifdef GGML_USE_VULKAN
    backends |= GGML_BACKEND_VULKAN;
#endif

#ifdef GGML_USE_SYCL
    backends |= GGML_BACKEND_SYCL;
#endif

#ifdef GGML_USE_OPENCL
    backends |= GGML_BACKEND_OPENCL;
#endif

    return backends;
}

GgmlBackend getPreferredBackend() {
    int available = getAvailableBackends();

    // Priority: CUDA > Metal > Vulkan > SYCL > OpenCL > CPU
    if (available & GGML_BACKEND_CUDA) return GGML_BACKEND_CUDA;
    if (available & GGML_BACKEND_METAL) return GGML_BACKEND_METAL;
    if (available & GGML_BACKEND_VULKAN) return GGML_BACKEND_VULKAN;
    if (available & GGML_BACKEND_SYCL) return GGML_BACKEND_SYCL;
    if (available & GGML_BACKEND_OPENCL) return GGML_BACKEND_OPENCL;

    return GGML_BACKEND_CPU;
}

// GgmlContextGuard implementation
GgmlContextGuard::GgmlContextGuard(size_t memSize) {
    _buffer.resize(memSize);

    struct ggml_init_params params = {
        .mem_size   = memSize,
        .mem_buffer = _buffer.data(),
        .no_alloc   = false,
    };

    _ctx = ggml_init(params);
}

GgmlContextGuard::~GgmlContextGuard() {
    if (_ctx != nullptr) {
        ggml_free(_ctx);
        _ctx = nullptr;
    }
}

}  // namespace llamacppUtils
}  // namespace sd

#endif  // HAVE_LLAMACPP
