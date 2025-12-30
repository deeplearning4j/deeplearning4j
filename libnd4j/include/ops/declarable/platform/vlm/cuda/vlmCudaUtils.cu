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
// CUDA utilities for VLM (Vision Language Model) GGML integration
//

#include "../vlmUtils.h"

#if HAVE_VLM && defined(GGML_USE_CUDA)

#include <ggml-cuda.h>
#include <ggml-backend.h>
#include <cuda_runtime.h>
#include <mutex>

namespace sd {
namespace vlmUtils {

// Static backend cache for VLM CUDA operations
static std::vector<ggml_backend_t> s_vlmCudaBackends;
static std::mutex s_vlmBackendMutex;

bool hasCudaBackend() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    return (err == cudaSuccess && deviceCount > 0);
}

int getCudaDeviceCount() {
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);
    if (err != cudaSuccess) {
        return 0;
    }
    return deviceCount;
}

void setCudaDevice(int deviceId) {
    cudaSetDevice(deviceId);
}

struct ggml_backend* getCudaBackend(int deviceId) {
    std::lock_guard<std::mutex> lock(s_vlmBackendMutex);

    // Ensure vector is large enough
    if (s_vlmCudaBackends.size() <= static_cast<size_t>(deviceId)) {
        s_vlmCudaBackends.resize(deviceId + 1, nullptr);
    }

    // Create backend if not already created
    if (s_vlmCudaBackends[deviceId] == nullptr) {
        s_vlmCudaBackends[deviceId] = ggml_backend_cuda_init(deviceId);
    }

    return s_vlmCudaBackends[deviceId];
}

struct ggml_tensor* createGgmlTensorCuda(struct ggml_context* ctx, const NDArray* array,
                                          struct ggml_backend* backend, const char* name) {
    if (array == nullptr || ctx == nullptr) {
        return nullptr;
    }

    // Get shape info
    const auto shape = array->shapeOf();
    const auto rank = array->rankOf();
    const auto dt = array->dataType();

    // Convert to GGML type
    ggml_type ggmlType = toGgmlType(dt);

    // Create tensor based on rank (GGML uses column-major, so we reverse dimensions)
    struct ggml_tensor* tensor = nullptr;
    switch (rank) {
        case 1:
            tensor = ggml_new_tensor_1d(ctx, ggmlType, shape[0]);
            break;
        case 2:
            tensor = ggml_new_tensor_2d(ctx, ggmlType, shape[1], shape[0]);
            break;
        case 3:
            tensor = ggml_new_tensor_3d(ctx, ggmlType, shape[2], shape[1], shape[0]);
            break;
        case 4:
            tensor = ggml_new_tensor_4d(ctx, ggmlType, shape[3], shape[2], shape[1], shape[0]);
            break;
        default:
            // Fallback for higher ranks - flatten to 4D
            if (rank > 4) {
                int64_t dim0 = 1;
                for (int i = 0; i < rank - 3; i++) {
                    dim0 *= shape[i];
                }
                tensor = ggml_new_tensor_4d(ctx, ggmlType, shape[rank-1], shape[rank-2], shape[rank-3], dim0);
            }
            break;
    }

    if (tensor != nullptr) {
        if (name != nullptr) {
            ggml_set_name(tensor, name);
        }

        // Copy data to GPU via backend
        if (backend != nullptr) {
            ggml_backend_tensor_set(tensor, array->buffer(), 0, ggml_nbytes(tensor));
        }
    }

    return tensor;
}

void copyGgmlCudaToNDArray(const struct ggml_tensor* tensor, NDArray* array,
                            struct ggml_backend* backend) {
    if (tensor == nullptr || array == nullptr) {
        return;
    }

    // Get data from GPU via backend
    if (backend != nullptr) {
        ggml_backend_tensor_get(tensor, array->buffer(), 0, ggml_nbytes(tensor));
    }
}

void executeGgmlGraphCuda(struct ggml_context* ctx, struct ggml_cgraph* graph,
                          struct ggml_backend* backend) {
    if (ctx == nullptr || graph == nullptr || backend == nullptr) {
        return;
    }

    // Create a backend buffer for the computation
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);

    // Compute the graph on CUDA
    ggml_backend_graph_compute(backend, graph);

    // Free the buffer
    if (buffer != nullptr) {
        ggml_backend_buffer_free(buffer);
    }
}

// RAII wrapper implementation for CUDA
GgmlCudaContextGuard::GgmlCudaContextGuard(size_t memSize, int deviceId)
    : _ctx(nullptr), _backend(nullptr), _buffer(nullptr), _deviceId(deviceId) {

    // Get or create CUDA backend
    _backend = getCudaBackend(deviceId);

    if (_backend == nullptr) {
        throw std::runtime_error("Failed to initialize VLM CUDA backend");
    }

    // Create GGML context
    struct ggml_init_params params = {
        .mem_size   = memSize,
        .mem_buffer = nullptr,
        .no_alloc   = true  // We'll use backend allocation
    };

    _ctx = ggml_init(params);

    if (_ctx == nullptr) {
        throw std::runtime_error("Failed to initialize GGML context for VLM CUDA");
    }
}

GgmlCudaContextGuard::~GgmlCudaContextGuard() {
    if (_buffer != nullptr) {
        ggml_backend_buffer_free(_buffer);
    }
    if (_ctx != nullptr) {
        ggml_free(_ctx);
    }
    // Note: We don't free the backend as it's cached
}

}  // namespace vlmUtils
}  // namespace sd

#endif  // HAVE_VLM && GGML_USE_CUDA
