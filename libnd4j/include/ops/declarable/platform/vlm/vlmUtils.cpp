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

#include "vlmUtils.h"

#if HAVE_VLM

#include <execution/Threads.h>
#include <helpers/LoopsCoordsHelper.h>
#include <system/env_functions.h>
#include <cmath>

#include <system/BackendNamespace.h>

namespace sd {
SD_BACKEND_ROOT_INLINE_NAMESPACE_BEGIN
namespace vlmUtils {

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

bool hasClipSupport() {
    // CLIP support is available when VLM helper is compiled with clip.h
#ifdef HAVE_CLIP
    return true;
#else
    return false;
#endif
}

static thread_local struct ggml_context* tls_vlm_ctx = nullptr;
static thread_local std::vector<uint8_t> tls_vlm_buffer;

struct ggml_context* getVlmContext() {
    if (tls_vlm_ctx == nullptr) {
        // Default context size: 512MB for VLM operations (larger than LLM due to image tensors)
        const size_t ctx_size = 512 * 1024 * 1024;
        tls_vlm_buffer.resize(ctx_size);

        struct ggml_init_params params = {
            .mem_size   = ctx_size,
            .mem_buffer = tls_vlm_buffer.data(),
            .no_alloc   = false,
        };

        tls_vlm_ctx = ggml_init(params);
    }
    return tls_vlm_ctx;
}

void releaseVlmContext(struct ggml_context* ctx) {
    if (ctx != nullptr && ctx == tls_vlm_ctx) {
        ggml_free(ctx);
        tls_vlm_ctx = nullptr;
        tls_vlm_buffer.clear();
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
    int backends = VLM_BACKEND_CPU;  // CPU is always available

#ifdef GGML_USE_CUDA
    backends |= VLM_BACKEND_CUDA;
#endif

#ifdef GGML_USE_METAL
    backends |= VLM_BACKEND_METAL;
#endif

#ifdef GGML_USE_VULKAN
    backends |= VLM_BACKEND_VULKAN;
#endif

#ifdef GGML_USE_SYCL
    backends |= VLM_BACKEND_SYCL;
#endif

#ifdef GGML_USE_OPENCL
    backends |= VLM_BACKEND_OPENCL;
#endif

    return backends;
}

VlmBackend getPreferredBackend() {
    int available = getAvailableBackends();

    // Priority: CUDA > Metal > Vulkan > SYCL > OpenCL > CPU
    if (available & VLM_BACKEND_CUDA) return VLM_BACKEND_CUDA;
    if (available & VLM_BACKEND_METAL) return VLM_BACKEND_METAL;
    if (available & VLM_BACKEND_VULKAN) return VLM_BACKEND_VULKAN;
    if (available & VLM_BACKEND_SYCL) return VLM_BACKEND_SYCL;
    if (available & VLM_BACKEND_OPENCL) return VLM_BACKEND_OPENCL;

    return VLM_BACKEND_CPU;
}

ImagePreprocessParams getDefaultPreprocessParams() {
    return ImagePreprocessParams{
        .targetWidth = 224,
        .targetHeight = 224,
        .meanR = 0.485f, .meanG = 0.456f, .meanB = 0.406f,
        .stdR = 0.229f, .stdG = 0.224f, .stdB = 0.225f,
        .centerCrop = true,
        .useAspectRatio = false
    };
}

ImagePreprocessParams getCLIPPreprocessParams() {
    return ImagePreprocessParams{
        .targetWidth = 224,
        .targetHeight = 224,
        .meanR = 0.48145466f, .meanG = 0.4578275f, .meanB = 0.40821073f,
        .stdR = 0.26862954f, .stdG = 0.26130258f, .stdB = 0.27577711f,
        .centerCrop = true,
        .useAspectRatio = false
    };
}

ImagePreprocessParams getSigLIPPreprocessParams() {
    return ImagePreprocessParams{
        .targetWidth = 384,
        .targetHeight = 384,
        .meanR = 0.5f, .meanG = 0.5f, .meanB = 0.5f,
        .stdR = 0.5f, .stdG = 0.5f, .stdB = 0.5f,
        .centerCrop = true,
        .useAspectRatio = true
    };
}

void extractPatches(const NDArray* input, NDArray* output, const PatchParams& params) {
    const auto batch = input->sizeAt(0);
    const auto channels = input->sizeAt(1);
    const auto height = input->sizeAt(2);
    const auto width = input->sizeAt(3);

    const auto patchSize = params.patchSize;
    const auto stride = params.stride > 0 ? params.stride : patchSize;

    const auto numPatchesH = (height - patchSize) / stride + 1;
    const auto numPatchesW = (width - patchSize) / stride + 1;
    const auto numPatches = numPatchesH * numPatchesW;
    const auto patchDim = channels * patchSize * patchSize;

    // Extract patches using nested loops
    auto inputBuf = input->bufferAsT<float>();
    auto outputBuf = output->bufferAsT<float>();

    for (LongType b = 0; b < batch; ++b) {
        LongType patchIdx = params.includeClsToken ? 1 : 0;  // Reserve first slot for CLS token

        for (LongType ph = 0; ph < numPatchesH; ++ph) {
            for (LongType pw = 0; pw < numPatchesW; ++pw) {
                LongType outOffset = (b * (numPatches + (params.includeClsToken ? 1 : 0)) + patchIdx) * patchDim;

                for (LongType c = 0; c < channels; ++c) {
                    for (LongType i = 0; i < patchSize; ++i) {
                        for (LongType j = 0; j < patchSize; ++j) {
                            LongType inH = ph * stride + i;
                            LongType inW = pw * stride + j;
                            LongType inOffset = ((b * channels + c) * height + inH) * width + inW;

                            LongType patchOffset = (c * patchSize + i) * patchSize + j;
                            outputBuf[outOffset + patchOffset] = inputBuf[inOffset];
                        }
                    }
                }
                ++patchIdx;
            }
        }
    }
}

void apply2DPositionEncoding(const NDArray* patches, NDArray* output,
                              int height, int width, float temperature) {
    const auto batch = patches->sizeAt(0);
    const auto numPatches = patches->sizeAt(1);
    const auto dim = patches->sizeAt(2);

    // Copy input to output first
    output->assign(patches);

    auto outputBuf = output->bufferAsT<float>();

    // Generate 2D sinusoidal position encoding
    const int halfDim = dim / 2;
    const int quarterDim = halfDim / 2;

    for (LongType b = 0; b < batch; ++b) {
        for (LongType p = 0; p < numPatches; ++p) {
            int py = p / width;  // Row position
            int px = p % width;  // Column position

            LongType outOffset = (b * numPatches + p) * dim;

            // Height position encoding (first quarter)
            for (int i = 0; i < quarterDim; ++i) {
                float freq = 1.0f / std::pow(temperature, (2.0f * i) / quarterDim);
                outputBuf[outOffset + i] += std::sin(py * freq);
                outputBuf[outOffset + quarterDim + i] += std::cos(py * freq);
            }

            // Width position encoding (second quarter)
            for (int i = 0; i < quarterDim; ++i) {
                float freq = 1.0f / std::pow(temperature, (2.0f * i) / quarterDim);
                outputBuf[outOffset + halfDim + i] += std::sin(px * freq);
                outputBuf[outOffset + halfDim + quarterDim + i] += std::cos(px * freq);
            }
        }
    }
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

// ClipModelContext implementation
ClipModelContext::ClipModelContext() : _clipCtx(nullptr) {
}

ClipModelContext::~ClipModelContext() {
#ifdef HAVE_CLIP
    if (_clipCtx != nullptr) {
        clip_free(_clipCtx);
        _clipCtx = nullptr;
    }
#endif
}

bool ClipModelContext::loadModel(const char* modelPath) {
#ifdef HAVE_CLIP
    if (_clipCtx != nullptr) {
        clip_free(_clipCtx);
    }
    _clipCtx = clip_model_load(modelPath, /* verbosity */ 1);
    return _clipCtx != nullptr;
#else
    return false;
#endif
}

bool ClipModelContext::encodeImage(const NDArray* image, NDArray* output) {
#ifdef HAVE_CLIP
    if (_clipCtx == nullptr) return false;

    // Implementation would use clip_image_encode or similar
    // This is a placeholder for the actual CLIP encoding
    return true;
#else
    return false;
#endif
}

int ClipModelContext::getEmbeddingDim() const {
#ifdef HAVE_CLIP
    if (_clipCtx != nullptr) {
        return clip_n_mmproj_embd(_clipCtx);
    }
#endif
    return 0;
}

int ClipModelContext::getImageSize() const {
#ifdef HAVE_CLIP
    if (_clipCtx != nullptr) {
        return clip_image_size(_clipCtx);
    }
#endif
    return 224;  // Default CLIP image size
}

}  // namespace vlmUtils
SD_BACKEND_ROOT_INLINE_NAMESPACE_END
}  // namespace sd

#endif  // HAVE_VLM
