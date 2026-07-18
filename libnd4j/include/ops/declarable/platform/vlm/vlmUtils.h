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

#ifndef DEV_TESTS_VLMUTILS_H
#define DEV_TESTS_VLMUTILS_H

#include <array/NDArray.h>
#include <graph/Context.h>
#include <legacy/NativeOps.h>
#include <ops/declarable/PlatformHelper.h>
#include <system/platform_boilerplate.h>

#if HAVE_VLM
#include <llama.h>
#include <ggml.h>
#include <clip.h>
#endif

using namespace samediff;

#include <system/BackendNamespace.h>

namespace sd {
namespace ops {
namespace platforms {
// No file-level SD_NS wrap: DECLARE_PLATFORM opens it per declaration
// (platform_boilerplate.h); wrapping the list again nests SD_NS in itself.

/**
 * Platform helper declarations for VLM (Vision Language Model) accelerated operations.
 * These operations support multimodal AI models that combine vision and language.
 */

// ============================================================================
// CPU Platform Declarations
// ============================================================================

// Vision encoder - extracts features from images using vision backbone
DECLARE_PLATFORM(vlm_vision_encode, ENGINE_CPU);

// Image embedding - converts image patches to embeddings
DECLARE_PLATFORM(vlm_image_embed, ENGINE_CPU);

// Patch embedding - extracts patches from images and projects them
DECLARE_PLATFORM(vlm_patch_embed, ENGINE_CPU);

// Cross attention - attention between vision and language features
DECLARE_PLATFORM(vlm_cross_attention, ENGINE_CPU);

// Multimodal fusion - combines vision and language representations
DECLARE_PLATFORM(vlm_multimodal_fusion, ENGINE_CPU);

// Vision projection - projects vision features to language space
DECLARE_PLATFORM(vlm_vision_projection, ENGINE_CPU);

// Image preprocessing - normalize, resize, and prepare images for VLM
DECLARE_PLATFORM(vlm_image_preprocess, ENGINE_CPU);

// Positional encoding for image patches (2D sinusoidal or learned)
DECLARE_PLATFORM(vlm_2d_position_encode, ENGINE_CPU);

// ============================================================================
// CUDA Platform Declarations
// ============================================================================

// Vision encoder on CUDA
DECLARE_PLATFORM(vlm_vision_encode, ENGINE_CUDA);

// Image embedding on CUDA
DECLARE_PLATFORM(vlm_image_embed, ENGINE_CUDA);

// Patch embedding on CUDA
DECLARE_PLATFORM(vlm_patch_embed, ENGINE_CUDA);

// Cross attention on CUDA
DECLARE_PLATFORM(vlm_cross_attention, ENGINE_CUDA);

// Multimodal fusion on CUDA
DECLARE_PLATFORM(vlm_multimodal_fusion, ENGINE_CUDA);

// Vision projection on CUDA
DECLARE_PLATFORM(vlm_vision_projection, ENGINE_CUDA);

// Image preprocessing on CUDA
DECLARE_PLATFORM(vlm_image_preprocess, ENGINE_CUDA);

// 2D position encoding on CUDA
DECLARE_PLATFORM(vlm_2d_position_encode, ENGINE_CUDA);

}  // namespace platforms
}  // namespace ops

SD_BACKEND_ROOT_INLINE_NAMESPACE_BEGIN
namespace vlmUtils {

#if HAVE_VLM

/**
 * Convert NDArray data type to GGML data type
 */
ggml_type toGgmlType(DataType dt);

/**
 * Convert GGML data type to NDArray data type
 */
DataType fromGgmlType(ggml_type gt);

/**
 * Check if a data type is supported by VLM operations
 */
bool isSupportedType(DataType dt);

/**
 * Check if CLIP model support is available
 */
bool hasClipSupport();

/**
 * Get the GGML context for VLM operations
 * Uses thread-local storage for thread safety
 */
struct ggml_context* getVlmContext();

/**
 * Release VLM context resources
 */
void releaseVlmContext(struct ggml_context* ctx);

/**
 * Create a GGML tensor from an NDArray
 * @param ctx GGML context
 * @param array Source NDArray
 * @param name Optional tensor name
 * @return GGML tensor wrapping the NDArray data
 */
struct ggml_tensor* createGgmlTensor(struct ggml_context* ctx, const NDArray* array, const char* name = nullptr);

/**
 * Copy GGML tensor data back to NDArray
 * @param tensor Source GGML tensor
 * @param array Destination NDArray
 */
void copyGgmlToNDArray(const struct ggml_tensor* tensor, NDArray* array);

/**
 * Execute a GGML computation graph
 * @param ctx GGML context
 * @param graph GGML computation graph
 * @param numThreads Number of threads to use
 */
void executeGgmlGraph(struct ggml_context* ctx, struct ggml_cgraph* graph, int numThreads = 0);

/**
 * Get available GGML backends for VLM
 * @return Bitmask of available backends
 */
int getAvailableBackends();

/**
 * Backend flags for VLM operations
 */
enum VlmBackend {
    VLM_BACKEND_CPU = 1 << 0,
    VLM_BACKEND_CUDA = 1 << 1,
    VLM_BACKEND_METAL = 1 << 2,
    VLM_BACKEND_VULKAN = 1 << 3,
    VLM_BACKEND_SYCL = 1 << 4,
    VLM_BACKEND_OPENCL = 1 << 5
};

/**
 * Get the preferred backend based on availability
 * @return Best available backend
 */
VlmBackend getPreferredBackend();

/**
 * Image preprocessing parameters
 */
struct ImagePreprocessParams {
    int targetWidth;
    int targetHeight;
    float meanR, meanG, meanB;      // Normalization means
    float stdR, stdG, stdB;          // Normalization stds
    bool centerCrop;
    bool useAspectRatio;
};

/**
 * Default preprocessing params for common VLM models
 */
ImagePreprocessParams getDefaultPreprocessParams();
ImagePreprocessParams getCLIPPreprocessParams();
ImagePreprocessParams getSigLIPPreprocessParams();

/**
 * Patch extraction parameters
 */
struct PatchParams {
    int patchSize;          // Size of each patch (e.g., 14, 16, 32)
    int stride;             // Stride between patches
    bool includeClsToken;   // Whether to prepend CLS token
};

/**
 * Extract image patches from input
 * @param input Input image tensor [B, C, H, W] or [B, H, W, C]
 * @param output Output patches [B, num_patches, patch_dim]
 * @param params Patch extraction parameters
 */
void extractPatches(const NDArray* input, NDArray* output, const PatchParams& params);

/**
 * Apply 2D sinusoidal position encoding
 * @param patches Input patches [B, num_patches, dim]
 * @param output Output with position encoding added
 * @param height Grid height
 * @param width Grid width
 * @param temperature Temperature for sinusoidal encoding
 */
void apply2DPositionEncoding(const NDArray* patches, NDArray* output,
                              int height, int width, float temperature = 10000.0f);

/**
 * GGML context wrapper for RAII (CPU)
 */
class GgmlContextGuard {
public:
    explicit GgmlContextGuard(size_t memSize = 512 * 1024 * 1024);
    ~GgmlContextGuard();

    struct ggml_context* get() { return _ctx; }
    operator struct ggml_context*() { return _ctx; }

private:
    struct ggml_context* _ctx;
    std::vector<uint8_t> _buffer;
};

// ============================================================================
// CUDA-specific utilities (only available when GGML_USE_CUDA is defined)
// ============================================================================

#if defined(GGML_USE_CUDA)

/**
 * Check if CUDA backend is available
 */
bool hasCudaBackend();

/**
 * Get number of available CUDA devices
 */
int getCudaDeviceCount();

/**
 * Set the current CUDA device
 */
void setCudaDevice(int deviceId);

/**
 * Get the CUDA backend for a specific device
 * @param deviceId CUDA device ID (default 0)
 * @return GGML backend for CUDA operations
 */
struct ggml_backend* getCudaBackend(int deviceId = 0);

/**
 * Create a GGML tensor on CUDA from an NDArray
 * @param ctx GGML context
 * @param array Source NDArray
 * @param backend CUDA backend
 * @param name Optional tensor name
 * @return GGML tensor with data on GPU
 */
struct ggml_tensor* createGgmlTensorCuda(struct ggml_context* ctx, const NDArray* array,
                                          struct ggml_backend* backend, const char* name = nullptr);

/**
 * Copy GGML tensor data from CUDA back to NDArray
 * @param tensor Source GGML tensor on GPU
 * @param array Destination NDArray
 * @param backend CUDA backend
 */
void copyGgmlCudaToNDArray(const struct ggml_tensor* tensor, NDArray* array,
                            struct ggml_backend* backend);

/**
 * Execute a GGML computation graph on CUDA
 * @param ctx GGML context
 * @param graph GGML computation graph
 * @param backend CUDA backend
 */
void executeGgmlGraphCuda(struct ggml_context* ctx, struct ggml_cgraph* graph,
                          struct ggml_backend* backend);

/**
 * GGML CUDA context wrapper for RAII
 */
class GgmlCudaContextGuard {
public:
    explicit GgmlCudaContextGuard(size_t memSize = 512 * 1024 * 1024, int deviceId = 0);
    ~GgmlCudaContextGuard();

    struct ggml_context* get() { return _ctx; }
    operator struct ggml_context*() { return _ctx; }

    struct ggml_backend* getBackend() { return _backend; }
    int getDeviceId() const { return _deviceId; }

private:
    struct ggml_context* _ctx;
    struct ggml_backend* _backend;
    struct ggml_backend_buffer* _buffer;
    int _deviceId;
};

#endif  // GGML_USE_CUDA

/**
 * CLIP model wrapper for vision encoding
 */
class ClipModelContext {
public:
    ClipModelContext();
    ~ClipModelContext();

    bool loadModel(const char* modelPath);
    bool isLoaded() const { return _clipCtx != nullptr; }

    /**
     * Encode image to vision features
     * @param image Input image tensor
     * @param output Output features
     * @return true if successful
     */
    bool encodeImage(const NDArray* image, NDArray* output);

    /**
     * Get the embedding dimension of the model
     */
    int getEmbeddingDim() const;

    /**
     * Get the expected image size
     */
    int getImageSize() const;

private:
    struct clip_ctx* _clipCtx;
};

#endif  // HAVE_VLM

}  // namespace vlmUtils
SD_BACKEND_ROOT_INLINE_NAMESPACE_END
}  // namespace sd

#endif  // DEV_TESTS_VLMUTILS_H
