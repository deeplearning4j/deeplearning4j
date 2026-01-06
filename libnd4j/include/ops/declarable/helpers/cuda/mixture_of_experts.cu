/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//
// @author Eclipse Deeplearning4j
//

#include <cuda_runtime.h>
#include <helpers/PointersManager.h>
#include <ops/declarable/helpers/mixture_of_experts.h>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_mixture_of_experts)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
__device__ static void deviceSoftmax(T* data, int size) {
    T maxVal = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > maxVal) maxVal = data[i];
    }
    T sum = static_cast<T>(0);
    for (int i = 0; i < size; i++) {
        data[i] = exp(data[i] - maxVal);
        sum += data[i];
    }
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

template <typename T>
__device__ static void deviceTopK(T* data, int size, int k, int* indices, T* values) {
    // Simple selection sort for top-K (efficient for small K)
    for (int i = 0; i < k; i++) {
        int maxIdx = i;
        T maxVal = data[i];
        for (int j = i + 1; j < size; j++) {
            if (data[j] > maxVal) {
                maxVal = data[j];
                maxIdx = j;
            }
        }
        // Swap
        T tmp = data[i];
        data[i] = data[maxIdx];
        data[maxIdx] = tmp;

        indices[i] = maxIdx;
        values[i] = data[i];
    }
}

template <typename T>
__global__ static void mixtureOfExpertsKernel(
    const T* input,
    const T* gatingWeights,
    const T* expertWeights,
    const T* expertBias,
    T* output,
    const sd::LongType batchSize,
    const sd::LongType seqLen,
    const sd::LongType hiddenDim,
    const sd::LongType expertDim,
    const int numExperts,
    const int topK,
    const sd::LongType inBatchStride,
    const sd::LongType inSeqStride,
    const sd::LongType inDimStride,
    const sd::LongType gatingDimStride,
    const sd::LongType gatingExpStride,
    const sd::LongType expExpStride,
    const sd::LongType expInStride,
    const sd::LongType expOutStride,
    const sd::LongType outBatchStride,
    const sd::LongType outSeqStride,
    const sd::LongType outDimStride) {

    // Each block handles one token
    const sd::LongType tokenIdx = blockIdx.x;
    const sd::LongType totalTokens = batchSize * seqLen;

    if (tokenIdx >= totalTokens) return;

    const sd::LongType b = tokenIdx / seqLen;
    const sd::LongType s = tokenIdx % seqLen;

    const T* tokenInput = input + b * inBatchStride + s * inSeqStride;
    T* tokenOutput = output + b * outBatchStride + s * outSeqStride;

    // Shared memory for router logits and top-K selection
    extern __shared__ char sharedMem[];
    T* routerLogits = reinterpret_cast<T*>(sharedMem);
    int* selectedExperts = reinterpret_cast<int*>(routerLogits + numExperts);
    T* expertScores = reinterpret_cast<T*>(selectedExperts + topK);

    // Initialize output to zero
    for (sd::LongType d = threadIdx.x; d < hiddenDim; d += blockDim.x) {
        tokenOutput[d * outDimStride] = static_cast<T>(0);
    }
    __syncthreads();

    // Compute router logits - each thread handles part of the experts
    for (int e = threadIdx.x; e < numExperts; e += blockDim.x) {
        T logit = static_cast<T>(0);
        for (sd::LongType d = 0; d < hiddenDim; d++) {
            logit += tokenInput[d * inDimStride] * gatingWeights[d * gatingDimStride + e * gatingExpStride];
        }
        routerLogits[e] = logit;
    }
    __syncthreads();

    // Thread 0 performs softmax and top-K selection
    if (threadIdx.x == 0) {
        deviceSoftmax(routerLogits, numExperts);
        deviceTopK(routerLogits, numExperts, topK, selectedExperts, expertScores);

        // Normalize scores
        T scoreSum = static_cast<T>(0);
        for (int k = 0; k < topK; k++) {
            scoreSum += expertScores[k];
        }
        for (int k = 0; k < topK; k++) {
            expertScores[k] /= scoreSum;
        }
    }
    __syncthreads();

    // Each thread handles part of the output dimensions
    for (sd::LongType od = threadIdx.x; od < expertDim && od < hiddenDim; od += blockDim.x) {
        T accumulator = static_cast<T>(0);

        for (int k = 0; k < topK; k++) {
            int expertIdx = selectedExperts[k];
            T score = expertScores[k];

            const T* expertW = expertWeights + expertIdx * expExpStride;

            T expertVal = static_cast<T>(0);
            for (sd::LongType id = 0; id < hiddenDim; id++) {
                expertVal += tokenInput[id * inDimStride] * expertW[id * expInStride + od * expOutStride];
            }

            if (expertBias != nullptr) {
                expertVal += expertBias[expertIdx * expertDim + od];
            }

            accumulator += score * expertVal;
        }

        tokenOutput[od * outDimStride] = accumulator;
    }
}

template <typename T>
static void mixtureOfExpertsSimpleCuda_(sd::LaunchContext* context,
                                         NDArray* input,
                                         NDArray* gatingWeights,
                                         NDArray* expertWeights,
                                         NDArray* expertBias,
                                         NDArray* output,
                                         int numExperts,
                                         int topK) {

    const auto batchSize = input->sizeAt(0);
    const auto seqLen = input->sizeAt(1);
    const auto hiddenDim = input->sizeAt(2);
    const auto expertDim = expertWeights->sizeAt(2);

    // Strides
    const auto inBatchStride = input->strideAt(0);
    const auto inSeqStride = input->strideAt(1);
    const auto inDimStride = input->strideAt(2);

    const auto gatingDimStride = gatingWeights->strideAt(0);
    const auto gatingExpStride = gatingWeights->strideAt(1);

    const auto expExpStride = expertWeights->strideAt(0);
    const auto expInStride = expertWeights->strideAt(1);
    const auto expOutStride = expertWeights->strideAt(2);

    const auto outBatchStride = output->strideAt(0);
    const auto outSeqStride = output->strideAt(1);
    const auto outDimStride = output->strideAt(2);

    const sd::LongType totalTokens = batchSize * seqLen;
    const int blockSize = 128;

    // Shared memory: routerLogits + selectedExperts + expertScores
    size_t sharedMemSize = numExperts * sizeof(T) + topK * sizeof(int) + topK * sizeof(T);

    PointersManager manager(context, "mixtureOfExperts");

    const T* inputBuffer = input->specialBufferasT<T>();
    const T* gatingBuffer = gatingWeights->specialBufferasT<T>();
    const T* expertBuffer = expertWeights->specialBufferasT<T>();
    const T* biasBuffer = expertBias != nullptr ? expertBias->specialBufferasT<T>() : nullptr;
    T* outputBuffer = output->specialBufferasT<T>();

    mixtureOfExpertsKernel<T><<<totalTokens, blockSize, sharedMemSize, *context->getCudaStream()>>>(
        inputBuffer, gatingBuffer, expertBuffer, biasBuffer, outputBuffer,
        batchSize, seqLen, hiddenDim, expertDim,
        numExperts, topK,
        inBatchStride, inSeqStride, inDimStride,
        gatingDimStride, gatingExpStride,
        expExpStride, expInStride, expOutStride,
        outBatchStride, outSeqStride, outDimStride);

    manager.synchronize();
}

void mixtureOfExpertsSimple(sd::LaunchContext* context,
                             NDArray* input,
                             NDArray* gatingWeights,
                             NDArray* expertWeights,
                             NDArray* expertBias,
                             NDArray* output,
                             int numExperts,
                             int topK) {

    BUILD_SINGLE_SELECTOR(input->dataType(), mixtureOfExpertsSimpleCuda_,
                          (context, input, gatingWeights, expertWeights, expertBias, output, numExperts, topK),
                          SD_FLOAT_TYPES);
}

void mixtureOfExperts(sd::LaunchContext* context,
                       NDArray* input,
                       NDArray* gatingWeights,
                       const std::vector<NDArray*>& expertWeights,
                       const std::vector<NDArray*>& expertBias,
                       NDArray* output,
                       NDArray* routerLogits,
                       NDArray* expertIndices,
                       int numExperts,
                       int topK,
                       int expertCapacity,
                       bool useSoftmaxGating,
                       bool useAuxLoss) {

    if (expertWeights.size() == 1) {
        mixtureOfExpertsSimple(context, input, gatingWeights, expertWeights[0],
                                expertBias.empty() ? nullptr : expertBias[0],
                                output, numExperts, topK);
    } else {
        THROW_EXCEPTION("mixtureOfExperts: Multiple expert weight tensors not yet supported");
    }
}

BUILD_SINGLE_TEMPLATE(void mixtureOfExpertsSimpleCuda_,
                      (sd::LaunchContext* context, NDArray* input, NDArray* gatingWeights,
                       NDArray* expertWeights, NDArray* expertBias, NDArray* output,
                       int numExperts, int topK),
                      SD_FLOAT_TYPES);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
