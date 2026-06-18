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
#include <helpers/DebugHelper.h>
#include <execution/cuda/LaunchDims.h>
#include <ops/declarable/helpers/moe_shared_experts.h>
#include <system/selective_rendering.h>

#if NOT_EXCLUDED(OP_moe_shared_experts)

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
SD_DEVICE static void deviceSoftmaxMoe(T* data, int size) {
    T maxVal = data[0];
    for (int i = 1; i < size; i++) {
        if (data[i] > maxVal) maxVal = data[i];
    }
    T sum = static_cast<T>(0);
    for (int i = 0; i < size; i++) {
        data[i] = static_cast<T>(__expf(static_cast<float>(data[i] - maxVal)));
        sum += data[i];
    }
    for (int i = 0; i < size; i++) {
        data[i] /= sum;
    }
}

template <typename T>
SD_DEVICE static void deviceTopKMoe(T* data, int size, int k, int* indices, T* values) {
    for (int i = 0; i < k; i++) {
        int maxIdx = i;
        T maxVal = data[i];
        for (int j = i + 1; j < size; j++) {
            if (data[j] > maxVal) {
                maxVal = data[j];
                maxIdx = j;
            }
        }
        T tmp = data[i];
        data[i] = data[maxIdx];
        data[maxIdx] = tmp;
        indices[i] = maxIdx;
        values[i] = data[i];
    }
}

template <typename T>
SD_DEVICE SD_INLINE static T deviceSilu(T x) {
    return x / (static_cast<T>(1) + static_cast<T>(__expf(static_cast<float>(-x))));
}

/**
 * Kernel: one block per token.
 * Each block computes shared expert + routed experts for one token.
 *
 * Shared memory layout:
 *   - routerLogits[numRoutedExperts]
 *   - selectedExperts[topK] (int)
 *   - expertScores[topK]
 *   - sharedIntermedBuf[sharedIntermediate] (for gate/up intermediate values)
 */
template <typename T>
SD_KERNEL __launch_bounds__(256, 2) static void moeSharedExpertsKernel(
    const T* input,
    const T* routerWeights,
    const T* routedExpertWeights,
    const T* sharedGateProj,
    const T* sharedUpProj,
    const T* sharedDownProj,
    const T* routedExpertBias,
    T* output,
    T* routerProbsOut,
    sd::LongType* expertIndicesOut,
    const sd::LongType batchSize,
    const sd::LongType seqLen,
    const sd::LongType hiddenDim,
    const sd::LongType expertHidden,
    const sd::LongType sharedIntermediate,
    const int numRoutedExperts,
    const int topK,
    const int normalizeProbs,
    // Input strides
    const sd::LongType inBatchStride,
    const sd::LongType inSeqStride,
    const sd::LongType inDimStride,
    // Router strides
    const sd::LongType rtrDimStride,
    const sd::LongType rtrExpStride,
    // Expert strides
    const sd::LongType expExpStride,
    const sd::LongType expInStride,
    const sd::LongType expOutStride,
    // Shared gate strides
    const sd::LongType gateInStride,
    const sd::LongType gateOutStride,
    // Shared up strides
    const sd::LongType upInStride,
    const sd::LongType upOutStride,
    // Shared down strides
    const sd::LongType downInStride,
    const sd::LongType downOutStride,
    // Output strides
    const sd::LongType outBatchStride,
    const sd::LongType outSeqStride,
    const sd::LongType outDimStride,
    // Router probs strides (may be 0 if not requested)
    const sd::LongType rpBatchStride,
    const sd::LongType rpSeqStride,
    const sd::LongType rpExpStride,
    // Expert indices strides
    const sd::LongType eiBatchStride,
    const sd::LongType eiSeqStride,
    const sd::LongType eiKStride) {

    const sd::LongType tokenIdx = blockIdx.x;
    const sd::LongType totalTokens = batchSize * seqLen;
    if (tokenIdx >= totalTokens) return;

    const sd::LongType b = tokenIdx / seqLen;
    const sd::LongType s = tokenIdx % seqLen;

    const T* tokenIn = input + b * inBatchStride + s * inSeqStride;
    T* tokenOut = output + b * outBatchStride + s * outSeqStride;

    // Shared memory: routerLogits | selectedExperts | expertScores
    extern __shared__ char sharedMem[];
    T* routerLogits = reinterpret_cast<T*>(sharedMem);
    int* selectedExperts = reinterpret_cast<int*>(routerLogits + numRoutedExperts);
    T* expertScores = reinterpret_cast<T*>(selectedExperts + topK);

    // === Step 1: Shared expert (SwiGLU) — computed per output dimension ===
    // output[d] = sum_j( silu(gate_proj[d,j]) * up_proj[d,j] ) * down_proj[j,d]
    // We compute shared expert directly into tokenOut

    // Each thread handles part of hidden dims for down_proj output
    for (sd::LongType d = threadIdx.x; d < hiddenDim; d += blockDim.x) {
        T sharedVal = static_cast<T>(0);
        for (sd::LongType j = 0; j < sharedIntermediate; j++) {
            // gate_proj: input @ gate -> silu
            T gateVal = static_cast<T>(0);
            T upVal = static_cast<T>(0);
            for (sd::LongType id = 0; id < hiddenDim; id++) {
                T inVal = tokenIn[id * inDimStride];
                gateVal += inVal * sharedGateProj[id * gateInStride + j * gateOutStride];
                upVal += inVal * sharedUpProj[id * upInStride + j * upOutStride];
            }
            gateVal = deviceSilu(gateVal);
            // down_proj contribution
            sharedVal += (gateVal * upVal) * sharedDownProj[j * downInStride + d * downOutStride];
        }
        tokenOut[d * outDimStride] = sharedVal;
    }
    __syncthreads();

    // === Step 2: Router logits for routed experts ===
    for (int e = threadIdx.x; e < numRoutedExperts; e += blockDim.x) {
        T logit = static_cast<T>(0);
        for (sd::LongType d = 0; d < hiddenDim; d++) {
            logit += tokenIn[d * inDimStride] * routerWeights[d * rtrDimStride + e * rtrExpStride];
        }
        routerLogits[e] = logit;
    }
    __syncthreads();

    // Thread 0: softmax + top-K
    if (threadIdx.x == 0) {
        deviceSoftmaxMoe(routerLogits, numRoutedExperts);

        // Store router probs if requested
        if (routerProbsOut != nullptr) {
            T* rpToken = routerProbsOut + b * rpBatchStride + s * rpSeqStride;
            for (int e = 0; e < numRoutedExperts; e++) {
                rpToken[e * rpExpStride] = routerLogits[e];
            }
        }

        deviceTopKMoe(routerLogits, numRoutedExperts, topK, selectedExperts, expertScores);

        // Store expert indices if requested
        if (expertIndicesOut != nullptr) {
            sd::LongType* eiToken = expertIndicesOut + b * eiBatchStride + s * eiSeqStride;
            for (int k = 0; k < topK; k++) {
                eiToken[k * eiKStride] = static_cast<sd::LongType>(selectedExperts[k]);
            }
        }

        // Normalize scores
        if (normalizeProbs) {
            T scoreSum = static_cast<T>(0);
            for (int k = 0; k < topK; k++) {
                scoreSum += expertScores[k];
            }
            if (scoreSum > static_cast<T>(0)) {
                for (int k = 0; k < topK; k++) {
                    expertScores[k] /= scoreSum;
                }
            }
        }
    }
    __syncthreads();

    // === Step 3: Routed experts — add weighted output to shared output ===
    for (sd::LongType od = threadIdx.x; od < expertHidden && od < hiddenDim; od += blockDim.x) {
        T accumulator = static_cast<T>(0);
        for (int k = 0; k < topK; k++) {
            int expertIdx = selectedExperts[k];
            T score = expertScores[k];
            const T* expertW = routedExpertWeights + expertIdx * expExpStride;

            T expertVal = static_cast<T>(0);
            for (sd::LongType id = 0; id < hiddenDim; id++) {
                expertVal += tokenIn[id * inDimStride] * expertW[id * expInStride + od * expOutStride];
            }
            if (routedExpertBias != nullptr) {
                expertVal += routedExpertBias[expertIdx * expertHidden + od];
            }
            accumulator += score * expertVal;
        }
        tokenOut[od * outDimStride] += accumulator;
    }
}

template <typename T>
static void moeSharedExpertsCuda_(sd::LaunchContext* context,
                                   NDArray* input,
                                   NDArray* routerWeights,
                                   NDArray* routedExpertWeights,
                                   NDArray* sharedGateProj,
                                   NDArray* sharedUpProj,
                                   NDArray* sharedDownProj,
                                   NDArray* routedExpertBias,
                                   NDArray* output,
                                   NDArray* routerProbs,
                                   NDArray* expertIndices,
                                   int numRoutedExperts,
                                   int topK,
                                   bool normalizeProbs,
                                   double capacityFactor) {

    const auto batchSize = input->sizeAt(0);
    const auto seqLen = input->sizeAt(1);
    const auto hiddenDim = input->sizeAt(2);
    const auto expertHidden = routedExpertWeights->sizeAt(2);
    const auto sharedIntermediate = sharedGateProj->sizeAt(1);

    const sd::LongType totalTokens = batchSize * seqLen;

    auto dims = getLaunchDims("moe_shared_experts");
    const int blockSize = dims.y;

    // Shared memory: routerLogits + selectedExperts + expertScores
    size_t sharedMemSize = numRoutedExperts * sizeof(T) + topK * sizeof(int) + topK * sizeof(T);

    PointersManager manager(context, "moeSharedExperts");

    NDArray::prepareSpecialUse({output, routerProbs, expertIndices},
                                {input, routerWeights, routedExpertWeights,
                                 sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias});

    const T* inputBuf = input->specialBufferasT<T>();
    const T* routerBuf = routerWeights->specialBufferasT<T>();
    const T* expertBuf = routedExpertWeights->specialBufferasT<T>();
    const T* gateProjBuf = sharedGateProj->specialBufferasT<T>();
    const T* upProjBuf = sharedUpProj->specialBufferasT<T>();
    const T* downProjBuf = sharedDownProj->specialBufferasT<T>();
    const T* biasBuf = routedExpertBias != nullptr ? routedExpertBias->specialBufferasT<T>() : nullptr;
    T* outputBuf = output->specialBufferasT<T>();
    T* rpBuf = routerProbs != nullptr ? routerProbs->specialBufferasT<T>() : nullptr;
    sd::LongType* eiBuf = expertIndices != nullptr ? expertIndices->specialBuffer() != nullptr ?
        reinterpret_cast<sd::LongType*>(expertIndices->specialBuffer()) : nullptr : nullptr;

    // Strides
    sd::LongType rpBatchStride = 0, rpSeqStride = 0, rpExpStride = 0;
    if (routerProbs != nullptr) {
        rpBatchStride = routerProbs->strideAt(0);
        rpSeqStride = routerProbs->strideAt(1);
        rpExpStride = routerProbs->strideAt(2);
    }
    sd::LongType eiBatchStride = 0, eiSeqStride = 0, eiKStride = 0;
    if (expertIndices != nullptr) {
        eiBatchStride = expertIndices->strideAt(0);
        eiSeqStride = expertIndices->strideAt(1);
        eiKStride = expertIndices->strideAt(2);
    }

    moeSharedExpertsKernel<T><<<totalTokens, blockSize, sharedMemSize, *context->getCudaStream()>>>(
        inputBuf, routerBuf, expertBuf, gateProjBuf, upProjBuf, downProjBuf, biasBuf,
        outputBuf, rpBuf, eiBuf,
        batchSize, seqLen, hiddenDim, expertHidden, sharedIntermediate,
        numRoutedExperts, topK, normalizeProbs ? 1 : 0,
        // Input strides
        input->strideAt(0), input->strideAt(1), input->strideAt(2),
        // Router strides
        routerWeights->strideAt(0), routerWeights->strideAt(1),
        // Expert strides
        routedExpertWeights->strideAt(0), routedExpertWeights->strideAt(1), routedExpertWeights->strideAt(2),
        // Shared gate strides
        sharedGateProj->strideAt(0), sharedGateProj->strideAt(1),
        // Shared up strides
        sharedUpProj->strideAt(0), sharedUpProj->strideAt(1),
        // Shared down strides
        sharedDownProj->strideAt(0), sharedDownProj->strideAt(1),
        // Output strides
        output->strideAt(0), output->strideAt(1), output->strideAt(2),
        // Router probs strides
        rpBatchStride, rpSeqStride, rpExpStride,
        // Expert indices strides
        eiBatchStride, eiSeqStride, eiKStride);

    DebugHelper::checkGlobalErrorCode("moeSharedExperts kernel failed");

    NDArray::registerSpecialUse({output, routerProbs, expertIndices},
                                 {input, routerWeights, routedExpertWeights,
                                  sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias});

    manager.synchronize();
}

BUILD_SINGLE_TEMPLATE(void moeSharedExpertsCuda_,
                      (sd::LaunchContext* context, NDArray* input, NDArray* routerWeights,
                       NDArray* routedExpertWeights, NDArray* sharedGateProj, NDArray* sharedUpProj,
                       NDArray* sharedDownProj, NDArray* routedExpertBias,
                       NDArray* output, NDArray* routerProbs, NDArray* expertIndices,
                       int numRoutedExperts, int topK, bool normalizeProbs, double capacityFactor),
                      SD_FLOAT_TYPES);

void moeSharedExperts(sd::LaunchContext* context,
                       NDArray* input,
                       NDArray* routerWeights,
                       NDArray* routedExpertWeights,
                       NDArray* sharedGateProj,
                       NDArray* sharedUpProj,
                       NDArray* sharedDownProj,
                       NDArray* routedExpertBias,
                       NDArray* output,
                       NDArray* routerProbs,
                       NDArray* expertIndices,
                       int numRoutedExperts,
                       int topK,
                       bool normalizeProbs,
                       double capacityFactor) {

    BUILD_SINGLE_SELECTOR(input->dataType(), moeSharedExpertsCuda_,
                          (context, input, routerWeights, routedExpertWeights,
                           sharedGateProj, sharedUpProj, sharedDownProj, routedExpertBias,
                           output, routerProbs, expertIndices,
                           numRoutedExperts, topK, normalizeProbs, capacityFactor),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif
