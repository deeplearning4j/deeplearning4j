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

#include <cuda_runtime.h>
#include <helpers/DebugHelper.h>
#include <helpers/MmulHelper.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <math/templatemath.h>
#include <types/float16.h>
#include <ops/declarable/helpers/gated_delta_net_block.h>
#include <ops/declarable/helpers/gated_delta_rule.h>

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
SD_KERNEL void sigmoidKernel(T* __restrict__ data, const LongType len) {
    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) return;
    // Compute in float32 to prevent FP16 overflow in exp
    float val = static_cast<float>(data[idx]);
    data[idx] = static_cast<T>(1.0f / (1.0f + sd::math::sd_exp<float, float>(-val)));
}

template <typename T>
SD_KERNEL void l2NormalizeKernel(T* __restrict__ k, const LongType totalVecs, const LongType headDimK) {
    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= totalVecs) return;

    T* base = k + idx * headDimK;
    // Accumulate in float32 to prevent FP16 overflow for large headDimK
    float normSq = 0.0f;
    for (LongType dk = 0; dk < headDimK; ++dk) {
        float val = static_cast<float>(base[dk]);
        normSq += val * val;
    }
    float invNorm = rsqrtf(normSq + 1e-12f);
    for (LongType dk = 0; dk < headDimK; ++dk)
        base[dk] = static_cast<T>(static_cast<float>(base[dk]) * invNorm);
}

template <typename T>
SD_KERNEL void qkvSplitKernel(
    const T* __restrict__ qkv, T* __restrict__ q, T* __restrict__ k, T* __restrict__ v,
    const LongType BL, const LongType numHeads, const LongType headDimK, const LongType headDimV,
    const LongType qkvDim) {

    const LongType idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= BL) return;

    const T* src = qkv + idx * qkvDim;
    LongType offset = 0;
    for (LongType h = 0; h < numHeads; ++h) {
        for (LongType dk = 0; dk < headDimK; ++dk)
            q[(idx * numHeads + h) * headDimK + dk] = src[offset++];
        for (LongType dk = 0; dk < headDimK; ++dk)
            k[(idx * numHeads + h) * headDimK + dk] = src[offset++];
        for (LongType dv = 0; dv < headDimV; ++dv)
            v[(idx * numHeads + h) * headDimV + dv] = src[offset++];
    }
}

template <typename T>
SD_KERNEL void gdnRmsNormKernel(T* __restrict__ data, const LongType numRows, const LongType rowLen, const float eps) {
    const LongType row = blockIdx.x;
    if (row >= numRows) return;

    extern __shared__ char sharedMem[];
    float* sdata = reinterpret_cast<float*>(sharedMem);

    T* rowPtr = data + row * rowLen;

    float threadSumSq = 0.0f;
    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x) {
        float val = static_cast<float>(rowPtr[i]);
        threadSumSq += val * val;
    }

    // Warp-level reduction
    for (int offset = 16; offset > 0; offset /= 2)
        threadSumSq += __shfl_down_sync(0xffffffff, threadSumSq, offset);

    int wid = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    if (lane == 0) sdata[wid] = threadSumSq;
    __syncthreads();

    // Cross-warp reduction
    int numWarps = (blockDim.x + 31) / 32;
    threadSumSq = (threadIdx.x < numWarps) ? sdata[threadIdx.x] : 0.0f;
    if (wid == 0) {
        for (int offset = 16; offset > 0; offset /= 2)
            threadSumSq += __shfl_down_sync(0xffffffff, threadSumSq, offset);
    }

    __shared__ float invRms;
    if (threadIdx.x == 0)
        invRms = rsqrtf(threadSumSq / static_cast<float>(rowLen) + eps);
    __syncthreads();

    for (LongType i = threadIdx.x; i < rowLen; i += blockDim.x)
        rowPtr[i] = static_cast<T>(static_cast<float>(rowPtr[i]) * invRms);
}

void gatedDeltaNetBlock(LaunchContext* context,
                         NDArray* x, NDArray* Wqkv, NDArray* Wbeta, NDArray* Wgate,
                         NDArray* Wout, NDArray* convWeight, NDArray* convBias,
                         NDArray* stateIn,
                         NDArray* output, NDArray* recurrentStateOut, NDArray* convStateOut,
                         int numHeads, int headDimK, int headDimV, float rmsEps) {
    const auto B = x->sizeAt(0);
    const auto L = x->sizeAt(1);
    const auto D = x->sizeAt(2);
    const LongType BL = B * L;

    auto stream = context->getCudaStream();
    auto dtype = x->dataType();

    // Linear projections via cuBLAS
    std::vector<LongType> xReshapeVec = {BL, D};
    auto xReshaped = x->reshape('c', xReshapeVec);
    auto qkvProjected = MmulHelper::mmul(xReshaped, Wqkv);
    auto betaProjected = MmulHelper::mmul(xReshaped, Wbeta);
    auto gateProjected = MmulHelper::mmul(xReshaped, Wgate);

    std::vector<LongType> bgShape = {B, L, static_cast<LongType>(numHeads)};
    auto betaReshaped = betaProjected->reshape('c', bgShape);
    auto gateReshaped = gateProjected->reshape('c', bgShape);

    const LongType qkvDim = numHeads * (2 * headDimK + headDimV);
    std::vector<LongType> qkvReshapeVec = {B, L, qkvDim};
    auto qkvReshaped = qkvProjected->reshape('c', qkvReshapeVec);

    std::vector<LongType> qkShape = {B, L, static_cast<LongType>(numHeads), static_cast<LongType>(headDimK)};
    std::vector<LongType> vShape = {B, L, static_cast<LongType>(numHeads), static_cast<LongType>(headDimV)};
    auto Q = new NDArray('c', qkShape, dtype, context);
    auto Kmat = new NDArray('c', qkShape, dtype, context);
    auto V = new NDArray('c', vShape, dtype, context);

    int threads = 256;

    // QKV split kernel
    NDArray::prepareSpecialUse({Q, Kmat, V}, {qkvReshaped});
    if (dtype == DataType::FLOAT32) {
        qkvSplitKernel<float><<<(BL + threads - 1) / threads, threads, 0, *stream>>>(
            reinterpret_cast<const float*>(qkvReshaped->specialBuffer()),
            reinterpret_cast<float*>(Q->specialBuffer()),
            reinterpret_cast<float*>(Kmat->specialBuffer()),
            reinterpret_cast<float*>(V->specialBuffer()),
            BL, numHeads, headDimK, headDimV, qkvDim);
    } else if (dtype == DataType::DOUBLE) {
        qkvSplitKernel<double><<<(BL + threads - 1) / threads, threads, 0, *stream>>>(
            reinterpret_cast<const double*>(qkvReshaped->specialBuffer()),
            reinterpret_cast<double*>(Q->specialBuffer()),
            reinterpret_cast<double*>(Kmat->specialBuffer()),
            reinterpret_cast<double*>(V->specialBuffer()),
            BL, numHeads, headDimK, headDimV, qkvDim);
    } else if (dtype == DataType::HALF) {
        qkvSplitKernel<float16><<<(BL + threads - 1) / threads, threads, 0, *stream>>>(
            reinterpret_cast<const float16*>(qkvReshaped->specialBuffer()),
            reinterpret_cast<float16*>(Q->specialBuffer()),
            reinterpret_cast<float16*>(Kmat->specialBuffer()),
            reinterpret_cast<float16*>(V->specialBuffer()),
            BL, numHeads, headDimK, headDimV, qkvDim);
    }
    DebugHelper::checkGlobalErrorCode("qkvSplitKernel failed");
    NDArray::registerSpecialUse({Q, Kmat, V}, {qkvReshaped});

    // Sigmoid on beta
    const LongType betaLen = betaReshaped->lengthOf();
    NDArray::prepareSpecialUse({betaReshaped}, {betaReshaped});
    if (dtype == DataType::FLOAT32)
        sigmoidKernel<float><<<(betaLen + threads - 1) / threads, threads, 0, *stream>>>(reinterpret_cast<float*>(betaReshaped->specialBuffer()), betaLen);
    else if (dtype == DataType::DOUBLE)
        sigmoidKernel<double><<<(betaLen + threads - 1) / threads, threads, 0, *stream>>>(reinterpret_cast<double*>(betaReshaped->specialBuffer()), betaLen);
    else if (dtype == DataType::HALF)
        sigmoidKernel<float16><<<(betaLen + threads - 1) / threads, threads, 0, *stream>>>(reinterpret_cast<float16*>(betaReshaped->specialBuffer()), betaLen);
    DebugHelper::checkGlobalErrorCode("sigmoidKernel failed");
    NDArray::registerSpecialUse({betaReshaped}, {betaReshaped});

    // L2-normalize K
    const LongType totalVecs = BL * numHeads;
    NDArray::prepareSpecialUse({Kmat}, {Kmat});
    if (dtype == DataType::FLOAT32)
        l2NormalizeKernel<float><<<(totalVecs + threads - 1) / threads, threads, 0, *stream>>>(reinterpret_cast<float*>(Kmat->specialBuffer()), totalVecs, headDimK);
    else if (dtype == DataType::DOUBLE)
        l2NormalizeKernel<double><<<(totalVecs + threads - 1) / threads, threads, 0, *stream>>>(reinterpret_cast<double*>(Kmat->specialBuffer()), totalVecs, headDimK);
    else if (dtype == DataType::HALF)
        l2NormalizeKernel<float16><<<(totalVecs + threads - 1) / threads, threads, 0, *stream>>>(reinterpret_cast<float16*>(Kmat->specialBuffer()), totalVecs, headDimK);
    DebugHelper::checkGlobalErrorCode("l2NormalizeKernel failed");
    NDArray::registerSpecialUse({Kmat}, {Kmat});

    // Gated delta rule recurrence
    auto gdnOutput = new NDArray('c', vShape, dtype, context);
    gatedDeltaRule(context, Q, Kmat, V, betaReshaped, gateReshaped,
                   stateIn, gdnOutput, recurrentStateOut);

    // RMSNorm on output
    const LongType hdv = static_cast<LongType>(numHeads) * headDimV;
    std::vector<LongType> flatShape = {BL, hdv};
    auto gdnFlat = gdnOutput->reshape('c', flatShape);

    NDArray::prepareSpecialUse({gdnFlat}, {gdnFlat});
    {
        int tpb = 256;
        if (hdv > 256) tpb = 512;
        if (hdv > 512) tpb = 1024;
        int numWarps = (tpb + 31) / 32;
        size_t sharedMemSize = numWarps * sizeof(float);

        if (dtype == DataType::FLOAT32)
            gdnRmsNormKernel<float><<<BL, tpb, sharedMemSize, *stream>>>(reinterpret_cast<float*>(gdnFlat->specialBuffer()), BL, hdv, rmsEps);
        else if (dtype == DataType::DOUBLE)
            gdnRmsNormKernel<double><<<BL, tpb, sharedMemSize, *stream>>>(reinterpret_cast<double*>(gdnFlat->specialBuffer()), BL, hdv, rmsEps);
        else if (dtype == DataType::HALF)
            gdnRmsNormKernel<float16><<<BL, tpb, sharedMemSize, *stream>>>(reinterpret_cast<float16*>(gdnFlat->specialBuffer()), BL, hdv, rmsEps);
        DebugHelper::checkGlobalErrorCode("gdnRmsNormKernel failed");
    }
    NDArray::registerSpecialUse({gdnFlat}, {gdnFlat});

    // Output projection via cuBLAS
    auto finalOutput = MmulHelper::mmul(gdnFlat, Wout);
    std::vector<LongType> finalReshapeVec = {B, L, D};
    auto finalReshaped = finalOutput->reshape('c', finalReshapeVec);
    output->assign(finalReshaped);

    convStateOut->nullify();

    delete xReshaped;
    delete qkvProjected;
    delete betaProjected;
    delete gateProjected;
    delete betaReshaped;
    delete gateReshaped;
    delete qkvReshaped;
    delete Q;
    delete Kmat;
    delete V;
    delete gdnOutput;
    delete gdnFlat;
    delete finalOutput;
    delete finalReshaped;
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
