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

#include <ops/declarable/helpers/distillation_kl_loss.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <math/templatemath.h>
#include <cuda_runtime.h>
#include <cfloat>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int DKL_WARP_SIZE = 32;

// Kernel: Compute softmax with temperature for each row, store log-softmax
template <typename T>
SD_KERNEL void klSoftmaxKernel(
    const T* __restrict__ logits,
    float* __restrict__ logSoftmax,
    float* __restrict__ softmax,
    const LongType batch,
    const LongType classes,
    const float temperature) {

    const int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % DKL_WARP_SIZE;
    const int wid = threadIdx.x / DKL_WARP_SIZE;
    const int numWarps = (blockDim.x + DKL_WARP_SIZE - 1) / DKL_WARP_SIZE;

    // Find max
    float threadMax = -FLT_MAX;
    for (LongType c = threadIdx.x; c < classes; c += blockDim.x) {
        float v = static_cast<float>(logits[b * classes + c]) / temperature;
        threadMax = sd::math::sd_max<float>(threadMax, v);
    }
    for (int offset = DKL_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadMax = sd::math::sd_max<float>(threadMax, __shfl_down_sync(0xffffffff, threadMax, offset));
    if (lane == 0) warpBuf[wid] = threadMax;
    __syncthreads();

    float rowMax = -FLT_MAX;
    if (threadIdx.x < numWarps) rowMax = warpBuf[threadIdx.x];
    for (int offset = DKL_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowMax = sd::math::sd_max<float>(rowMax, __shfl_down_sync(0xffffffff, rowMax, offset));
    __shared__ float sharedMax;
    if (threadIdx.x == 0) sharedMax = rowMax;
    __syncthreads();
    rowMax = sharedMax;

    // Compute sum of exp
    float threadSum = 0.0f;
    for (LongType c = threadIdx.x; c < classes; c += blockDim.x) {
        float v = static_cast<float>(logits[b * classes + c]) / temperature - rowMax;
        threadSum += sd::math::sd_exp<float, float>(v);
    }
    for (int offset = DKL_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadSum += __shfl_down_sync(0xffffffff, threadSum, offset);
    if (lane == 0) warpBuf[wid] = threadSum;
    __syncthreads();

    float rowSum = 0.0f;
    if (threadIdx.x < numWarps) rowSum = warpBuf[threadIdx.x];
    for (int offset = DKL_WARP_SIZE / 2; offset > 0; offset /= 2)
        rowSum += __shfl_down_sync(0xffffffff, rowSum, offset);
    __shared__ float sharedLogSum;
    if (threadIdx.x == 0) sharedLogSum = logf(rowSum);
    __syncthreads();

    // Write log-softmax and softmax
    for (LongType c = threadIdx.x; c < classes; c += blockDim.x) {
        float ls = static_cast<float>(logits[b * classes + c]) / temperature - rowMax - sharedLogSum;
        logSoftmax[b * classes + c] = ls;
        softmax[b * classes + c] = sd::math::sd_exp<float, float>(ls);
    }
}

// Kernel: Compute KL divergence sum per sample
SD_KERNEL void klDivergenceKernel(
    const float* __restrict__ teacherSoftmax,
    const float* __restrict__ teacherLogSoftmax,
    const float* __restrict__ studentLogSoftmax,
    float* __restrict__ sampleLosses,
    const LongType batch,
    const LongType classes) {

    const int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ char sharedMem[];
    float* warpBuf = reinterpret_cast<float*>(sharedMem);

    const int lane = threadIdx.x % DKL_WARP_SIZE;
    const int wid = threadIdx.x / DKL_WARP_SIZE;
    const int numWarps = (blockDim.x + DKL_WARP_SIZE - 1) / DKL_WARP_SIZE;

    float threadKL = 0.0f;
    for (LongType c = threadIdx.x; c < classes; c += blockDim.x) {
        float pt = teacherSoftmax[b * classes + c];
        float logPt = teacherLogSoftmax[b * classes + c];
        float logPs = studentLogSoftmax[b * classes + c];
        threadKL += pt * (logPt - logPs);
    }

    for (int offset = DKL_WARP_SIZE / 2; offset > 0; offset /= 2)
        threadKL += __shfl_down_sync(0xffffffff, threadKL, offset);
    if (lane == 0) warpBuf[wid] = threadKL;
    __syncthreads();

    float totalKL = 0.0f;
    if (threadIdx.x < numWarps) totalKL = warpBuf[threadIdx.x];
    for (int offset = DKL_WARP_SIZE / 2; offset > 0; offset /= 2)
        totalKL += __shfl_down_sync(0xffffffff, totalKL, offset);

    if (threadIdx.x == 0)
        sampleLosses[b] = totalKL;
}

// Kernel: Sum sample losses to scalar
template <typename T>
SD_KERNEL void klSumLossKernel(
    const float* __restrict__ klLosses,
    T* __restrict__ output,
    const LongType batch,
    const float tempSq,
    const float alpha) {

    float total = 0.0f;
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        total += klLosses[i];

    for (int offset = DKL_WARP_SIZE / 2; offset > 0; offset /= 2)
        total += __shfl_down_sync(0xffffffff, total, offset);

    if (threadIdx.x == 0)
        output[0] = static_cast<T>(alpha * tempSq * total / batch);
}

template <typename T>
void distillationKLLossCudaLauncher(const cudaStream_t* stream,
                                      const void* vStudentLogits, const void* vTeacherLogits,
                                      void* vOutput,
                                      void* vStudentLogSm, void* vStudentSm,
                                      void* vTeacherLogSm, void* vTeacherSm,
                                      void* vKlLosses,
                                      LongType batch, LongType classes,
                                      float temperature, float alpha) {
    auto studentLogits = reinterpret_cast<const T*>(vStudentLogits);
    auto teacherLogits = reinterpret_cast<const T*>(vTeacherLogits);
    auto output = reinterpret_cast<T*>(vOutput);
    auto studentLogSm = reinterpret_cast<float*>(vStudentLogSm);
    auto studentSm = reinterpret_cast<float*>(vStudentSm);
    auto teacherLogSm = reinterpret_cast<float*>(vTeacherLogSm);
    auto teacherSm = reinterpret_cast<float*>(vTeacherSm);
    auto klLosses = reinterpret_cast<float*>(vKlLosses);

    int smThreads = 256;
    if (classes < 256) {
        smThreads = ((classes + DKL_WARP_SIZE - 1) / DKL_WARP_SIZE) * DKL_WARP_SIZE;
        if (smThreads < DKL_WARP_SIZE) smThreads = DKL_WARP_SIZE;
    }
    int numWarps = (smThreads + DKL_WARP_SIZE - 1) / DKL_WARP_SIZE;
    size_t sharedSize = numWarps * sizeof(float);

    // Compute softmax for student and teacher
    klSoftmaxKernel<T><<<batch, smThreads, sharedSize, *stream>>>(
        studentLogits, studentLogSm, studentSm, batch, classes, temperature);
    DebugHelper::checkGlobalErrorCode("klSoftmaxKernel student failed");

    klSoftmaxKernel<T><<<batch, smThreads, sharedSize, *stream>>>(
        teacherLogits, teacherLogSm, teacherSm, batch, classes, temperature);
    DebugHelper::checkGlobalErrorCode("klSoftmaxKernel teacher failed");

    // KL divergence per sample
    klDivergenceKernel<<<batch, smThreads, sharedSize, *stream>>>(
        teacherSm, teacherLogSm, studentLogSm, klLosses, batch, classes);
    DebugHelper::checkGlobalErrorCode("klDivergenceKernel failed");

    // Sum to scalar
    klSumLossKernel<T><<<1, DKL_WARP_SIZE, 0, *stream>>>(
        klLosses, output, batch, temperature * temperature, alpha);
    DebugHelper::checkGlobalErrorCode("klSumLossKernel failed");
}

BUILD_SINGLE_TEMPLATE(void distillationKLLossCudaLauncher,
                      (const cudaStream_t* stream,
                       const void* vStudentLogits, const void* vTeacherLogits,
                       void* vOutput,
                       void* vStudentLogSm, void* vStudentSm,
                       void* vTeacherLogSm, void* vTeacherSm,
                       void* vKlLosses,
                       LongType batch, LongType classes,
                       float temperature, float alpha),
                      SD_FLOAT_TYPES);

void distillationKLLoss(NDArray* studentLogits, NDArray* teacherLogits,
                         NDArray* hardLabels, NDArray* output,
                         double temperature, double alpha,
                         LaunchContext* context) {
    auto batch = studentLogits->sizeAt(0);
    auto classes = studentLogits->sizeAt(1);
    auto stream = context->getCudaStream();

    auto studentLogSm = NDArrayFactory::create('c', {batch, classes}, DataType::FLOAT32, context);
    auto studentSm = NDArrayFactory::create('c', {batch, classes}, DataType::FLOAT32, context);
    auto teacherLogSm = NDArrayFactory::create('c', {batch, classes}, DataType::FLOAT32, context);
    auto teacherSm = NDArrayFactory::create('c', {batch, classes}, DataType::FLOAT32, context);
    auto klLosses = NDArrayFactory::create('c', {batch}, DataType::FLOAT32, context);

    NDArray::prepareSpecialUse({output}, {studentLogits, teacherLogits});

    BUILD_SINGLE_SELECTOR(studentLogits->dataType(), distillationKLLossCudaLauncher,
                          (stream,
                           studentLogits->specialBuffer(), teacherLogits->specialBuffer(),
                           output->specialBuffer(),
                           studentLogSm->specialBuffer(), studentSm->specialBuffer(),
                           teacherLogSm->specialBuffer(), teacherSm->specialBuffer(),
                           klLosses->specialBuffer(),
                           batch, classes,
                           static_cast<float>(temperature), static_cast<float>(alpha)),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output}, {studentLogits, teacherLogits});

    delete studentLogSm;
    delete studentSm;
    delete teacherLogSm;
    delete teacherSm;
    delete klLosses;
}

void distillationKLLossBp(NDArray* studentLogits, NDArray* teacherLogits,
                            NDArray* hardLabels,
                            NDArray* dLdStudent, NDArray* dLdTeacher,
                            double temperature, double alpha,
                            LaunchContext* context) {
    auto batch = studentLogits->sizeAt(0);
    auto classes = studentLogits->sizeAt(1);

    NDArray::prepareSpecialUse({dLdStudent, dLdTeacher}, {studentLogits, teacherLogits});

    // sScaled = studentLogits / temperature
    NDArray sScaled(studentLogits->shapeInfo(), false, context);
    studentLogits->applyScalar<double>(scalar::Divide, temperature, &sScaled);

    // tScaled = teacherLogits / temperature
    NDArray tScaled(teacherLogits->shapeInfo(), false, context);
    teacherLogits->applyScalar<double>(scalar::Divide, temperature, &tScaled);

    // Softmax via exp(x - max) / sum(exp(x - max))
    std::vector<sd::LongType> axes = {1};
    NDArray* sMax = sScaled.reduceAlongDimension(reduce::SameOps::Max, &axes, true);
    NDArray* tMax = tScaled.reduceAlongDimension(reduce::SameOps::Max, &axes, true);

    // sExp = exp(sScaled - sMax)
    NDArray sShifted(sScaled.shapeInfo(), false, context);
    sScaled.applyPairwiseTransform(pairwise::Subtract, sMax, &sShifted);
    NDArray sExp(sScaled.shapeInfo(), false, context);
    sShifted.applyTransform(transform::Exp, &sExp);

    // tExp = exp(tScaled - tMax)
    NDArray tShifted(tScaled.shapeInfo(), false, context);
    tScaled.applyPairwiseTransform(pairwise::Subtract, tMax, &tShifted);
    NDArray tExp(tScaled.shapeInfo(), false, context);
    tShifted.applyTransform(transform::Exp, &tExp);

    NDArray* sSum = sExp.reduceAlongDimension(reduce::SameOps::Sum, &axes, true);
    NDArray* tSum = tExp.reduceAlongDimension(reduce::SameOps::Sum, &axes, true);

    // sSoftmax = sExp / sSum, tSoftmax = tExp / tSum
    NDArray sSoftmax(sScaled.shapeInfo(), false, context);
    sExp.applyPairwiseTransform(pairwise::Divide, sSum, &sSoftmax);
    NDArray tSoftmax(tScaled.shapeInfo(), false, context);
    tExp.applyPairwiseTransform(pairwise::Divide, tSum, &tSoftmax);

    // dLdStudent = (sSoftmax - tSoftmax) * scale
    double scale = alpha * temperature / static_cast<double>(batch);
    NDArray diff(sScaled.shapeInfo(), false, context);
    sSoftmax.applyPairwiseTransform(pairwise::Subtract, &tSoftmax, &diff);
    diff.applyScalar<double>(scalar::Multiply, scale, dLdStudent);

    // dLdTeacher = 0 (teacher is frozen)
    NDArray* zeroArr = NDArrayFactory::create_<float>(0.0f, context);
    dLdTeacher->assign(zeroArr);
    delete zeroArr;

    delete sMax;
    delete tMax;
    delete sSum;
    delete tSum;

    NDArray::registerSpecialUse({dLdStudent, dLdTeacher}, {studentLogits, teacherLogits});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
