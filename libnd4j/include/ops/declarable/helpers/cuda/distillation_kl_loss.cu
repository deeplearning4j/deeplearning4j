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
#include <ops/declarable/helpers/activations.h>
#include <array/NDArray.h>
#include <array/NDArrayFactory.h>
#include <helpers/DebugHelper.h>
#include <math/templatemath.h>
#include <cuda_runtime.h>
#include <cfloat>
#include <ops/declarable/helpers/cuda/device_primitives.cuh>

namespace sd {
namespace ops {
namespace helpers {

static constexpr int DKL_WARP_SIZE = 32;

// Accumulator/scratch type: double when T=double for precision, float otherwise.
template <typename T>
struct AccType { using type = float; };
template <>
struct AccType<double> { using type = double; };

// Kernel: Compute softmax with temperature for each row, store log-softmax
template <typename T>
SD_KERNEL void klSoftmaxKernel(
    const T* __restrict__ logits,
    typename AccType<T>::type* __restrict__ logSoftmax,
    typename AccType<T>::type* __restrict__ softmax,
    const LongType batch,
    const LongType classes,
    const float temperature) {

    using AccT = typename AccType<T>::type;

    const int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int lane = threadIdx.x % DKL_WARP_SIZE;
    const int wid = threadIdx.x / DKL_WARP_SIZE;
    const int numWarps = (blockDim.x + DKL_WARP_SIZE - 1) / DKL_WARP_SIZE;

    // Find max (broadcast to all threads for numerically stable softmax)
    AccT threadMax = -sd::DataTypeUtils::max<AccT>();
    for (LongType c = threadIdx.x; c < classes; c += blockDim.x) {
        AccT v = static_cast<AccT>(logits[b * classes + c]) / static_cast<AccT>(temperature);
        threadMax = sd::math::sd_max<AccT>(threadMax, v);
    }

    AccT rowMax = sd::device::blockAllReduceMax(threadMax, warpBuf);

    // Compute sum of exp
    AccT threadSum = static_cast<AccT>(0);
    for (LongType c = threadIdx.x; c < classes; c += blockDim.x) {
        AccT v = static_cast<AccT>(logits[b * classes + c]) / static_cast<AccT>(temperature) - rowMax;
        threadSum += sd::math::sd_exp<AccT, AccT>(v);
    }

    // blockReduceSum is sufficient here because only thread 0 uses rowSum (to write sharedLogSum)
    // but we need sharedLogSum visible to ALL threads for the final write loop.
    // Use blockAllReduceSum so every thread gets the total.
    AccT rowSum = sd::device::blockAllReduceSum(threadSum, warpBuf);
    AccT logSum = sd::math::sd_log<AccT, AccT>(rowSum);

    // Write log-softmax and softmax
    for (LongType c = threadIdx.x; c < classes; c += blockDim.x) {
        AccT ls = static_cast<AccT>(logits[b * classes + c]) / static_cast<AccT>(temperature) - rowMax - logSum;
        logSoftmax[b * classes + c] = ls;
        softmax[b * classes + c] = sd::math::sd_exp<AccT, AccT>(ls);
    }
}

// Kernel: Compute KL divergence sum per sample
// Templated on AccT since all inputs are already AccT scratch buffers.
template <typename AccT>
SD_KERNEL void klDivergenceKernel(
    const AccT* __restrict__ teacherSoftmax,
    const AccT* __restrict__ teacherLogSoftmax,
    const AccT* __restrict__ studentLogSoftmax,
    AccT* __restrict__ sampleLosses,
    const LongType batch,
    const LongType classes) {

    const int b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ char sharedMem[];
    AccT* warpBuf = reinterpret_cast<AccT*>(sharedMem);

    const int lane = threadIdx.x % DKL_WARP_SIZE;
    const int wid = threadIdx.x / DKL_WARP_SIZE;
    const int numWarps = (blockDim.x + DKL_WARP_SIZE - 1) / DKL_WARP_SIZE;

    AccT threadKL = static_cast<AccT>(0);
    for (LongType c = threadIdx.x; c < classes; c += blockDim.x) {
        AccT pt = teacherSoftmax[b * classes + c];
        AccT logPt = teacherLogSoftmax[b * classes + c];
        AccT logPs = studentLogSoftmax[b * classes + c];
        threadKL += pt * (logPt - logPs);
    }

    // Result only needed by thread 0 to write sampleLosses[b]
    AccT totalKL = sd::device::blockReduceSum(threadKL, warpBuf);

    if (threadIdx.x == 0)
        sampleLosses[b] = totalKL;
}

// Kernel: Sum sample losses to scalar
template <typename T>
SD_KERNEL void klSumLossKernel(
    const typename AccType<T>::type* __restrict__ klLosses,
    T* __restrict__ output,
    const LongType batch,
    const float tempSq,
    const float alpha) {

    using AccT = typename AccType<T>::type;

    AccT total = static_cast<AccT>(0);
    for (LongType i = threadIdx.x; i < batch; i += blockDim.x)
        total += klLosses[i];

    // Single-warp kernel — warpReduceSum is sufficient
    total = sd::device::warpReduceSum(total);

    if (threadIdx.x == 0)
        output[0] = static_cast<T>(static_cast<AccT>(alpha) * static_cast<AccT>(tempSq) * total / static_cast<AccT>(batch));
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
    using AccT = typename AccType<T>::type;
    auto studentLogSm = reinterpret_cast<AccT*>(vStudentLogSm);
    auto studentSm = reinterpret_cast<AccT*>(vStudentSm);
    auto teacherLogSm = reinterpret_cast<AccT*>(vTeacherLogSm);
    auto teacherSm = reinterpret_cast<AccT*>(vTeacherSm);
    auto klLosses = reinterpret_cast<AccT*>(vKlLosses);

    int smThreads = 256;
    if (classes < 256) {
        smThreads = ((classes + DKL_WARP_SIZE - 1) / DKL_WARP_SIZE) * DKL_WARP_SIZE;
        if (smThreads < DKL_WARP_SIZE) smThreads = DKL_WARP_SIZE;
    }
    int numWarps = (smThreads + DKL_WARP_SIZE - 1) / DKL_WARP_SIZE;
    size_t sharedSize = numWarps * sizeof(AccT);

    // Compute softmax for student and teacher
    klSoftmaxKernel<T><<<batch, smThreads, sharedSize, *stream>>>(
        studentLogits, studentLogSm, studentSm, batch, classes, temperature);
    DebugHelper::checkGlobalErrorCode("klSoftmaxKernel student failed");

    klSoftmaxKernel<T><<<batch, smThreads, sharedSize, *stream>>>(
        teacherLogits, teacherLogSm, teacherSm, batch, classes, temperature);
    DebugHelper::checkGlobalErrorCode("klSoftmaxKernel teacher failed");

    // KL divergence per sample
    klDivergenceKernel<AccT><<<batch, smThreads, sharedSize, *stream>>>(
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
    // Flatten leading dims for rank-3 [B,S,V] -> batch=B*S, classes=V so a rank-3
    // input gives the same per-sample-averaged loss as its reshaped rank-2 form
    // (matches the CPU helper distillation_kl_loss.cpp). Contiguous layout means the
    // kernels' logits[b*classes + c] indexing is already correct after flattening.
    const int rank = studentLogits->rankOf();
    auto batch = (rank == 3) ? studentLogits->sizeAt(0) * studentLogits->sizeAt(1)
                             : studentLogits->sizeAt(0);
    auto classes = (rank == 3) ? studentLogits->sizeAt(2)
                               : studentLogits->sizeAt(1);
    auto stream = context->getCudaStream();

    auto accDtype = studentLogits->dataType() == DataType::DOUBLE ? DataType::DOUBLE : DataType::FLOAT32;
    auto studentLogSm = NDArrayFactory::create('c', {batch, classes}, accDtype, context);
    auto studentSm = NDArrayFactory::create('c', {batch, classes}, accDtype, context);
    auto teacherLogSm = NDArrayFactory::create('c', {batch, classes}, accDtype, context);
    auto teacherSm = NDArrayFactory::create('c', {batch, classes}, accDtype, context);
    auto klLosses = NDArrayFactory::create('c', {batch}, accDtype, context);

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
    // dLdStudent = (softmax(student/T) - softmax(teacher/T)) * (alpha*T / N), where softmax is
    // taken over the classes axis (the last dimension) and N is the number of distributions
    // (product of the leading dims). helpers::softmax reduces along a TAD over that axis, so it
    // is rank-invariant: rank-2 [B*S,V] and rank-3 [B,S,V] give identical per-element gradients
    // with no reshape/broadcast handling here.
    const int rank = studentLogits->rankOf();
    const int softmaxDim = rank - 1;
    const sd::LongType batch = studentLogits->lengthOf() / studentLogits->sizeAt(rank - 1);

    NDArray::prepareSpecialUse({dLdStudent, dLdTeacher}, {studentLogits, teacherLogits});

    // Scaled logits, then softmax over the classes axis.
    NDArray sScaled(studentLogits->shapeInfo(), false, context);
    studentLogits->applyScalar<double>(scalar::Divide, temperature, &sScaled);
    NDArray tScaled(teacherLogits->shapeInfo(), false, context);
    teacherLogits->applyScalar<double>(scalar::Divide, temperature, &tScaled);

    NDArray sSoftmax(sScaled.shapeInfo(), false, context);
    softmax(context, &sScaled, &sSoftmax, softmaxDim);
    NDArray tSoftmax(tScaled.shapeInfo(), false, context);
    softmax(context, &tScaled, &tSoftmax, softmaxDim);

    // dLdStudent = (sSoftmax - tSoftmax) * scale   (elementwise, same shape as the logits)
    const double scale = alpha * temperature / static_cast<double>(batch);
    NDArray diff(sScaled.shapeInfo(), false, context);
    sSoftmax.applyPairwiseTransform(pairwise::Subtract, &tSoftmax, &diff);
    diff.applyScalar<double>(scalar::Multiply, scale, dLdStudent);

    // dLdTeacher = 0 (teacher is frozen)
    NDArray* zeroArr = NDArrayFactory::create_<float>(0.0f, context);
    dLdTeacher->assign(zeroArr);
    delete zeroArr;

    NDArray::registerSpecialUse({dLdStudent, dLdTeacher}, {studentLogits, teacherLogits});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
