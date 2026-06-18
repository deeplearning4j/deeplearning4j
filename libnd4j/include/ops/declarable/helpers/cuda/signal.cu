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

#include <ops/declarable/helpers/signal.h>
#include <helpers/PointersManager.h>
#include <system/op_boilerplate.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
SD_KERNEL SD_INLINE void dftKernel(const T* input, T* output,
                           sd::LongType outerSize, sd::LongType innerSize,
                           sd::LongType N, sd::LongType M,
                           T sign, T scale) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto totalElements = outerSize * M * innerSize;

    if (tid >= totalElements) return;

    const auto outer = tid / (M * innerSize);
    const auto ki = tid % (M * innerSize);
    const auto k = ki / innerSize;
    const auto inner = ki % innerSize;

    T sumReal = 0;
    T sumImag = 0;

    for (sd::LongType n = 0; n < N; n++) {
        auto inOffset = outer * N * innerSize * 2 + n * innerSize * 2 + inner * 2;
        T inReal = input[inOffset];
        T inImag = input[inOffset + 1];

        T angle = sign * static_cast<T>(2.0 * M_PI * k * n / N);
        T cosVal = cosf(angle);
        T sinVal = sinf(angle);

        sumReal += inReal * cosVal - inImag * sinVal;
        sumImag += inReal * sinVal + inImag * cosVal;
    }

    auto outOffset = outer * M * innerSize * 2 + k * innerSize * 2 + inner * 2;
    output[outOffset] = sumReal * scale;
    output[outOffset + 1] = sumImag * scale;
}

template <typename T>
static void dft_(LaunchContext* context, NDArray* input, int axis,
                  bool inverse, bool onesided, NDArray* output) {
    const auto inputShape = input->shapeOf();
    const auto inputRank = input->rankOf();

    if (axis < 0) axis += inputRank;

    const auto N = input->sizeAt(axis);
    const auto M = onesided ? N / 2 + 1 : N;

    sd::LongType outerSize = 1;
    sd::LongType innerSize = 1;
    for (int i = 0; i < axis; i++) outerSize *= inputShape[i];
    for (int i = axis + 1; i < inputRank - 1; i++) innerSize *= inputShape[i];

    const T sign = inverse ? static_cast<T>(1.0) : static_cast<T>(-1.0);
    const T scale = inverse ? static_cast<T>(1.0 / N) : static_cast<T>(1.0);

    const auto totalElements = outerSize * M * innerSize;
    const int blockSize = 256;
    const int numBlocks = (totalElements + blockSize - 1) / blockSize;

    PointersManager manager(context, "dft");

    dftKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        outerSize, innerSize, N, M, sign, scale);

    manager.synchronize();
}

void dft(LaunchContext* context, NDArray* input, int axis,
         bool inverse, bool onesided, NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), dft_, (context, input, axis, inverse, onesided, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

template <typename T>
SD_KERNEL SD_INLINE void stftKernel(const T* signal, const T* window, T* output,
                            sd::LongType batchSize, sd::LongType signalLength,
                            int frameStep, int frameLength,
                            sd::LongType numFrames, sd::LongType numBins,
                            bool hasWindow) {
    const auto tid = blockIdx.x * blockDim.x + threadIdx.x;
    const auto totalElements = batchSize * numFrames * numBins;

    if (tid >= totalElements) return;

    const auto b = tid / (numFrames * numBins);
    const auto fk = tid % (numFrames * numBins);
    const auto f = fk / numBins;
    const auto k = fk % numBins;

    sd::LongType frameStart = f * frameStep;

    T sumReal = 0;
    T sumImag = 0;

    for (int n = 0; n < frameLength; n++) {
        T val = signal[b * signalLength + frameStart + n];
        if (hasWindow) {
            val *= window[n];
        }

        T angle = static_cast<T>(-2.0 * M_PI * k * n / frameLength);
        sumReal += val * cosf(angle);
        sumImag += val * sinf(angle);
    }

    auto outOffset = b * numFrames * numBins * 2 + f * numBins * 2 + k * 2;
    output[outOffset] = sumReal;
    output[outOffset + 1] = sumImag;
}

template <typename T>
static void stft_(LaunchContext* context, NDArray* signal, int frameStep,
                   NDArray* window, int frameLength, bool onesided, NDArray* output) {
    const auto batchSize = signal->sizeAt(0);
    const auto signalLength = signal->sizeAt(1);
    const auto numFrames = (signalLength - frameLength) / frameStep + 1;
    const auto numBins = onesided ? frameLength / 2 + 1 : frameLength;

    const auto totalElements = batchSize * numFrames * numBins;
    const int blockSize = 256;
    const int numBlocks = (totalElements + blockSize - 1) / blockSize;

    PointersManager manager(context, "stft");

    const T* windowPtr = window ? reinterpret_cast<const T*>(window->specialBuffer()) : nullptr;

    stftKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(signal->specialBuffer()),
        windowPtr,
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, signalLength,
        frameStep, frameLength,
        numFrames, numBins,
        window != nullptr);

    manager.synchronize();
}

void stft(LaunchContext* context, NDArray* signal, int frameStep,
          NDArray* window, int frameLength, bool onesided, NDArray* output) {
    NDArray::prepareSpecialUse({output}, {signal});
    if (window) NDArray::prepareSpecialUse({}, {window});

    BUILD_SINGLE_SELECTOR(signal->dataType(), stft_, (context, signal, frameStep, window, frameLength, onesided, output),
                          SD_FLOAT_TYPES);

    NDArray::registerSpecialUse({output}, {signal});
    if (window) NDArray::registerSpecialUse({}, {window});
}

template <typename T>
SD_KERNEL SD_INLINE void hannWindowKernel(T* output, int size, int denom) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    output[i] = static_cast<T>(0.5 * (1.0 - cosf(2.0 * M_PI * i / denom)));
}

template <typename T>
static void hannWindow_(LaunchContext* context, int size, bool periodic, NDArray* output) {
    const int denom = periodic ? size : size - 1;
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;

    PointersManager manager(context, "hannWindow");

    hannWindowKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<T*>(output->specialBuffer()), size, denom);

    manager.synchronize();
}

void hannWindow(LaunchContext* context, int size, bool periodic, NDArray* output) {
    NDArray::prepareSpecialUse({output}, {});
    BUILD_SINGLE_SELECTOR(output->dataType(), hannWindow_, (context, size, periodic, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {});
}

template <typename T>
SD_KERNEL SD_INLINE void hammingWindowKernel(T* output, int size, int denom) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    output[i] = static_cast<T>(0.54 - 0.46 * cosf(2.0 * M_PI * i / denom));
}

template <typename T>
static void hammingWindow_(LaunchContext* context, int size, bool periodic, NDArray* output) {
    const int denom = periodic ? size : size - 1;
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;

    PointersManager manager(context, "hammingWindow");

    hammingWindowKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<T*>(output->specialBuffer()), size, denom);

    manager.synchronize();
}

void hammingWindow(LaunchContext* context, int size, bool periodic, NDArray* output) {
    NDArray::prepareSpecialUse({output}, {});
    BUILD_SINGLE_SELECTOR(output->dataType(), hammingWindow_, (context, size, periodic, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {});
}

template <typename T>
SD_KERNEL SD_INLINE void blackmanWindowKernel(T* output, int size, int denom) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) return;

    output[i] = static_cast<T>(0.42 - 0.5 * cosf(2.0 * M_PI * i / denom) +
                               0.08 * cosf(4.0 * M_PI * i / denom));
}

template <typename T>
static void blackmanWindow_(LaunchContext* context, int size, bool periodic, NDArray* output) {
    const int denom = periodic ? size : size - 1;
    const int blockSize = 256;
    const int numBlocks = (size + blockSize - 1) / blockSize;

    PointersManager manager(context, "blackmanWindow");

    blackmanWindowKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<T*>(output->specialBuffer()), size, denom);

    manager.synchronize();
}

void blackmanWindow(LaunchContext* context, int size, bool periodic, NDArray* output) {
    NDArray::prepareSpecialUse({output}, {});
    BUILD_SINGLE_SELECTOR(output->dataType(), blackmanWindow_, (context, size, periodic, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
