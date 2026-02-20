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

#include <execution/Threads.h>
#include <ops/declarable/helpers/signal.h>
#include <cmath>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sd {
namespace ops {
namespace helpers {

template <typename T>
static void dft_(LaunchContext* context, NDArray* input, int axis,
                  bool inverse, bool onesided, NDArray* output) {
    const auto inputShape = input->shapeOf();
    const auto inputRank = input->rankOf();

    // Normalize axis
    if (axis < 0) axis += inputRank;

    const auto N = input->sizeAt(axis);
    const auto M = onesided ? N / 2 + 1 : N;

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    // Calculate strides for the complex dimension
    sd::LongType outerSize = 1;
    sd::LongType innerSize = 1;
    for (int i = 0; i < axis; i++) outerSize *= inputShape[i];
    for (int i = axis + 1; i < inputRank - 1; i++) innerSize *= inputShape[i];  // -1 for complex dim

    const T sign = inverse ? static_cast<T>(1.0) : static_cast<T>(-1.0);
    const T scale = inverse ? static_cast<T>(1.0 / N) : static_cast<T>(1.0);

    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto outer = start_x; outer < stop_x; outer++) {
            for (auto k = start_y; k < stop_y; k++) {
                for (sd::LongType inner = 0; inner < innerSize; inner++) {
                    T sumReal = 0;
                    T sumImag = 0;

                    for (sd::LongType n = 0; n < N; n++) {
                        // Input index: [outer, n, inner, 0/1]
                        auto inOffset = outer * N * innerSize * 2 + n * innerSize * 2 + inner * 2;
                        T inReal = inputPtr[inOffset];
                        T inImag = inputPtr[inOffset + 1];

                        // DFT coefficient
                        T angle = sign * static_cast<T>(2.0 * M_PI * k * n / N);
                        T cosVal = std::cos(angle);
                        T sinVal = std::sin(angle);

                        // Complex multiplication: (inReal + i*inImag) * (cosVal + i*sinVal)
                        sumReal += inReal * cosVal - inImag * sinVal;
                        sumImag += inReal * sinVal + inImag * cosVal;
                    }

                    // Output index
                    auto outOffset = outer * M * innerSize * 2 + k * innerSize * 2 + inner * 2;
                    outputPtr[outOffset] = sumReal * scale;
                    outputPtr[outOffset + 1] = sumImag * scale;
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, outerSize, 1, 0, M, 1);
}

void dft(LaunchContext* context, NDArray* input, int axis,
         bool inverse, bool onesided, NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), dft_, (context, input, axis, inverse, onesided, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void stft_(LaunchContext* context, NDArray* signal, int frameStep,
                   NDArray* window, int frameLength, bool onesided, NDArray* output) {
    const auto batchSize = signal->sizeAt(0);
    const auto signalLength = signal->sizeAt(1);
    const auto numFrames = (signalLength - frameLength) / frameStep + 1;
    const auto numBins = onesided ? frameLength / 2 + 1 : frameLength;

    auto signalPtr = signal->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();
    const T* windowPtr = window ? window->bufferAsT<T>() : nullptr;

    auto func = PRAGMA_THREADS_FOR_2D {
        std::vector<T> frame(frameLength);
        std::vector<T> dftReal(numBins);
        std::vector<T> dftImag(numBins);

        for (auto b = start_x; b < stop_x; b++) {
            for (auto f = start_y; f < stop_y; f++) {
                // Extract and window the frame
                sd::LongType frameStart = f * frameStep;
                for (int i = 0; i < frameLength; i++) {
                    T val = signalPtr[b * signalLength + frameStart + i];
                    frame[i] = windowPtr ? val * windowPtr[i] : val;
                }

                // Compute DFT
                for (int k = 0; k < numBins; k++) {
                    T sumReal = 0;
                    T sumImag = 0;

                    for (int n = 0; n < frameLength; n++) {
                        T angle = static_cast<T>(-2.0 * M_PI * k * n / frameLength);
                        sumReal += frame[n] * std::cos(angle);
                        sumImag += frame[n] * std::sin(angle);
                    }

                    dftReal[k] = sumReal;
                    dftImag[k] = sumImag;
                }

                // Store output: [batch, frame, bin, 2]
                for (int k = 0; k < numBins; k++) {
                    auto outOffset = b * numFrames * numBins * 2 + f * numBins * 2 + k * 2;
                    outputPtr[outOffset] = dftReal[k];
                    outputPtr[outOffset + 1] = dftImag[k];
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, numFrames, 1);
}

void stft(LaunchContext* context, NDArray* signal, int frameStep,
          NDArray* window, int frameLength, bool onesided, NDArray* output) {
    BUILD_SINGLE_SELECTOR(signal->dataType(), stft_, (context, signal, frameStep, window, frameLength, onesided, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void hannWindow_(LaunchContext* context, int size, bool periodic, NDArray* output) {
    auto outputPtr = output->bufferAsT<T>();
    const int denom = periodic ? size : size - 1;

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
            outputPtr[i] = static_cast<T>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / denom)));
        }
    };

    samediff::Threads::parallel_for(func, 0, size, 1);
}

void hannWindow(LaunchContext* context, int size, bool periodic, NDArray* output) {
    BUILD_SINGLE_SELECTOR(output->dataType(), hannWindow_, (context, size, periodic, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void hammingWindow_(LaunchContext* context, int size, bool periodic, NDArray* output) {
    auto outputPtr = output->bufferAsT<T>();
    const int denom = periodic ? size : size - 1;

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
            outputPtr[i] = static_cast<T>(0.54 - 0.46 * std::cos(2.0 * M_PI * i / denom));
        }
    };

    samediff::Threads::parallel_for(func, 0, size, 1);
}

void hammingWindow(LaunchContext* context, int size, bool periodic, NDArray* output) {
    BUILD_SINGLE_SELECTOR(output->dataType(), hammingWindow_, (context, size, periodic, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void blackmanWindow_(LaunchContext* context, int size, bool periodic, NDArray* output) {
    auto outputPtr = output->bufferAsT<T>();
    const int denom = periodic ? size : size - 1;

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
            outputPtr[i] = static_cast<T>(0.42 - 0.5 * std::cos(2.0 * M_PI * i / denom) +
                                          0.08 * std::cos(4.0 * M_PI * i / denom));
        }
    };

    samediff::Threads::parallel_for(func, 0, size, 1);
}

void blackmanWindow(LaunchContext* context, int size, bool periodic, NDArray* output) {
    BUILD_SINGLE_SELECTOR(output->dataType(), blackmanWindow_, (context, size, periodic, output),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
