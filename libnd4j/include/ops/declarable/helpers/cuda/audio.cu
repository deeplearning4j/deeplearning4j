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

#include <ops/declarable/helpers/audio.h>
#include <helpers/PointersManager.h>
#include <system/op_boilerplate.h>
#include <cuda_runtime.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sd {
namespace ops {
namespace helpers {

// Device functions for mel conversion
SD_DEVICE SD_INLINE double d_hzToMel(double hz) {
    return 2595.0 * log10(1.0 + hz / 700.0);
}

SD_DEVICE SD_INLINE double d_melToHz(double mel) {
    return 700.0 * (pow(10.0, mel / 2595.0) - 1.0);
}

// ======================== Mel Filterbank ========================

template <typename T>
SD_KERNEL void melFilterbankKernel(T* output, int numMelBins, int numFreqBins, int fftSize,
                                    int sampleRate, double lowerMel, double upperMel) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = numMelBins * numFreqBins;
    if (idx >= total) return;

    const auto m = idx / numFreqBins;
    const auto k = idx % numFreqBins;

    double melStep = (upperMel - lowerMel) / (numMelBins + 1);
    double fLeft = d_melToHz(lowerMel + m * melStep) * fftSize / sampleRate;
    double fCenter = d_melToHz(lowerMel + (m + 1) * melStep) * fftSize / sampleRate;
    double fRight = d_melToHz(lowerMel + (m + 2) * melStep) * fftSize / sampleRate;

    double freq = static_cast<double>(k);
    T val = 0;

    if (freq >= fLeft && freq <= fCenter && fCenter > fLeft) {
        val = static_cast<T>((freq - fLeft) / (fCenter - fLeft));
    } else if (freq > fCenter && freq <= fRight && fRight > fCenter) {
        val = static_cast<T>((fRight - freq) / (fRight - fCenter));
    }

    output[idx] = val;
}

template <typename T>
static void melFilterbank_(LaunchContext* context, int numMelBins, int fftSize,
                            int sampleRate, double lowerEdgeHz, double upperEdgeHz,
                            NDArray* output) {
    const int numFreqBins = fftSize / 2 + 1;
    double lowerMel = 2595.0 * std::log10(1.0 + lowerEdgeHz / 700.0);
    double upperMel = 2595.0 * std::log10(1.0 + upperEdgeHz / 700.0);

    const auto total = numMelBins * numFreqBins;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;

    PointersManager manager(context, "melFilterbank");

    melFilterbankKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<T*>(output->specialBuffer()),
        numMelBins, numFreqBins, fftSize, sampleRate, lowerMel, upperMel);

    manager.synchronize();
}

void melFilterbank(LaunchContext* context, int numMelBins, int fftSize,
                    int sampleRate, double lowerEdgeHz, double upperEdgeHz,
                    NDArray* output) {
    NDArray::prepareSpecialUse({output}, {});
    BUILD_SINGLE_SELECTOR(output->dataType(), melFilterbank_,
                          (context, numMelBins, fftSize, sampleRate, lowerEdgeHz, upperEdgeHz, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {});
}

// ======================== Pre-Emphasis ========================

template <typename T>
SD_KERNEL void preEmphasisKernel(const T* input, T* output,
                                  sd::LongType batchSize, sd::LongType numSamples,
                                  T coefficient) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numSamples;
    if (idx >= total) return;

    const auto b = idx / numSamples;
    const auto i = idx % numSamples;

    if (i == 0) {
        output[idx] = input[idx];
    } else {
        output[idx] = input[idx] - coefficient * input[idx - 1];
    }
}

template <typename T>
static void preEmphasis_(LaunchContext* context, NDArray* input,
                          double coefficient, NDArray* output) {
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);

    const auto total = batchSize * numSamples;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;

    PointersManager manager(context, "preEmphasis");

    preEmphasisKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, numSamples, static_cast<T>(coefficient));

    manager.synchronize();
}

void preEmphasis(LaunchContext* context, NDArray* input,
                  double coefficient, NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), preEmphasis_,
                          (context, input, coefficient, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== Audio Normalize ========================

template <typename T>
SD_KERNEL void audioNormalizeKernel(const T* input, T* output,
                                     sd::LongType batchSize, sd::LongType numSamples,
                                     T targetLevel, bool useRms) {
    // Each block handles one batch element
    const auto b = blockIdx.x;
    if (b >= batchSize) return;

    extern __shared__ char sharedMem[];
    T* sharedVal = reinterpret_cast<T*>(sharedMem);

    const T* batchInput = input + b * numSamples;
    T* batchOutput = output + b * numSamples;

    // Compute level (peak or RMS)
    T localVal = 0;
    for (sd::LongType i = threadIdx.x; i < numSamples; i += blockDim.x) {
        T val = batchInput[i];
        if (useRms) {
            localVal += val * val;
        } else {
            T absVal = (val >= 0) ? val : -val;
            if (absVal > localVal) localVal = absVal;
        }
    }

    sharedVal[threadIdx.x] = localVal;
    __syncthreads();

    // Reduce
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (useRms) {
                sharedVal[threadIdx.x] += sharedVal[threadIdx.x + stride];
            } else {
                T other = sharedVal[threadIdx.x + stride];
                if (other > sharedVal[threadIdx.x]) sharedVal[threadIdx.x] = other;
            }
        }
        __syncthreads();
    }

    T currentLevel = sharedVal[0];
    if (useRms) currentLevel = sqrt(currentLevel / static_cast<T>(numSamples));

    T scale = (currentLevel > static_cast<T>(1e-10)) ? targetLevel / currentLevel : static_cast<T>(0);

    // Apply normalization
    for (sd::LongType i = threadIdx.x; i < numSamples; i += blockDim.x) {
        batchOutput[i] = batchInput[i] * scale;
    }
}

template <typename T>
static void audioNormalize_(LaunchContext* context, NDArray* input,
                             double targetLevel, bool useRms,
                             NDArray* output) {
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);

    const int blockSize = 256;
    const int sharedMemSize = blockSize * sizeof(T);

    PointersManager manager(context, "audioNormalize");

    audioNormalizeKernel<T><<<batchSize, blockSize, sharedMemSize, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, numSamples, static_cast<T>(targetLevel), useRms);

    manager.synchronize();
}

void audioNormalize(LaunchContext* context, NDArray* input,
                     double targetLevel, bool useRms,
                     NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), audioNormalize_,
                          (context, input, targetLevel, useRms, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== A-Weighting ========================

template <typename T>
SD_KERNEL void aWeightingKernel(const T* frequencies, T* output, sd::LongType length) {
    const auto i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= length) return;

    double f = static_cast<double>(frequencies[i]);
    double f2 = f * f;

    double num = 12194.0 * 12194.0 * f2 * f2;
    double den = (f2 + 20.6 * 20.6) *
                 sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9)) *
                 (f2 + 12194.0 * 12194.0);

    double ra = (den > 1e-30) ? num / den : 0.0;
    double aWeight = (ra > 1e-30) ? 20.0 * log10(ra) + 2.0 : -100.0;

    output[i] = static_cast<T>(aWeight);
}

template <typename T>
static void aWeighting_(LaunchContext* context, NDArray* frequencies,
                         NDArray* output) {
    const auto length = frequencies->lengthOf();
    const int blockSize = 256;
    const int numBlocks = (length + blockSize - 1) / blockSize;

    PointersManager manager(context, "aWeighting");

    aWeightingKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(frequencies->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        length);

    manager.synchronize();
}

void aWeighting(LaunchContext* context, NDArray* frequencies,
                 NDArray* output) {
    NDArray::prepareSpecialUse({output}, {frequencies});
    BUILD_SINGLE_SELECTOR(frequencies->dataType(), aWeighting_,
                          (context, frequencies, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {frequencies});
}

// ======================== Zero Crossing Rate ========================

template <typename T>
SD_KERNEL void zeroCrossingRateKernel(const T* input, T* output,
                                       sd::LongType batchSize, sd::LongType numSamples,
                                       int frameLength, int hopLength, sd::LongType numFrames) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numFrames;
    if (idx >= total) return;

    const auto b = idx / numFrames;
    const auto f = idx % numFrames;
    sd::LongType frameStart = f * hopLength;

    int crossings = 0;
    for (int i = 1; i < frameLength; i++) {
        T curr = input[b * numSamples + frameStart + i];
        T prev = input[b * numSamples + frameStart + i - 1];
        if ((curr >= 0 && prev < 0) || (curr < 0 && prev >= 0)) {
            crossings++;
        }
    }

    output[idx] = static_cast<T>(crossings) / static_cast<T>(frameLength - 1);
}

template <typename T>
static void zeroCrossingRate_(LaunchContext* context, NDArray* input,
                               int frameLength, int hopLength,
                               NDArray* output) {
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const auto numFrames = (numSamples - frameLength) / hopLength + 1;

    const auto total = batchSize * numFrames;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;

    PointersManager manager(context, "zeroCrossingRate");

    zeroCrossingRateKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, numSamples, frameLength, hopLength, numFrames);

    manager.synchronize();
}

void zeroCrossingRate(LaunchContext* context, NDArray* input,
                       int frameLength, int hopLength,
                       NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), zeroCrossingRate_,
                          (context, input, frameLength, hopLength, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== Spectral Centroid ========================

template <typename T>
SD_KERNEL void spectralCentroidKernel(const T* input, T* output,
                                       sd::LongType batchSize, sd::LongType numFreqBins,
                                       sd::LongType numFrames, T freqScale) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numFrames;
    if (idx >= total) return;

    const auto b = idx / numFrames;
    const auto f = idx % numFrames;

    T weightedSum = 0;
    T magnitudeSum = 0;

    for (sd::LongType k = 0; k < numFreqBins; k++) {
        T mag = input[b * numFreqBins * numFrames + k * numFrames + f];
        T freq = static_cast<T>(k) * freqScale;
        weightedSum += freq * mag;
        magnitudeSum += mag;
    }

    output[idx] = (magnitudeSum > static_cast<T>(1e-10)) ? weightedSum / magnitudeSum : static_cast<T>(0);
}

template <typename T>
static void spectralCentroid_(LaunchContext* context, NDArray* input,
                               int sampleRate, int fftSize,
                               NDArray* output) {
    const auto batchSize = input->sizeAt(0);
    const auto numFreqBins = input->sizeAt(1);
    const auto numFrames = input->sizeAt(2);
    const T freqScale = static_cast<T>(sampleRate) / static_cast<T>(fftSize);

    const auto total = batchSize * numFrames;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;

    PointersManager manager(context, "spectralCentroid");

    spectralCentroidKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, numFreqBins, numFrames, freqScale);

    manager.synchronize();
}

void spectralCentroid(LaunchContext* context, NDArray* input,
                       int sampleRate, int fftSize,
                       NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), spectralCentroid_,
                          (context, input, sampleRate, fftSize, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== Spectral Rolloff ========================

template <typename T>
SD_KERNEL void spectralRolloffKernel(const T* input, T* output,
                                      sd::LongType batchSize, sd::LongType numFreqBins,
                                      sd::LongType numFrames, T freqScale, T threshold) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numFrames;
    if (idx >= total) return;

    const auto b = idx / numFrames;
    const auto f = idx % numFrames;

    T totalEnergy = 0;
    for (sd::LongType k = 0; k < numFreqBins; k++) {
        totalEnergy += input[b * numFreqBins * numFrames + k * numFrames + f];
    }

    T cumulativeEnergy = 0;
    T rolloffFreq = 0;
    for (sd::LongType k = 0; k < numFreqBins; k++) {
        cumulativeEnergy += input[b * numFreqBins * numFrames + k * numFrames + f];
        if (cumulativeEnergy >= threshold * totalEnergy) {
            rolloffFreq = static_cast<T>(k) * freqScale;
            break;
        }
    }

    output[idx] = rolloffFreq;
}

template <typename T>
static void spectralRolloff_(LaunchContext* context, NDArray* input,
                              int sampleRate, int fftSize, double rolloffPercent,
                              NDArray* output) {
    const auto batchSize = input->sizeAt(0);
    const auto numFreqBins = input->sizeAt(1);
    const auto numFrames = input->sizeAt(2);
    const T freqScale = static_cast<T>(sampleRate) / static_cast<T>(fftSize);

    const auto total = batchSize * numFrames;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;

    PointersManager manager(context, "spectralRolloff");

    spectralRolloffKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, numFreqBins, numFrames, freqScale, static_cast<T>(rolloffPercent));

    manager.synchronize();
}

void spectralRolloff(LaunchContext* context, NDArray* input,
                      int sampleRate, int fftSize, double rolloffPercent,
                      NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), spectralRolloff_,
                          (context, input, sampleRate, fftSize, rolloffPercent, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== Chroma Features ========================

template <typename T>
SD_KERNEL void chromaFeaturesKernel(const T* input, T* output,
                                     sd::LongType batchSize, sd::LongType numFreqBins,
                                     sd::LongType numFrames, int numChroma,
                                     int sampleRate, int fftSize) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numChroma * numFrames;
    if (idx >= total) return;

    const auto b = idx / (numChroma * numFrames);
    const auto cf = idx % (numChroma * numFrames);
    const auto c = cf / numFrames;
    const auto f = cf % numFrames;

    T sum = 0;
    for (sd::LongType k = 1; k < numFreqBins; k++) {
        double freqHz = static_cast<double>(k) * sampleRate / fftSize;
        double semitone = 12.0 * log2(freqHz / 261.63);
        int chromaBin = static_cast<int>(fmod(semitone + 1200.0, 12.0));
        if (chromaBin < 0) chromaBin += numChroma;
        chromaBin = chromaBin % numChroma;

        if (chromaBin == c) {
            sum += input[b * numFreqBins * numFrames + k * numFrames + f];
        }
    }

    output[idx] = sum;
}

template <typename T>
static void chromaFeatures_(LaunchContext* context, NDArray* input,
                             int sampleRate, int fftSize, int numChroma,
                             NDArray* output) {
    const auto batchSize = input->sizeAt(0);
    const auto numFreqBins = input->sizeAt(1);
    const auto numFrames = input->sizeAt(2);

    const auto total = batchSize * numChroma * numFrames;
    const int blockSize = 256;
    const int numBlocks = (total + blockSize - 1) / blockSize;

    PointersManager manager(context, "chromaFeatures");

    chromaFeaturesKernel<T><<<numBlocks, blockSize, 0, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, numFreqBins, numFrames, numChroma, sampleRate, fftSize);

    manager.synchronize();
}

void chromaFeatures(LaunchContext* context, NDArray* input,
                     int sampleRate, int fftSize, int numChroma,
                     NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), chromaFeatures_,
                          (context, input, sampleRate, fftSize, numChroma, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== Mel Spectrogram (delegates to CPU for now) ========================
// Complex ops like mel_spectrogram, mfcc, griffin_lim, pitch_detection, audio_resample
// use the CPU implementation path for correctness. They can be optimized with dedicated
// CUDA kernels later.

template <typename T>
static void melSpectrogram_(LaunchContext* context, NDArray* input,
                             int sampleRate, int fftSize, int hopLength, int numMelBins,
                             double lowerEdgeHz, double upperEdgeHz, double power,
                             NDArray* output) {
    // Sync input to host for CPU-path computation
    input->syncToHost();

    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const int numFreqBins = fftSize / 2 + 1;
    const auto numFrames = (numSamples - fftSize) / hopLength + 1;

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    // Build mel filterbank
    double lowerMel = 2595.0 * std::log10(1.0 + lowerEdgeHz / 700.0);
    double upperMel = 2595.0 * std::log10(1.0 + upperEdgeHz / 700.0);
    std::vector<double> binPoints(numMelBins + 2);
    for (int i = 0; i < numMelBins + 2; i++) {
        double mel = lowerMel + (upperMel - lowerMel) * i / (numMelBins + 1);
        binPoints[i] = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0) * fftSize / sampleRate;
    }
    std::vector<T> melFb(numMelBins * numFreqBins, 0);
    for (int m = 0; m < numMelBins; m++) {
        double fLeft = binPoints[m], fCenter = binPoints[m + 1], fRight = binPoints[m + 2];
        for (int k = 0; k < numFreqBins; k++) {
            double freq = static_cast<double>(k);
            if (freq >= fLeft && freq <= fCenter && fCenter > fLeft)
                melFb[m * numFreqBins + k] = static_cast<T>((freq - fLeft) / (fCenter - fLeft));
            else if (freq > fCenter && freq <= fRight && fRight > fCenter)
                melFb[m * numFreqBins + k] = static_cast<T>((fRight - freq) / (fRight - fCenter));
        }
    }

    // Hann window
    std::vector<T> window(fftSize);
    for (int i = 0; i < fftSize; i++)
        window[i] = static_cast<T>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / fftSize)));

    for (sd::LongType b = 0; b < batchSize; b++) {
        for (sd::LongType f = 0; f < numFrames; f++) {
            sd::LongType frameStart = f * hopLength;
            std::vector<T> powerSpec(numFreqBins);

            for (int k = 0; k < numFreqBins; k++) {
                T sumReal = 0, sumImag = 0;
                for (int n = 0; n < fftSize; n++) {
                    T val = inputPtr[b * numSamples + frameStart + n] * window[n];
                    T angle = static_cast<T>(-2.0 * M_PI * k * n / fftSize);
                    sumReal += val * std::cos(angle);
                    sumImag += val * std::sin(angle);
                }
                T mag = std::sqrt(sumReal * sumReal + sumImag * sumImag);
                powerSpec[k] = (power == 2.0) ? mag * mag : static_cast<T>(std::pow(static_cast<float>(mag), static_cast<float>(power)));
            }

            for (int m = 0; m < numMelBins; m++) {
                T sum = 0;
                for (int k = 0; k < numFreqBins; k++)
                    sum += melFb[m * numFreqBins + k] * powerSpec[k];
                outputPtr[b * numMelBins * numFrames + m * numFrames + f] = sum;
            }
        }
    }

    output->syncToDevice();
}

void melSpectrogram(LaunchContext* context, NDArray* input,
                     int sampleRate, int fftSize, int hopLength, int numMelBins,
                     double lowerEdgeHz, double upperEdgeHz, double power,
                     NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), melSpectrogram_,
                          (context, input, sampleRate, fftSize, hopLength, numMelBins,
                           lowerEdgeHz, upperEdgeHz, power, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void mfcc_(LaunchContext* context, NDArray* input,
                   int sampleRate, int fftSize, int hopLength,
                   int numMelBins, int numMfcc,
                   double lowerEdgeHz, double upperEdgeHz,
                   NDArray* output) {
    input->syncToHost();

    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const int numFreqBins = fftSize / 2 + 1;
    const auto numFrames = (numSamples - fftSize) / hopLength + 1;

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    double lowerMel = 2595.0 * std::log10(1.0 + lowerEdgeHz / 700.0);
    double upperMel = 2595.0 * std::log10(1.0 + upperEdgeHz / 700.0);
    std::vector<double> binPoints(numMelBins + 2);
    for (int i = 0; i < numMelBins + 2; i++) {
        double mel = lowerMel + (upperMel - lowerMel) * i / (numMelBins + 1);
        binPoints[i] = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0) * fftSize / sampleRate;
    }
    std::vector<T> melFb(numMelBins * numFreqBins, 0);
    for (int m = 0; m < numMelBins; m++) {
        double fLeft = binPoints[m], fCenter = binPoints[m + 1], fRight = binPoints[m + 2];
        for (int k = 0; k < numFreqBins; k++) {
            double freq = static_cast<double>(k);
            if (freq >= fLeft && freq <= fCenter && fCenter > fLeft)
                melFb[m * numFreqBins + k] = static_cast<T>((freq - fLeft) / (fCenter - fLeft));
            else if (freq > fCenter && freq <= fRight && fRight > fCenter)
                melFb[m * numFreqBins + k] = static_cast<T>((fRight - freq) / (fRight - fCenter));
        }
    }

    std::vector<T> window(fftSize);
    for (int i = 0; i < fftSize; i++)
        window[i] = static_cast<T>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / fftSize)));

    for (sd::LongType b = 0; b < batchSize; b++) {
        for (sd::LongType f = 0; f < numFrames; f++) {
            sd::LongType frameStart = f * hopLength;
            std::vector<T> powerSpec(numFreqBins);
            std::vector<T> melEnergies(numMelBins);

            for (int k = 0; k < numFreqBins; k++) {
                T sumReal = 0, sumImag = 0;
                for (int n = 0; n < fftSize; n++) {
                    T val = inputPtr[b * numSamples + frameStart + n] * window[n];
                    T angle = static_cast<T>(-2.0 * M_PI * k * n / fftSize);
                    sumReal += val * std::cos(angle);
                    sumImag += val * std::sin(angle);
                }
                powerSpec[k] = sumReal * sumReal + sumImag * sumImag;
            }

            for (int m = 0; m < numMelBins; m++) {
                T sum = 0;
                for (int k = 0; k < numFreqBins; k++)
                    sum += melFb[m * numFreqBins + k] * powerSpec[k];
                melEnergies[m] = std::log(std::max(sum, static_cast<T>(1e-10)));
            }

            for (int c = 0; c < numMfcc; c++) {
                T sum = 0;
                for (int m = 0; m < numMelBins; m++)
                    sum += melEnergies[m] * std::cos(M_PI * c * (m + 0.5) / numMelBins);
                outputPtr[b * numMfcc * numFrames + c * numFrames + f] = sum;
            }
        }
    }

    output->syncToDevice();
}

void mfcc(LaunchContext* context, NDArray* input,
           int sampleRate, int fftSize, int hopLength,
           int numMelBins, int numMfcc,
           double lowerEdgeHz, double upperEdgeHz,
           NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), mfcc_,
                          (context, input, sampleRate, fftSize, hopLength, numMelBins, numMfcc,
                           lowerEdgeHz, upperEdgeHz, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void griffinLim_(LaunchContext* context, NDArray* magnitudeSpectrogram,
                         int fftSize, int hopLength, int numIterations,
                         NDArray* output) {
    magnitudeSpectrogram->syncToHost();

    const auto batchSize = magnitudeSpectrogram->sizeAt(0);
    const auto numFreqBins = magnitudeSpectrogram->sizeAt(1);
    const auto numFrames = magnitudeSpectrogram->sizeAt(2);
    const auto numSamples = (numFrames - 1) * hopLength + fftSize;

    auto magPtr = magnitudeSpectrogram->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    std::vector<T> window(fftSize);
    for (int i = 0; i < fftSize; i++)
        window[i] = static_cast<T>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / fftSize)));

    for (sd::LongType b = 0; b < batchSize; b++) {
        std::vector<T> phase(numFreqBins * numFrames, 0);
        std::vector<T> signal(numSamples, 0);
        std::vector<T> windowSum(numSamples, 0);

        for (int iter = 0; iter < numIterations; iter++) {
            std::fill(signal.begin(), signal.end(), static_cast<T>(0));
            std::fill(windowSum.begin(), windowSum.end(), static_cast<T>(0));

            for (sd::LongType f = 0; f < numFrames; f++) {
                sd::LongType frameStart = f * hopLength;
                for (int n = 0; n < fftSize; n++) {
                    T sum = 0;
                    for (int k = 0; k < numFreqBins; k++) {
                        T mag = magPtr[b * numFreqBins * numFrames + k * numFrames + f];
                        T ph = phase[k * numFrames + f];
                        T angle = static_cast<T>(2.0 * M_PI * k * n / fftSize);
                        sum += mag * std::cos(angle + ph);
                    }
                    signal[frameStart + n] += window[n] * sum / fftSize;
                    windowSum[frameStart + n] += window[n] * window[n];
                }
            }

            for (sd::LongType i = 0; i < numSamples; i++) {
                if (windowSum[i] > static_cast<T>(1e-8))
                    signal[i] /= windowSum[i];
            }

            if (iter < numIterations - 1) {
                for (sd::LongType f = 0; f < numFrames; f++) {
                    sd::LongType frameStart = f * hopLength;
                    for (int k = 0; k < numFreqBins; k++) {
                        T sumReal = 0, sumImag = 0;
                        for (int n = 0; n < fftSize; n++) {
                            T val = signal[frameStart + n] * window[n];
                            T angle = static_cast<T>(-2.0 * M_PI * k * n / fftSize);
                            sumReal += val * std::cos(angle);
                            sumImag += val * std::sin(angle);
                        }
                        phase[k * numFrames + f] = std::atan2(sumImag, sumReal);
                    }
                }
            }
        }

        for (sd::LongType i = 0; i < numSamples; i++)
            outputPtr[b * numSamples + i] = signal[i];
    }

    output->syncToDevice();
}

void griffinLim(LaunchContext* context, NDArray* magnitudeSpectrogram,
                 int fftSize, int hopLength, int numIterations,
                 NDArray* output) {
    BUILD_SINGLE_SELECTOR(magnitudeSpectrogram->dataType(), griffinLim_,
                          (context, magnitudeSpectrogram, fftSize, hopLength, numIterations, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void pitchDetection_(LaunchContext* context, NDArray* input,
                             int sampleRate, int frameLength, int hopLength,
                             double minFreq, double maxFreq,
                             NDArray* output) {
    input->syncToHost();

    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const auto numFrames = (numSamples - frameLength) / hopLength + 1;

    const int minLag = static_cast<int>(sampleRate / maxFreq);
    const int maxLag = static_cast<int>(sampleRate / minFreq);

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    for (sd::LongType b = 0; b < batchSize; b++) {
        for (sd::LongType f = 0; f < numFrames; f++) {
            sd::LongType frameStart = f * hopLength;
            T maxCorr = 0;
            int bestLag = 0;

            for (int lag = minLag; lag <= maxLag && lag < frameLength; lag++) {
                T corr = 0, normA = 0, normB = 0;
                for (int n = 0; n < frameLength - lag; n++) {
                    T a = inputPtr[b * numSamples + frameStart + n];
                    T bv = inputPtr[b * numSamples + frameStart + n + lag];
                    corr += a * bv;
                    normA += a * a;
                    normB += bv * bv;
                }
                T norm = std::sqrt(normA * normB);
                T normalizedCorr = (norm > static_cast<T>(1e-10)) ? corr / norm : static_cast<T>(0);
                if (normalizedCorr > maxCorr) {
                    maxCorr = normalizedCorr;
                    bestLag = lag;
                }
            }

            outputPtr[b * numFrames + f] = (bestLag > 0 && maxCorr > static_cast<T>(0.2))
                ? static_cast<T>(sampleRate) / static_cast<T>(bestLag) : static_cast<T>(0);
        }
    }

    output->syncToDevice();
}

void pitchDetection(LaunchContext* context, NDArray* input,
                     int sampleRate, int frameLength, int hopLength,
                     double minFreq, double maxFreq,
                     NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), pitchDetection_,
                          (context, input, sampleRate, frameLength, hopLength, minFreq, maxFreq, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void audioResample_(LaunchContext* context, NDArray* input,
                            int origSampleRate, int targetSampleRate,
                            NDArray* output) {
    input->syncToHost();

    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto origSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const auto targetSamples = hasBatch ? output->sizeAt(1) : output->sizeAt(0);

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    const double ratio = static_cast<double>(origSampleRate) / static_cast<double>(targetSampleRate);
    const int sincRadius = 8;

    for (sd::LongType b = 0; b < batchSize; b++) {
        for (sd::LongType i = 0; i < targetSamples; i++) {
            double srcPos = i * ratio;
            int srcCenter = static_cast<int>(srcPos);
            T sum = 0;
            T weightSum = 0;

            for (int j = srcCenter - sincRadius + 1; j <= srcCenter + sincRadius; j++) {
                if (j >= 0 && j < origSamples) {
                    double x = srcPos - j;
                    double weight;
                    if (std::abs(x) < 1e-10) {
                        weight = 1.0;
                    } else if (std::abs(x) < sincRadius) {
                        double pix = M_PI * x;
                        double pixOverA = M_PI * x / sincRadius;
                        weight = (std::sin(pix) / pix) * (std::sin(pixOverA) / pixOverA);
                    } else {
                        weight = 0.0;
                    }
                    sum += inputPtr[b * origSamples + j] * static_cast<T>(weight);
                    weightSum += static_cast<T>(weight);
                }
            }

            outputPtr[b * targetSamples + i] = (weightSum > static_cast<T>(1e-10))
                ? sum / weightSum : static_cast<T>(0);
        }
    }

    output->syncToDevice();
}

void audioResample(LaunchContext* context, NDArray* input,
                    int origSampleRate, int targetSampleRate,
                    NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), audioResample_,
                          (context, input, origSampleRate, targetSampleRate, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void whisperMelSpectrogram_(LaunchContext* context, NDArray* input,
                                    int sampleRate, int fftSize, int hopLength,
                                    int numMelBins, int targetFrames,
                                    double lowerEdgeHz, double upperEdgeHz,
                                    NDArray* output) {
    // Sync input to host for CPU-path computation (same pattern as melSpectrogram_)
    input->syncToHost();

    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const int numFreqBins = fftSize / 2 + 1;
    const auto rawNumFrames = (numSamples - fftSize) / hopLength + 1;

    // Build mel filterbank
    double lowerMel = 2595.0 * std::log10(1.0 + lowerEdgeHz / 700.0);
    double upperMel = 2595.0 * std::log10(1.0 + upperEdgeHz / 700.0);
    std::vector<double> binPoints(numMelBins + 2);
    for (int i = 0; i < numMelBins + 2; i++) {
        double mel = lowerMel + (upperMel - lowerMel) * i / (numMelBins + 1);
        binPoints[i] = 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0) * fftSize / sampleRate;
    }
    std::vector<T> melFb(numMelBins * numFreqBins, 0);
    for (int m = 0; m < numMelBins; m++) {
        double fLeft = binPoints[m], fCenter = binPoints[m + 1], fRight = binPoints[m + 2];
        for (int k = 0; k < numFreqBins; k++) {
            double freq = static_cast<double>(k);
            if (freq >= fLeft && freq <= fCenter && fCenter > fLeft)
                melFb[m * numFreqBins + k] = static_cast<T>((freq - fLeft) / (fCenter - fLeft));
            else if (freq > fCenter && freq <= fRight && fRight > fCenter)
                melFb[m * numFreqBins + k] = static_cast<T>((fRight - freq) / (fRight - fCenter));
        }
    }

    // Hann window
    std::vector<T> window(fftSize);
    for (int i = 0; i < fftSize; i++)
        window[i] = static_cast<T>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / fftSize)));

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    // Compute mel spectrogram per batch, pad/trim to targetFrames, apply log normalization
    for (sd::LongType b = 0; b < batchSize; b++) {
        // Temporary buffer for raw mel spectrogram
        std::vector<T> rawMel(numMelBins * rawNumFrames, 0);

        for (sd::LongType f = 0; f < rawNumFrames; f++) {
            sd::LongType frameStart = f * hopLength;
            std::vector<T> powerSpec(numFreqBins);

            for (int k = 0; k < numFreqBins; k++) {
                T sumReal = 0, sumImag = 0;
                for (int n = 0; n < fftSize; n++) {
                    T val = inputPtr[b * numSamples + frameStart + n] * window[n];
                    T angle = static_cast<T>(-2.0 * M_PI * k * n / fftSize);
                    sumReal += val * std::cos(angle);
                    sumImag += val * std::sin(angle);
                }
                T mag = std::sqrt(sumReal * sumReal + sumImag * sumImag);
                powerSpec[k] = mag * mag;  // power=2.0
            }

            for (int m = 0; m < numMelBins; m++) {
                T sum = 0;
                for (int k = 0; k < numFreqBins; k++)
                    sum += melFb[m * numFreqBins + k] * powerSpec[k];
                rawMel[m * rawNumFrames + f] = sum;
            }
        }

        T* dstBatch = outputPtr + b * numMelBins * targetFrames;
        sd::LongType framesToCopy = std::min(rawNumFrames, static_cast<sd::LongType>(targetFrames));

        // Pad/trim to targetFrames
        for (int m = 0; m < numMelBins; m++) {
            for (sd::LongType f = 0; f < framesToCopy; f++) {
                dstBatch[m * targetFrames + f] = rawMel[m * rawNumFrames + f];
            }
            for (sd::LongType f = framesToCopy; f < targetFrames; f++) {
                dstBatch[m * targetFrames + f] = static_cast<T>(0);
            }
        }

        // Whisper log normalization: log10(max(x, 1e-10)), clamp to max-8, (x+4)/4
        sd::LongType totalElements = numMelBins * targetFrames;
        T globalMax = static_cast<T>(-1e30);
        for (sd::LongType i = 0; i < totalElements; i++) {
            T val = dstBatch[i];
            val = val > static_cast<T>(1e-10) ? val : static_cast<T>(1e-10);
            val = static_cast<T>(std::log10(static_cast<double>(val)));
            dstBatch[i] = val;
            if (val > globalMax) globalMax = val;
        }

        T clampMin = globalMax - static_cast<T>(8.0);
        for (sd::LongType i = 0; i < totalElements; i++) {
            if (dstBatch[i] < clampMin) dstBatch[i] = clampMin;
        }

        for (sd::LongType i = 0; i < totalElements; i++) {
            dstBatch[i] = (dstBatch[i] + static_cast<T>(4.0)) / static_cast<T>(4.0);
        }
    }

    output->syncToDevice();
}

void whisperMelSpectrogram(LaunchContext* context, NDArray* input,
                            int sampleRate, int fftSize, int hopLength,
                            int numMelBins, int targetFrames,
                            double lowerEdgeHz, double upperEdgeHz,
                            NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), whisperMelSpectrogram_,
                          (context, input, sampleRate, fftSize, hopLength,
                           numMelBins, targetFrames, lowerEdgeHz, upperEdgeHz, output),
                          SD_FLOAT_TYPES);
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
