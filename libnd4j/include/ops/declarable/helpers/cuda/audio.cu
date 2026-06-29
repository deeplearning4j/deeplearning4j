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
#include <math/templatemath.h>
#include <system/op_boilerplate.h>
#include <execution/cuda/LaunchDims.h>
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
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (total + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "melFilterbank");

    melFilterbankKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
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
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (total + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "preEmphasis");

    preEmphasisKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
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

    dim3 launchDims = getLaunchDims("audio");
    const int sharedMemSize = launchDims.y * sizeof(T);

    PointersManager manager(context, "audioNormalize");

    // grid = batchSize (one block per batch element, data-driven), block+shmem from launchDims
    audioNormalizeKernel<T><<<batchSize, launchDims.y, sharedMemSize, *context->getCudaStream()>>>(
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
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (length + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "aWeighting");

    aWeightingKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
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
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (total + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "zeroCrossingRate");

    zeroCrossingRateKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
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
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (total + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "spectralCentroid");

    spectralCentroidKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
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
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (total + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "spectralRolloff");

    spectralRolloffKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
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
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (total + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "chromaFeatures");

    chromaFeaturesKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
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

// ======================== Mel Spectrogram ========================
// Fused CUDA kernel: each thread computes one output element (b, m, f).
// DFT power spectrum with Hann window + mel filterbank dot product, all on device.

template <typename T>
SD_KERNEL void melSpectrogramKernelFull(const T* input, T* output,
                                         sd::LongType batchSize, sd::LongType numSamples,
                                         int fftSize, int hopLength,
                                         int numMelBins, int numFreqBins,
                                         sd::LongType numFrames, int sampleRate,
                                         double lowerMel, double upperMel,
                                         double power) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numMelBins * numFrames;
    if (idx >= total) return;

    const auto b = idx / (numMelBins * numFrames);
    const auto mf = idx % (numMelBins * numFrames);
    const auto m = mf / numFrames;
    const auto f = mf % numFrames;

    const sd::LongType frameStart = f * hopLength;

    double melStep = (upperMel - lowerMel) / (numMelBins + 1);
    double fLeft   = d_melToHz(lowerMel + m * melStep) * fftSize / sampleRate;
    double fCenter = d_melToHz(lowerMel + (m + 1) * melStep) * fftSize / sampleRate;
    double fRight  = d_melToHz(lowerMel + (m + 2) * melStep) * fftSize / sampleRate;

    T melSum = 0;
    for (int k = 0; k < numFreqBins; k++) {
        T sumReal = 0, sumImag = 0;
        for (int n = 0; n < fftSize; n++) {
            T hann = static_cast<T>(0.5 * (1.0 - cos(2.0 * M_PI * n / fftSize)));
            T val = input[b * numSamples + frameStart + n] * hann;
            double angle = -2.0 * M_PI * k * n / fftSize;
            sumReal += val * static_cast<T>(cos(angle));
            sumImag += val * static_cast<T>(sin(angle));
        }
        T mag = sd::math::sd_sqrt<T, T>(sumReal * sumReal + sumImag * sumImag);
        T powerVal = (power == 2.0) ? mag * mag : static_cast<T>(::pow(static_cast<double>(mag), power));

        double freq = static_cast<double>(k);
        T weight = 0;
        if (freq >= fLeft && freq <= fCenter && fCenter > fLeft)
            weight = static_cast<T>((freq - fLeft) / (fCenter - fLeft));
        else if (freq > fCenter && freq <= fRight && fRight > fCenter)
            weight = static_cast<T>((fRight - freq) / (fRight - fCenter));

        melSum += weight * powerVal;
    }

    output[b * numMelBins * numFrames + m * numFrames + f] = melSum;
}

template <typename T>
static void melSpectrogram_(LaunchContext* context, NDArray* input,
                             int sampleRate, int fftSize, int hopLength, int numMelBins,
                             double lowerEdgeHz, double upperEdgeHz, double power,
                             NDArray* output) {
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const int numFreqBins = fftSize / 2 + 1;
    const auto numFrames = (numSamples - fftSize) / hopLength + 1;

    double lowerMel = 2595.0 * ::log10(1.0 + lowerEdgeHz / 700.0);
    double upperMel = 2595.0 * ::log10(1.0 + upperEdgeHz / 700.0);

    const auto total = batchSize * numMelBins * numFrames;
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (total + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "melSpectrogram");

    melSpectrogramKernelFull<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, numSamples, fftSize, hopLength,
        numMelBins, numFreqBins, numFrames, sampleRate,
        lowerMel, upperMel, power);

    manager.synchronize();
}

void melSpectrogram(LaunchContext* context, NDArray* input,
                     int sampleRate, int fftSize, int hopLength, int numMelBins,
                     double lowerEdgeHz, double upperEdgeHz, double power,
                     NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), melSpectrogram_,
                          (context, input, sampleRate, fftSize, hopLength, numMelBins,
                           lowerEdgeHz, upperEdgeHz, power, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== Log-Mel kernel (shared by mfcc and whisperMelSpectrogram) ========================

template <typename T>
SD_KERNEL void logMelSpectrogramKernel(const T* input, T* melOut,
                                        sd::LongType batchSize, sd::LongType numSamples,
                                        int fftSize, int hopLength,
                                        int numMelBins, int numFreqBins,
                                        sd::LongType numFrames, int sampleRate,
                                        double lowerMel, double upperMel) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numMelBins * numFrames;
    if (idx >= total) return;

    const auto b = idx / (numMelBins * numFrames);
    const auto mf = idx % (numMelBins * numFrames);
    const auto m = mf / numFrames;
    const auto f = mf % numFrames;

    const sd::LongType frameStart = f * hopLength;

    double melStep = (upperMel - lowerMel) / (numMelBins + 1);
    double fLeft   = d_melToHz(lowerMel + m * melStep) * fftSize / sampleRate;
    double fCenter = d_melToHz(lowerMel + (m + 1) * melStep) * fftSize / sampleRate;
    double fRight  = d_melToHz(lowerMel + (m + 2) * melStep) * fftSize / sampleRate;

    T melSum = 0;
    for (int k = 0; k < numFreqBins; k++) {
        T sumReal = 0, sumImag = 0;
        for (int n = 0; n < fftSize; n++) {
            T hann = static_cast<T>(0.5 * (1.0 - cos(2.0 * M_PI * n / fftSize)));
            T val = input[b * numSamples + frameStart + n] * hann;
            double angle = -2.0 * M_PI * k * n / fftSize;
            sumReal += val * static_cast<T>(cos(angle));
            sumImag += val * static_cast<T>(sin(angle));
        }
        T power = sumReal * sumReal + sumImag * sumImag;  // power=2

        double freq = static_cast<double>(k);
        T weight = 0;
        if (freq >= fLeft && freq <= fCenter && fCenter > fLeft)
            weight = static_cast<T>((freq - fLeft) / (fCenter - fLeft));
        else if (freq > fCenter && freq <= fRight && fRight > fCenter)
            weight = static_cast<T>((fRight - freq) / (fRight - fCenter));

        melSum += weight * power;
    }

    T logMel = sd::math::sd_log<T, T>(sd::math::sd_max<T>(melSum, static_cast<T>(1e-10)));
    melOut[b * numMelBins * numFrames + m * numFrames + f] = logMel;
}

// ======================== MFCC ========================
// Two kernels: kernel 1 → log-mel temp buffer, kernel 2 → DCT-II

template <typename T>
SD_KERNEL void dctKernel(const T* logMel, T* mfccOut,
                          sd::LongType batchSize, int numMelBins, int numMfcc,
                          sd::LongType numFrames) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numMfcc * numFrames;
    if (idx >= total) return;

    const auto b = idx / (numMfcc * numFrames);
    const auto cf = idx % (numMfcc * numFrames);
    const auto c = cf / numFrames;
    const auto f = cf % numFrames;

    T sum = 0;
    for (int mm = 0; mm < numMelBins; mm++) {
        T lm = logMel[b * numMelBins * numFrames + mm * numFrames + f];
        sum += lm * static_cast<T>(cos(M_PI * c * (mm + 0.5) / numMelBins));
    }
    mfccOut[b * numMfcc * numFrames + c * numFrames + f] = sum;
}

template <typename T>
static void mfcc_(LaunchContext* context, NDArray* input,
                   int sampleRate, int fftSize, int hopLength,
                   int numMelBins, int numMfcc,
                   double lowerEdgeHz, double upperEdgeHz,
                   NDArray* output) {
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const int numFreqBins = fftSize / 2 + 1;
    const auto numFrames = (numSamples - fftSize) / hopLength + 1;

    double lowerMel = 2595.0 * ::log10(1.0 + lowerEdgeHz / 700.0);
    double upperMel = 2595.0 * ::log10(1.0 + upperEdgeHz / 700.0);

    // Temp buffer for log-mel [batchSize, numMelBins, numFrames]
    std::vector<sd::LongType> melShape = {batchSize, numMelBins, numFrames};
    NDArray melTemp('c', melShape, input->dataType(), context);

    // Kernel 1: DFT + mel filterbank + log → melTemp
    // input is already prepared by the outer public wrapper; melTemp is a fresh device array.
    NDArray::prepareSpecialUse({&melTemp}, {input});

    const auto melTotal = batchSize * numMelBins * numFrames;
    dim3 launchDims = getLaunchDims("audio");
    int numBlocks = (melTotal + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "mfcc");

    logMelSpectrogramKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(melTemp.specialBuffer()),
        batchSize, numSamples, fftSize, hopLength,
        numMelBins, numFreqBins, numFrames, sampleRate,
        lowerMel, upperMel);

    manager.synchronize();
    NDArray::registerSpecialUse({&melTemp}, {input});

    // Kernel 2: DCT-II → output
    NDArray::prepareSpecialUse({output}, {&melTemp});

    const auto mfccTotal = batchSize * numMfcc * numFrames;
    numBlocks = (mfccTotal + launchDims.y - 1) / launchDims.y;

    PointersManager manager2(context, "mfcc_dct");

    dctKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(melTemp.specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, numMelBins, numMfcc, numFrames);

    manager2.synchronize();
    NDArray::registerSpecialUse({output}, {&melTemp});
}

void mfcc(LaunchContext* context, NDArray* input,
           int sampleRate, int fftSize, int hopLength,
           int numMelBins, int numMfcc,
           double lowerEdgeHz, double upperEdgeHz,
           NDArray* output) {
    // Note: inner mfcc_ manages per-kernel prepare/register for its temp buffer.
    // The outer prepare/register here covers the public NDArray tracking contract.
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), mfcc_,
                          (context, input, sampleRate, fftSize, hopLength, numMelBins, numMfcc,
                           lowerEdgeHz, upperEdgeHz, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== Griffin-Lim ========================
// Iterative ISTFT on device.
// Kernel 1: istftKernel — IDFT overlap-add for one iteration
// Kernel 2: stftPhaseKernel — extract phase from DFT of reconstructed signal
// Kernel 3: windowSumNormalizeKernel — divide signal by window sum
// Kernel 4: copyOutputKernel — copy final signal to output

template <typename T>
SD_KERNEL void istftKernel(const T* magnitude, const T* phase,
                            T* signal, T* windowSum,
                            sd::LongType batchSize, sd::LongType numFreqBins,
                            sd::LongType numFrames, sd::LongType numSamples,
                            int fftSize, int hopLength) {
    // Each thread handles one (b, frameStart+n) sample position contribution from one frame
    // Grid: batchSize * numFrames * fftSize threads
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numFrames * fftSize;
    if (idx >= total) return;

    const auto b  = idx / (numFrames * fftSize);
    const auto fn = idx % (numFrames * fftSize);
    const auto f  = fn / fftSize;
    const auto n  = fn % fftSize;

    const sd::LongType sampleIdx = f * hopLength + n;
    if (sampleIdx >= numSamples) return;

    double hann = 0.5 * (1.0 - cos(2.0 * M_PI * n / fftSize));

    double sampleVal = 0;
    for (sd::LongType k = 0; k < numFreqBins; k++) {
        double mag = static_cast<double>(magnitude[b * numFreqBins * numFrames + k * numFrames + f]);
        double ph  = static_cast<double>(phase[b * numFreqBins * numFrames + k * numFrames + f]);
        double angle = 2.0 * M_PI * k * n / fftSize;
        sampleVal += mag * cos(angle + ph);
    }
    sampleVal = hann * sampleVal / fftSize;

    // Atomic add because multiple frames overlap the same sample
    sd::math::atomics::sd_atomicAdd<T>(&signal[b * numSamples + sampleIdx], static_cast<T>(sampleVal));
    sd::math::atomics::sd_atomicAdd<T>(&windowSum[b * numSamples + sampleIdx], static_cast<T>(hann * hann));
}

template <typename T>
SD_KERNEL void windowSumNormalizeKernel(T* signal, const T* windowSum,
                                         sd::LongType batchSize, sd::LongType numSamples) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batchSize * numSamples) return;
    T ws = windowSum[idx];
    if (ws > static_cast<T>(1e-8))
        signal[idx] /= ws;
}

template <typename T>
SD_KERNEL void stftPhaseKernel(const T* signal, T* phase,
                                sd::LongType batchSize, sd::LongType numFreqBins,
                                sd::LongType numFrames, sd::LongType numSamples,
                                int fftSize, int hopLength) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numFreqBins * numFrames;
    if (idx >= total) return;

    const auto b  = idx / (numFreqBins * numFrames);
    const auto kf = idx % (numFreqBins * numFrames);
    const auto k  = kf / numFrames;
    const auto f  = kf % numFrames;

    const sd::LongType frameStart = f * hopLength;

    double sumReal = 0, sumImag = 0;
    for (int n = 0; n < fftSize; n++) {
        sd::LongType si = frameStart + n;
        if (si >= numSamples) continue;
        double hann = 0.5 * (1.0 - cos(2.0 * M_PI * n / fftSize));
        double val = static_cast<double>(signal[b * numSamples + si]) * hann;
        double angle = -2.0 * M_PI * k * n / fftSize;
        sumReal += val * cos(angle);
        sumImag += val * sin(angle);
    }

    phase[b * numFreqBins * numFrames + k * numFrames + f] = static_cast<T>(::atan2(sumImag, sumReal));
}

template <typename T>
SD_KERNEL void zeroBufferKernel(T* buf, sd::LongType n) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) buf[idx] = static_cast<T>(0);
}

template <typename T>
SD_KERNEL void copySignalToOutputKernel(const T* signal, T* output,
                                         sd::LongType batchSize, sd::LongType numSamples) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < batchSize * numSamples)
        output[idx] = signal[idx];
}

template <typename T>
static void griffinLim_(LaunchContext* context, NDArray* magnitudeSpectrogram,
                         int fftSize, int hopLength, int numIterations,
                         NDArray* output) {
    const auto batchSize   = magnitudeSpectrogram->sizeAt(0);
    const auto numFreqBins = magnitudeSpectrogram->sizeAt(1);
    const auto numFrames   = magnitudeSpectrogram->sizeAt(2);
    const auto numSamples  = (numFrames - 1) * hopLength + fftSize;

    // Temp buffers: signal, windowSum, phase — all device-side NDArrays
    std::vector<sd::LongType> sigShape   = {batchSize, numSamples};
    std::vector<sd::LongType> phaseShape = {batchSize, numFreqBins, numFrames};

    NDArray sigArr('c', sigShape,   magnitudeSpectrogram->dataType(), context);
    NDArray winArr('c', sigShape,   magnitudeSpectrogram->dataType(), context);
    NDArray phaseArr('c', phaseShape, magnitudeSpectrogram->dataType(), context);

    NDArray::prepareSpecialUse({&sigArr, &winArr, &phaseArr}, {});
    NDArray::registerSpecialUse({&sigArr, &winArr, &phaseArr}, {});

    dim3 launchDims = getLaunchDims("audio");

    // Zero-initialise phase buffer (already zero from NDArray ctor, but explicit is safe)
    {
        sd::LongType phaseN = batchSize * numFreqBins * numFrames;
        int nb = (phaseN + launchDims.y - 1) / launchDims.y;
        PointersManager mgr(context, "griffinLim_initPhase");
        zeroBufferKernel<T><<<nb, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
            reinterpret_cast<T*>(phaseArr.specialBuffer()), phaseN);
        mgr.synchronize();
    }

    for (int iter = 0; iter < numIterations; iter++) {
        // Zero signal and windowSum
        {
            sd::LongType sigN = batchSize * numSamples;
            int nb = (sigN + launchDims.y - 1) / launchDims.y;
            PointersManager mgr(context, "griffinLim_zeroSig");
            zeroBufferKernel<T><<<nb, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
                reinterpret_cast<T*>(sigArr.specialBuffer()), sigN);
            zeroBufferKernel<T><<<nb, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
                reinterpret_cast<T*>(winArr.specialBuffer()), sigN);
            mgr.synchronize();
        }

        // ISTFT overlap-add
        {
            sd::LongType istftTotal = batchSize * numFrames * fftSize;
            int nb = (istftTotal + launchDims.y - 1) / launchDims.y;
            PointersManager mgr(context, "griffinLim_istft");
            NDArray::prepareSpecialUse({&sigArr, &winArr}, {magnitudeSpectrogram, &phaseArr});
            istftKernel<T><<<nb, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
                reinterpret_cast<const T*>(magnitudeSpectrogram->specialBuffer()),
                reinterpret_cast<const T*>(phaseArr.specialBuffer()),
                reinterpret_cast<T*>(sigArr.specialBuffer()),
                reinterpret_cast<T*>(winArr.specialBuffer()),
                batchSize, numFreqBins, numFrames, numSamples, fftSize, hopLength);
            mgr.synchronize();
            NDArray::registerSpecialUse({&sigArr, &winArr}, {magnitudeSpectrogram, &phaseArr});
        }

        // Normalize by window sum
        {
            sd::LongType sigN = batchSize * numSamples;
            int nb = (sigN + launchDims.y - 1) / launchDims.y;
            PointersManager mgr(context, "griffinLim_norm");
            NDArray::prepareSpecialUse({&sigArr}, {&winArr});
            windowSumNormalizeKernel<T><<<nb, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
                reinterpret_cast<T*>(sigArr.specialBuffer()),
                reinterpret_cast<const T*>(winArr.specialBuffer()),
                batchSize, numSamples);
            mgr.synchronize();
            NDArray::registerSpecialUse({&sigArr}, {&winArr});
        }

        // Update phase (skip on last iteration — output is signal)
        if (iter < numIterations - 1) {
            sd::LongType phaseTotal = batchSize * numFreqBins * numFrames;
            int nb = (phaseTotal + launchDims.y - 1) / launchDims.y;
            PointersManager mgr(context, "griffinLim_phase");
            NDArray::prepareSpecialUse({&phaseArr}, {&sigArr});
            stftPhaseKernel<T><<<nb, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
                reinterpret_cast<const T*>(sigArr.specialBuffer()),
                reinterpret_cast<T*>(phaseArr.specialBuffer()),
                batchSize, numFreqBins, numFrames, numSamples, fftSize, hopLength);
            mgr.synchronize();
            NDArray::registerSpecialUse({&phaseArr}, {&sigArr});
        }
    }

    // Copy signal to output
    {
        sd::LongType sigN = batchSize * numSamples;
        int nb = (sigN + launchDims.y - 1) / launchDims.y;
        PointersManager mgr(context, "griffinLim_copy");
        NDArray::prepareSpecialUse({output}, {&sigArr});
        copySignalToOutputKernel<T><<<nb, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
            reinterpret_cast<const T*>(sigArr.specialBuffer()),
            reinterpret_cast<T*>(output->specialBuffer()),
            batchSize, numSamples);
        mgr.synchronize();
        NDArray::registerSpecialUse({output}, {&sigArr});
    }
}

void griffinLim(LaunchContext* context, NDArray* magnitudeSpectrogram,
                 int fftSize, int hopLength, int numIterations,
                 NDArray* output) {
    NDArray::prepareSpecialUse({output}, {magnitudeSpectrogram});
    BUILD_SINGLE_SELECTOR(magnitudeSpectrogram->dataType(), griffinLim_,
                          (context, magnitudeSpectrogram, fftSize, hopLength, numIterations, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {magnitudeSpectrogram});
}

// ======================== Pitch Detection ========================
// One thread per (b, frame): autocorrelation over lag range

template <typename T>
SD_KERNEL void pitchDetectionKernel(const T* input, T* output,
                                     sd::LongType batchSize, sd::LongType numSamples,
                                     sd::LongType numFrames, int frameLength,
                                     int hopLength, int minLag, int maxLag,
                                     int sampleRate) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * numFrames;
    if (idx >= total) return;

    const auto b = idx / numFrames;
    const auto f = idx % numFrames;
    const sd::LongType frameStart = f * hopLength;

    T maxCorr = 0;
    int bestLag = 0;

    for (int lag = minLag; lag <= maxLag && lag < frameLength; lag++) {
        T corr = 0, normA = 0, normB = 0;
        for (int n = 0; n < frameLength - lag; n++) {
            T a  = input[b * numSamples + frameStart + n];
            T bv = input[b * numSamples + frameStart + n + lag];
            corr  += a * bv;
            normA += a * a;
            normB += bv * bv;
        }
        T norm = sd::math::sd_sqrt<T, T>(normA * normB);
        T normalizedCorr = (norm > static_cast<T>(1e-10)) ? corr / norm : static_cast<T>(0);
        if (normalizedCorr > maxCorr) {
            maxCorr = normalizedCorr;
            bestLag = lag;
        }
    }

    output[idx] = (bestLag > 0 && maxCorr > static_cast<T>(0.2))
        ? static_cast<T>(sampleRate) / static_cast<T>(bestLag)
        : static_cast<T>(0);
}

template <typename T>
static void pitchDetection_(LaunchContext* context, NDArray* input,
                             int sampleRate, int frameLength, int hopLength,
                             double minFreq, double maxFreq,
                             NDArray* output) {
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const auto numFrames = (numSamples - frameLength) / hopLength + 1;

    const int minLag = static_cast<int>(sampleRate / maxFreq);
    const int maxLag = static_cast<int>(sampleRate / minFreq);

    const auto total = batchSize * numFrames;
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (total + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "pitchDetection");

    pitchDetectionKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, numSamples, numFrames, frameLength, hopLength,
        minLag, maxLag, sampleRate);

    manager.synchronize();
}

void pitchDetection(LaunchContext* context, NDArray* input,
                     int sampleRate, int frameLength, int hopLength,
                     double minFreq, double maxFreq,
                     NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), pitchDetection_,
                          (context, input, sampleRate, frameLength, hopLength, minFreq, maxFreq, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== Audio Resample ========================
// One thread per (b, output_sample): sinc interpolation with Blackman-like window

template <typename T>
SD_KERNEL void resampleKernel(const T* input, T* output,
                               sd::LongType batchSize,
                               sd::LongType origSamples, sd::LongType targetSamples,
                               double ratio, int sincRadius) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * targetSamples;
    if (idx >= total) return;

    const auto b = idx / targetSamples;
    const auto i = idx % targetSamples;

    double srcPos = i * ratio;
    int srcCenter = static_cast<int>(srcPos);
    double sum       = 0;
    double weightSum = 0;

    for (int j = srcCenter - sincRadius + 1; j <= srcCenter + sincRadius; j++) {
        if (j >= 0 && j < origSamples) {
            double x = srcPos - j;
            double weight;
            if (x < 1e-10 && x > -1e-10) {
                weight = 1.0;
            } else if (x < sincRadius && x > -sincRadius) {
                double pix     = M_PI * x;
                double pixOverA = M_PI * x / sincRadius;
                weight = (sin(pix) / pix) * (sin(pixOverA) / pixOverA);
            } else {
                weight = 0.0;
            }
            sum       += static_cast<double>(input[b * origSamples + j]) * weight;
            weightSum += weight;
        }
    }

    output[idx] = (weightSum > 1e-10)
        ? static_cast<T>(sum / weightSum)
        : static_cast<T>(0);
}

template <typename T>
static void audioResample_(LaunchContext* context, NDArray* input,
                            int origSampleRate, int targetSampleRate,
                            NDArray* output) {
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize    = hasBatch ? input->sizeAt(0)  : 1;
    const auto origSamples  = hasBatch ? input->sizeAt(1)  : input->sizeAt(0);
    const auto targetSamples = hasBatch ? output->sizeAt(1) : output->sizeAt(0);

    const double ratio = static_cast<double>(origSampleRate) / static_cast<double>(targetSampleRate);
    const int sincRadius = 8;

    const auto total = batchSize * targetSamples;
    dim3 launchDims = getLaunchDims("audio");
    const int numBlocks = (total + launchDims.y - 1) / launchDims.y;

    PointersManager manager(context, "audioResample");

    resampleKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
        reinterpret_cast<const T*>(input->specialBuffer()),
        reinterpret_cast<T*>(output->specialBuffer()),
        batchSize, origSamples, targetSamples, ratio, sincRadius);

    manager.synchronize();
}

void audioResample(LaunchContext* context, NDArray* input,
                    int origSampleRate, int targetSampleRate,
                    NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), audioResample_,
                          (context, input, origSampleRate, targetSampleRate, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

// ======================== Whisper Mel Spectrogram ========================
// Two kernels:
//   Kernel 1: compute mel spectrogram (power=2) with pad/trim to targetFrames → device temp
//             (reuses melSpectrogramKernelFull; we write directly into output and zero padding)
//   Kernel 2: per-batch log10 normalization (global-max reduction → clamp → scale)

// Kernel: apply log10(max(x,1e-10)) in-place
template <typename T>
SD_KERNEL void log10InPlaceKernel(T* buf, sd::LongType n) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    T v = buf[idx];
    if (v < static_cast<T>(1e-10)) v = static_cast<T>(1e-10);
    buf[idx] = static_cast<T>(::log10(static_cast<double>(v)));
}

// Kernel: per-batch global-max reduction (one block per batch element, shared mem)
template <typename T>
SD_KERNEL void batchMaxKernel(const T* buf, T* batchMax,
                               sd::LongType batchSize, sd::LongType elemsPerBatch) {
    extern __shared__ char shmem[];
    T* shMax = reinterpret_cast<T*>(shmem);

    const auto b = blockIdx.x;
    if (b >= batchSize) return;

    const T* bPtr = buf + b * elemsPerBatch;
    T localMax = static_cast<T>(-1e30);
    for (sd::LongType i = threadIdx.x; i < elemsPerBatch; i += blockDim.x) {
        T v = bPtr[i];
        if (v > localMax) localMax = v;
    }
    shMax[threadIdx.x] = localMax;
    __syncthreads();

    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (shMax[threadIdx.x + stride] > shMax[threadIdx.x])
                shMax[threadIdx.x] = shMax[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) batchMax[b] = shMax[0];
}

// Kernel: clamp to (globalMax - 8) then (x+4)/4 in-place
template <typename T>
SD_KERNEL void whisperNormalizeKernel(T* buf, const T* batchMax,
                                       sd::LongType batchSize, sd::LongType elemsPerBatch) {
    const auto idx = blockIdx.x * blockDim.x + threadIdx.x;
    const auto total = batchSize * elemsPerBatch;
    if (idx >= total) return;

    const auto b = idx / elemsPerBatch;
    T clampMin = batchMax[b] - static_cast<T>(8.0);
    T v = buf[idx];
    if (v < clampMin) v = clampMin;
    buf[idx] = (v + static_cast<T>(4.0)) / static_cast<T>(4.0);
}

template <typename T>
static void whisperMelSpectrogram_(LaunchContext* context, NDArray* input,
                                    int sampleRate, int fftSize, int hopLength,
                                    int numMelBins, int targetFrames,
                                    double lowerEdgeHz, double upperEdgeHz,
                                    NDArray* output) {
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const int numFreqBins = fftSize / 2 + 1;
    const auto rawNumFrames = (numSamples - fftSize) / hopLength + 1;

    double lowerMel = 2595.0 * ::log10(1.0 + lowerEdgeHz / 700.0);
    double upperMel = 2595.0 * ::log10(1.0 + upperEdgeHz / 700.0);

    dim3 launchDims = getLaunchDims("audio");

    // Kernel 1: compute mel spectrogram (power=2) into a temp buffer [batchSize, numMelBins, rawNumFrames],
    // then D2D copy with pad/trim into output [batchSize, numMelBins, targetFrames].
    {
        const auto rawTotal = batchSize * numMelBins * rawNumFrames;
        int numBlocks = (rawTotal + launchDims.y - 1) / launchDims.y;

        std::vector<sd::LongType> tmpShape = {batchSize, numMelBins, rawNumFrames};
        NDArray tmpMel('c', tmpShape, input->dataType(), context);
        NDArray::prepareSpecialUse({&tmpMel}, {input});

        PointersManager mgr(context, "whisperMel_spec");
        melSpectrogramKernelFull<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
            reinterpret_cast<const T*>(input->specialBuffer()),
            reinterpret_cast<T*>(tmpMel.specialBuffer()),
            batchSize, numSamples, fftSize, hopLength,
            numMelBins, numFreqBins, rawNumFrames, sampleRate,
            lowerMel, upperMel, 2.0);
        mgr.synchronize();
        NDArray::registerSpecialUse({&tmpMel}, {input});

        // Pad/trim: copy each (b, m) stripe via D2D cudaMemcpyAsync on context stream
        NDArray::prepareSpecialUse({output}, {&tmpMel});
        const sd::LongType framesToCopy = (rawNumFrames < targetFrames) ? rawNumFrames : targetFrames;
        const size_t elemSz = sizeof(T);
        T* tmpPtr = reinterpret_cast<T*>(tmpMel.specialBuffer());
        T* outPtr = reinterpret_cast<T*>(output->specialBuffer());
        for (sd::LongType b = 0; b < batchSize; b++) {
            for (int m = 0; m < numMelBins; m++) {
                T* src = tmpPtr + b * numMelBins * rawNumFrames + m * rawNumFrames;
                T* dst = outPtr + b * numMelBins * targetFrames + m * targetFrames;
                cudaMemcpyAsync(dst, src, framesToCopy * elemSz,
                                cudaMemcpyDeviceToDevice, *context->getCudaStream());
                if (framesToCopy < targetFrames)
                    cudaMemsetAsync(dst + framesToCopy, 0,
                                   (targetFrames - framesToCopy) * elemSz,
                                   *context->getCudaStream());
            }
        }
        PointersManager syncMgr(context, "whisperMel_padtrim");
        syncMgr.synchronize();
        NDArray::registerSpecialUse({output}, {&tmpMel});
    }

    // Kernel 2: log10 in-place on output
    {
        const sd::LongType outTotal = batchSize * numMelBins * targetFrames;
        int numBlocks = (outTotal + launchDims.y - 1) / launchDims.y;
        PointersManager mgr(context, "whisperMel_log10");
        NDArray::prepareSpecialUse({output}, {});
        log10InPlaceKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
            reinterpret_cast<T*>(output->specialBuffer()), outTotal);
        mgr.synchronize();
        NDArray::registerSpecialUse({output}, {});
    }

    // Kernel 3: per-batch global max reduction
    const sd::LongType elemsPerBatch = numMelBins * targetFrames;
    std::vector<sd::LongType> maxShape = {batchSize};
    NDArray batchMaxArr('c', maxShape, input->dataType(), context);
    {
        NDArray::prepareSpecialUse({&batchMaxArr}, {output});
        int sharedSz = launchDims.y * sizeof(T);
        PointersManager mgr(context, "whisperMel_max");
        // grid = batchSize (one block per batch element, data-driven), block+shmem from launchDims
        batchMaxKernel<T><<<batchSize, launchDims.y, sharedSz, *context->getCudaStream()>>>(
            reinterpret_cast<const T*>(output->specialBuffer()),
            reinterpret_cast<T*>(batchMaxArr.specialBuffer()),
            batchSize, elemsPerBatch);
        mgr.synchronize();
        NDArray::registerSpecialUse({&batchMaxArr}, {output});
    }

    // Kernel 4: clamp + scale in-place
    {
        const sd::LongType outTotal = batchSize * elemsPerBatch;
        int numBlocks = (outTotal + launchDims.y - 1) / launchDims.y;
        PointersManager mgr(context, "whisperMel_normalize");
        NDArray::prepareSpecialUse({output}, {&batchMaxArr});
        whisperNormalizeKernel<T><<<numBlocks, launchDims.y, launchDims.z, *context->getCudaStream()>>>(
            reinterpret_cast<T*>(output->specialBuffer()),
            reinterpret_cast<const T*>(batchMaxArr.specialBuffer()),
            batchSize, elemsPerBatch);
        mgr.synchronize();
        NDArray::registerSpecialUse({output}, {&batchMaxArr});
    }
}

void whisperMelSpectrogram(LaunchContext* context, NDArray* input,
                            int sampleRate, int fftSize, int hopLength,
                            int numMelBins, int targetFrames,
                            double lowerEdgeHz, double upperEdgeHz,
                            NDArray* output) {
    NDArray::prepareSpecialUse({output}, {input});
    BUILD_SINGLE_SELECTOR(input->dataType(), whisperMelSpectrogram_,
                          (context, input, sampleRate, fftSize, hopLength,
                           numMelBins, targetFrames, lowerEdgeHz, upperEdgeHz, output),
                          SD_FLOAT_TYPES);
    NDArray::registerSpecialUse({output}, {input});
}

}  // namespace helpers
}  // namespace ops
}  // namespace sd
