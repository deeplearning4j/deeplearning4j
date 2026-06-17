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
#include <ops/declarable/helpers/audio.h>
#include <math/templatemath.h>
#include <cmath>
#include <algorithm>
#include <vector>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

namespace sd {
namespace ops {
namespace helpers {

// Convert frequency in Hz to mel scale
static double hzToMel(double hz) {
    return 2595.0 * std::log10(1.0 + hz / 700.0);
}

// Convert mel scale to frequency in Hz
static double melToHz(double mel) {
    return 700.0 * (std::pow(10.0, mel / 2595.0) - 1.0);
}

template <typename T>
static void melFilterbank_(LaunchContext* context, int numMelBins, int fftSize,
                            int sampleRate, double lowerEdgeHz, double upperEdgeHz,
                            NDArray* output) {
    auto outputPtr = output->bufferAsT<T>();
    const int numFreqBins = fftSize / 2 + 1;

    double lowerMel = hzToMel(lowerEdgeHz);
    double upperMel = hzToMel(upperEdgeHz);

    // Create evenly spaced mel points
    std::vector<double> melPoints(numMelBins + 2);
    for (int i = 0; i < numMelBins + 2; i++) {
        melPoints[i] = lowerMel + (upperMel - lowerMel) * i / (numMelBins + 1);
    }

    // Convert mel points to Hz then to FFT bin indices
    std::vector<double> hzPoints(numMelBins + 2);
    std::vector<double> binPoints(numMelBins + 2);
    for (int i = 0; i < numMelBins + 2; i++) {
        hzPoints[i] = melToHz(melPoints[i]);
        binPoints[i] = hzPoints[i] * fftSize / sampleRate;
    }

    auto func = PRAGMA_THREADS_FOR {
        for (auto m = start; m < stop; m++) {
            double fLeft = binPoints[m];
            double fCenter = binPoints[m + 1];
            double fRight = binPoints[m + 2];

            for (int k = 0; k < numFreqBins; k++) {
                double freq = static_cast<double>(k);
                T val = 0;

                if (freq >= fLeft && freq <= fCenter && fCenter > fLeft) {
                    val = static_cast<T>((freq - fLeft) / (fCenter - fLeft));
                } else if (freq > fCenter && freq <= fRight && fRight > fCenter) {
                    val = static_cast<T>((fRight - freq) / (fRight - fCenter));
                }

                outputPtr[m * numFreqBins + k] = val;
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, numMelBins, 1);
}

void melFilterbank(LaunchContext* context, int numMelBins, int fftSize,
                    int sampleRate, double lowerEdgeHz, double upperEdgeHz,
                    NDArray* output) {
    BUILD_SINGLE_SELECTOR(output->dataType(), melFilterbank_,
                          (context, numMelBins, fftSize, sampleRate, lowerEdgeHz, upperEdgeHz, output),
                          SD_FLOAT_TYPES);
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

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    // Build mel filterbank
    std::vector<T> melFb(numMelBins * numFreqBins, 0);
    {
        double lowerMel = hzToMel(lowerEdgeHz);
        double upperMel = hzToMel(upperEdgeHz);
        std::vector<double> melPoints(numMelBins + 2);
        for (int i = 0; i < numMelBins + 2; i++)
            melPoints[i] = lowerMel + (upperMel - lowerMel) * i / (numMelBins + 1);
        std::vector<double> binPoints(numMelBins + 2);
        for (int i = 0; i < numMelBins + 2; i++)
            binPoints[i] = melToHz(melPoints[i]) * fftSize / sampleRate;

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
    }

    // Hann window
    std::vector<T> window(fftSize);
    for (int i = 0; i < fftSize; i++)
        window[i] = static_cast<T>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / fftSize)));

    auto func = PRAGMA_THREADS_FOR_2D {
        std::vector<T> frame(fftSize);
        std::vector<T> powerSpec(numFreqBins);

        for (auto b = start_x; b < stop_x; b++) {
            for (auto f = start_y; f < stop_y; f++) {
                sd::LongType frameStart = f * hopLength;

                // Extract windowed frame
                for (int i = 0; i < fftSize; i++) {
                    T val = inputPtr[b * numSamples + frameStart + i];
                    frame[i] = val * window[i];
                }

                // Compute power spectrum via DFT
                for (int k = 0; k < numFreqBins; k++) {
                    T sumReal = 0, sumImag = 0;
                    for (int n = 0; n < fftSize; n++) {
                        T angle = static_cast<T>(-2.0 * M_PI * k * n / fftSize);
                        sumReal += frame[n] * sd::math::sd_cos<T>(angle);
                        sumImag += frame[n] * sd::math::sd_sin<T>(angle);
                    }
                    T mag = sd::math::sd_sqrt<T>(sumReal * sumReal + sumImag * sumImag);
                    powerSpec[k] = (power == 2.0) ? mag * mag : static_cast<T>(sd::math::sd_pow<T,T,T>(mag, static_cast<T>(power)));
                }

                // Apply mel filterbank
                for (int m = 0; m < numMelBins; m++) {
                    T sum = 0;
                    for (int k = 0; k < numFreqBins; k++) {
                        sum += melFb[m * numFreqBins + k] * powerSpec[k];
                    }
                    outputPtr[b * numMelBins * numFrames + m * numFrames + f] = sum;
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, numFrames, 1);
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
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const int numFreqBins = fftSize / 2 + 1;
    const auto numFrames = (numSamples - fftSize) / hopLength + 1;

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    // Build mel filterbank
    std::vector<T> melFb(numMelBins * numFreqBins, 0);
    {
        double lowerMel = hzToMel(lowerEdgeHz);
        double upperMel = hzToMel(upperEdgeHz);
        std::vector<double> melPoints(numMelBins + 2);
        for (int i = 0; i < numMelBins + 2; i++)
            melPoints[i] = lowerMel + (upperMel - lowerMel) * i / (numMelBins + 1);
        std::vector<double> binPoints(numMelBins + 2);
        for (int i = 0; i < numMelBins + 2; i++)
            binPoints[i] = melToHz(melPoints[i]) * fftSize / sampleRate;

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
    }

    // Hann window
    std::vector<T> window(fftSize);
    for (int i = 0; i < fftSize; i++)
        window[i] = static_cast<T>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / fftSize)));

    auto func = PRAGMA_THREADS_FOR_2D {
        std::vector<T> frame(fftSize);
        std::vector<T> powerSpec(numFreqBins);
        std::vector<T> melEnergies(numMelBins);

        for (auto b = start_x; b < stop_x; b++) {
            for (auto f = start_y; f < stop_y; f++) {
                sd::LongType frameStart = f * hopLength;

                // Extract windowed frame
                for (int i = 0; i < fftSize; i++) {
                    T val = inputPtr[b * numSamples + frameStart + i];
                    frame[i] = val * window[i];
                }

                // Compute power spectrum
                for (int k = 0; k < numFreqBins; k++) {
                    T sumReal = 0, sumImag = 0;
                    for (int n = 0; n < fftSize; n++) {
                        T angle = static_cast<T>(-2.0 * M_PI * k * n / fftSize);
                        sumReal += frame[n] * sd::math::sd_cos<T>(angle);
                        sumImag += frame[n] * sd::math::sd_sin<T>(angle);
                    }
                    powerSpec[k] = sumReal * sumReal + sumImag * sumImag;
                }

                // Apply mel filterbank + log
                for (int m = 0; m < numMelBins; m++) {
                    T sum = 0;
                    for (int k = 0; k < numFreqBins; k++) {
                        sum += melFb[m * numFreqBins + k] * powerSpec[k];
                    }
                    melEnergies[m] = sd::math::sd_log<T>(sd::math::sd_max<T>(sum, static_cast<T>(1e-10)));
                }

                // DCT-II to get MFCCs
                for (int c = 0; c < numMfcc; c++) {
                    T sum = 0;
                    for (int m = 0; m < numMelBins; m++) {
                        sum += melEnergies[m] * std::cos(M_PI * c * (m + 0.5) / numMelBins);
                    }
                    outputPtr[b * numMfcc * numFrames + c * numFrames + f] = sum;
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, numFrames, 1);
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
    const auto batchSize = magnitudeSpectrogram->sizeAt(0);
    const auto numFreqBins = magnitudeSpectrogram->sizeAt(1);
    const auto numFrames = magnitudeSpectrogram->sizeAt(2);
    const auto numSamples = (numFrames - 1) * hopLength + fftSize;

    auto magPtr = magnitudeSpectrogram->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    // Hann window
    std::vector<T> window(fftSize);
    for (int i = 0; i < fftSize; i++)
        window[i] = static_cast<T>(0.5 * (1.0 - std::cos(2.0 * M_PI * i / fftSize)));

    for (sd::LongType b = 0; b < batchSize; b++) {
        // Initialize with random phase (use magnitude as initial estimate)
        std::vector<T> phase(numFreqBins * numFrames, 0);
        std::vector<T> signal(numSamples, 0);
        std::vector<T> windowSum(numSamples, 0);

        for (int iter = 0; iter < numIterations; iter++) {
            // ISTFT: reconstruct signal from magnitude + estimated phase
            std::fill(signal.begin(), signal.end(), static_cast<T>(0));
            std::fill(windowSum.begin(), windowSum.end(), static_cast<T>(0));

            for (sd::LongType f = 0; f < numFrames; f++) {
                sd::LongType frameStart = f * hopLength;

                // Inverse DFT for this frame
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

            // Normalize by window sum
            for (sd::LongType i = 0; i < numSamples; i++) {
                if (windowSum[i] > static_cast<T>(1e-8))
                    signal[i] /= windowSum[i];
            }

            // STFT: compute phase from current signal estimate
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

        // Copy to output
        for (sd::LongType i = 0; i < numSamples; i++) {
            outputPtr[b * numSamples + i] = signal[i];
        }
    }
}

void griffinLim(LaunchContext* context, NDArray* magnitudeSpectrogram,
                 int fftSize, int hopLength, int numIterations,
                 NDArray* output) {
    BUILD_SINGLE_SELECTOR(magnitudeSpectrogram->dataType(), griffinLim_,
                          (context, magnitudeSpectrogram, fftSize, hopLength, numIterations, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void spectralCentroid_(LaunchContext* context, NDArray* input,
                               int sampleRate, int fftSize,
                               NDArray* output) {
    const auto batchSize = input->sizeAt(0);
    const auto numFreqBins = input->sizeAt(1);
    const auto numFrames = input->sizeAt(2);
    const T freqScale = static_cast<T>(sampleRate) / static_cast<T>(fftSize);

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto b = start_x; b < stop_x; b++) {
            for (auto f = start_y; f < stop_y; f++) {
                T weightedSum = 0;
                T magnitudeSum = 0;

                for (sd::LongType k = 0; k < numFreqBins; k++) {
                    T mag = inputPtr[b * numFreqBins * numFrames + k * numFrames + f];
                    T freq = static_cast<T>(k) * freqScale;
                    weightedSum += freq * mag;
                    magnitudeSum += mag;
                }

                outputPtr[b * numFrames + f] = (magnitudeSum > static_cast<T>(1e-10))
                    ? weightedSum / magnitudeSum : static_cast<T>(0);
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, numFrames, 1);
}

void spectralCentroid(LaunchContext* context, NDArray* input,
                       int sampleRate, int fftSize,
                       NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), spectralCentroid_,
                          (context, input, sampleRate, fftSize, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void spectralRolloff_(LaunchContext* context, NDArray* input,
                              int sampleRate, int fftSize, double rolloffPercent,
                              NDArray* output) {
    const auto batchSize = input->sizeAt(0);
    const auto numFreqBins = input->sizeAt(1);
    const auto numFrames = input->sizeAt(2);
    const T freqScale = static_cast<T>(sampleRate) / static_cast<T>(fftSize);
    const T threshold = static_cast<T>(rolloffPercent);

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto b = start_x; b < stop_x; b++) {
            for (auto f = start_y; f < stop_y; f++) {
                T totalEnergy = 0;
                for (sd::LongType k = 0; k < numFreqBins; k++) {
                    totalEnergy += inputPtr[b * numFreqBins * numFrames + k * numFrames + f];
                }

                T cumulativeEnergy = 0;
                T rolloffFreq = 0;
                for (sd::LongType k = 0; k < numFreqBins; k++) {
                    cumulativeEnergy += inputPtr[b * numFreqBins * numFrames + k * numFrames + f];
                    if (cumulativeEnergy >= threshold * totalEnergy) {
                        rolloffFreq = static_cast<T>(k) * freqScale;
                        break;
                    }
                }

                outputPtr[b * numFrames + f] = rolloffFreq;
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, numFrames, 1);
}

void spectralRolloff(LaunchContext* context, NDArray* input,
                      int sampleRate, int fftSize, double rolloffPercent,
                      NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), spectralRolloff_,
                          (context, input, sampleRate, fftSize, rolloffPercent, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void chromaFeatures_(LaunchContext* context, NDArray* input,
                             int sampleRate, int fftSize, int numChroma,
                             NDArray* output) {
    const auto batchSize = input->sizeAt(0);
    const auto numFreqBins = input->sizeAt(1);
    const auto numFrames = input->sizeAt(2);

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    // Precompute frequency-to-chroma mapping
    std::vector<int> chromaMap(numFreqBins);
    for (int k = 0; k < numFreqBins; k++) {
        double freqHz = static_cast<double>(k) * sampleRate / fftSize;
        if (freqHz > 0) {
            // Map frequency to chroma bin: 12 * log2(freq/refFreq) mod 12
            // Using A440 as reference: C4 = 261.63 Hz
            double semitone = 12.0 * std::log2(freqHz / 261.63);
            int chromaBin = static_cast<int>(std::fmod(semitone + 1200.0, 12.0));
            if (chromaBin < 0) chromaBin += numChroma;
            chromaMap[k] = chromaBin % numChroma;
        } else {
            chromaMap[k] = 0;
        }
    }

    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto b = start_x; b < stop_x; b++) {
            for (auto f = start_y; f < stop_y; f++) {
                // Zero output
                for (int c = 0; c < numChroma; c++) {
                    outputPtr[b * numChroma * numFrames + c * numFrames + f] = 0;
                }

                // Accumulate magnitudes into chroma bins
                for (sd::LongType k = 1; k < numFreqBins; k++) {
                    T mag = inputPtr[b * numFreqBins * numFrames + k * numFrames + f];
                    int c = chromaMap[k];
                    outputPtr[b * numChroma * numFrames + c * numFrames + f] += mag;
                }
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, numFrames, 1);
}

void chromaFeatures(LaunchContext* context, NDArray* input,
                     int sampleRate, int fftSize, int numChroma,
                     NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), chromaFeatures_,
                          (context, input, sampleRate, fftSize, numChroma, output),
                          SD_FLOAT_TYPES);
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

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto b = start_x; b < stop_x; b++) {
            for (auto f = start_y; f < stop_y; f++) {
                sd::LongType frameStart = f * hopLength;
                int crossings = 0;

                for (int i = 1; i < frameLength; i++) {
                    T curr = inputPtr[b * numSamples + frameStart + i];
                    T prev = inputPtr[b * numSamples + frameStart + i - 1];
                    if ((curr >= 0 && prev < 0) || (curr < 0 && prev >= 0)) {
                        crossings++;
                    }
                }

                outputPtr[b * numFrames + f] = static_cast<T>(crossings) / static_cast<T>(frameLength - 1);
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, numFrames, 1);
}

void zeroCrossingRate(LaunchContext* context, NDArray* input,
                       int frameLength, int hopLength,
                       NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), zeroCrossingRate_,
                          (context, input, frameLength, hopLength, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void preEmphasis_(LaunchContext* context, NDArray* input,
                          double coefficient, NDArray* output) {
    const auto length = input->lengthOf();
    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();
    const T coeff = static_cast<T>(coefficient);

    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);

    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; b++) {
            outputPtr[b * numSamples] = inputPtr[b * numSamples];
            for (sd::LongType i = 1; i < numSamples; i++) {
                outputPtr[b * numSamples + i] = inputPtr[b * numSamples + i] -
                    coeff * inputPtr[b * numSamples + i - 1];
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1);
}

void preEmphasis(LaunchContext* context, NDArray* input,
                  double coefficient, NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), preEmphasis_,
                          (context, input, coefficient, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void audioNormalize_(LaunchContext* context, NDArray* input,
                             double targetLevel, bool useRms,
                             NDArray* output) {
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();
    const T target = static_cast<T>(targetLevel);

    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; b++) {
            T currentLevel;

            if (useRms) {
                T sumSq = 0;
                for (sd::LongType i = 0; i < numSamples; i++) {
                    T val = inputPtr[b * numSamples + i];
                    sumSq += val * val;
                }
                currentLevel = std::sqrt(sumSq / static_cast<T>(numSamples));
            } else {
                currentLevel = 0;
                for (sd::LongType i = 0; i < numSamples; i++) {
                    T absVal = std::abs(inputPtr[b * numSamples + i]);
                    if (absVal > currentLevel) currentLevel = absVal;
                }
            }

            T scale = (currentLevel > static_cast<T>(1e-10)) ? target / currentLevel : static_cast<T>(0);

            for (sd::LongType i = 0; i < numSamples; i++) {
                outputPtr[b * numSamples + i] = inputPtr[b * numSamples + i] * scale;
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1);
}

void audioNormalize(LaunchContext* context, NDArray* input,
                     double targetLevel, bool useRms,
                     NDArray* output) {
    BUILD_SINGLE_SELECTOR(input->dataType(), audioNormalize_,
                          (context, input, targetLevel, useRms, output),
                          SD_FLOAT_TYPES);
}

template <typename T>
static void aWeighting_(LaunchContext* context, NDArray* frequencies,
                         NDArray* output) {
    auto freqPtr = frequencies->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();
    const auto length = frequencies->lengthOf();

    auto func = PRAGMA_THREADS_FOR {
        for (auto i = start; i < stop; i++) {
            double f = static_cast<double>(freqPtr[i]);
            double f2 = f * f;

            // A-weighting formula (IEC 61672)
            double num = 12194.0 * 12194.0 * f2 * f2;
            double den = (f2 + 20.6 * 20.6) *
                         std::sqrt((f2 + 107.7 * 107.7) * (f2 + 737.9 * 737.9)) *
                         (f2 + 12194.0 * 12194.0);

            double ra = (den > 1e-30) ? num / den : 0.0;
            // Convert to dB, with A-weighting reference offset
            double aWeight = (ra > 1e-30) ? 20.0 * std::log10(ra) + 2.0 : -100.0;

            outputPtr[i] = static_cast<T>(aWeight);
        }
    };

    samediff::Threads::parallel_for(func, 0, length, 1);
}

void aWeighting(LaunchContext* context, NDArray* frequencies,
                 NDArray* output) {
    BUILD_SINGLE_SELECTOR(frequencies->dataType(), aWeighting_,
                          (context, frequencies, output),
                          SD_FLOAT_TYPES);
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

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto b = start_x; b < stop_x; b++) {
            for (auto f = start_y; f < stop_y; f++) {
                sd::LongType frameStart = f * hopLength;

                // Autocorrelation-based pitch detection
                T maxCorr = 0;
                int bestLag = 0;

                for (int lag = minLag; lag <= maxLag && lag < frameLength; lag++) {
                    T corr = 0;
                    T normA = 0, normB = 0;
                    for (int n = 0; n < frameLength - lag; n++) {
                        T a = inputPtr[b * numSamples + frameStart + n];
                        T b_val = inputPtr[b * numSamples + frameStart + n + lag];
                        corr += a * b_val;
                        normA += a * a;
                        normB += b_val * b_val;
                    }

                    T norm = std::sqrt(normA * normB);
                    T normalizedCorr = (norm > static_cast<T>(1e-10)) ? corr / norm : static_cast<T>(0);

                    if (normalizedCorr > maxCorr) {
                        maxCorr = normalizedCorr;
                        bestLag = lag;
                    }
                }

                // Convert lag to frequency
                outputPtr[b * numFrames + f] = (bestLag > 0 && maxCorr > static_cast<T>(0.2))
                    ? static_cast<T>(sampleRate) / static_cast<T>(bestLag) : static_cast<T>(0);
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, numFrames, 1);
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
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto origSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);
    const auto targetSamples = hasBatch ? output->sizeAt(1) : output->sizeAt(0);

    auto inputPtr = input->bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    const double ratio = static_cast<double>(origSampleRate) / static_cast<double>(targetSampleRate);
    const int sincRadius = 8;  // Lanczos kernel half-width

    auto func = PRAGMA_THREADS_FOR_2D {
        for (auto b = start_x; b < stop_x; b++) {
            for (auto i = start_y; i < stop_y; i++) {
                double srcPos = i * ratio;
                int srcCenter = static_cast<int>(srcPos);

                T sum = 0;
                T weightSum = 0;

                for (int j = srcCenter - sincRadius + 1; j <= srcCenter + sincRadius; j++) {
                    if (j >= 0 && j < origSamples) {
                        double x = srcPos - j;
                        // Lanczos kernel: sinc(x) * sinc(x/a)
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
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1, 0, targetSamples, 1);
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
    const auto inputRank = input->rankOf();
    const bool hasBatch = inputRank == 2;
    const auto batchSize = hasBatch ? input->sizeAt(0) : 1;
    const auto numSamples = hasBatch ? input->sizeAt(1) : input->sizeAt(0);

    // Compute mel spectrogram into a temporary buffer
    sd::LongType rawNumFrames = (numSamples - fftSize) / hopLength + 1;

    // Allocate temporary mel spectrogram buffer
    std::vector<sd::LongType> tempShape;
    if (hasBatch) {
        tempShape = {static_cast<sd::LongType>(batchSize),
                     static_cast<sd::LongType>(numMelBins),
                     static_cast<sd::LongType>(rawNumFrames)};
    } else {
        tempShape = {static_cast<sd::LongType>(numMelBins),
                     static_cast<sd::LongType>(rawNumFrames)};
    }

    NDArray tempMel(input->ordering(), tempShape, input->dataType(), context);

    // Reuse existing melSpectrogram_ for the heavy lifting (STFT + filterbank)
    melSpectrogram_<T>(context, input, sampleRate, fftSize, hopLength, numMelBins,
                       lowerEdgeHz, upperEdgeHz, 2.0, &tempMel);

    auto tempPtr = tempMel.bufferAsT<T>();
    auto outputPtr = output->bufferAsT<T>();

    // For each batch: pad/trim to targetFrames, then apply Whisper log normalization
    auto func = PRAGMA_THREADS_FOR {
        for (auto b = start; b < stop; b++) {
            T* srcBatch = tempPtr + b * numMelBins * rawNumFrames;
            T* dstBatch = outputPtr + b * numMelBins * targetFrames;

            // Copy mel data to output, padding with zeros or trimming
            sd::LongType framesToCopy = std::min(rawNumFrames, static_cast<sd::LongType>(targetFrames));

            for (int m = 0; m < numMelBins; m++) {
                // Copy existing frames
                for (sd::LongType f = 0; f < framesToCopy; f++) {
                    dstBatch[m * targetFrames + f] = srcBatch[m * rawNumFrames + f];
                }
                // Zero-pad remaining frames
                for (sd::LongType f = framesToCopy; f < targetFrames; f++) {
                    dstBatch[m * targetFrames + f] = static_cast<T>(0);
                }
            }

            // Step 1: log10(max(mel, 1e-10))
            sd::LongType totalElements = numMelBins * targetFrames;
            T globalMax = static_cast<T>(-1e30);
            for (sd::LongType i = 0; i < totalElements; i++) {
                T val = dstBatch[i];
                val = val > static_cast<T>(1e-10) ? val : static_cast<T>(1e-10);
                val = static_cast<T>(std::log10(static_cast<double>(val)));
                dstBatch[i] = val;
                if (val > globalMax) globalMax = val;
            }

            // Step 2: clamp to globalMax - 8.0
            T clampMin = globalMax - static_cast<T>(8.0);
            for (sd::LongType i = 0; i < totalElements; i++) {
                if (dstBatch[i] < clampMin) dstBatch[i] = clampMin;
            }

            // Step 3: (log_spec + 4.0) / 4.0
            for (sd::LongType i = 0; i < totalElements; i++) {
                dstBatch[i] = (dstBatch[i] + static_cast<T>(4.0)) / static_cast<T>(4.0);
            }
        }
    };

    samediff::Threads::parallel_for(func, 0, batchSize, 1);
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
