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

#ifndef LIBND4J_HELPERS_AUDIO_H
#define LIBND4J_HELPERS_AUDIO_H

#include <array/NDArray.h>
#include <system/op_boilerplate.h>

namespace sd {
namespace ops {
namespace helpers {

SD_LIB_EXPORT void melFilterbank(LaunchContext* context, int numMelBins, int fftSize,
                                  int sampleRate, double lowerEdgeHz, double upperEdgeHz,
                                  NDArray* output);

SD_LIB_EXPORT void melSpectrogram(LaunchContext* context, NDArray* input,
                                   int sampleRate, int fftSize, int hopLength, int numMelBins,
                                   double lowerEdgeHz, double upperEdgeHz, double power,
                                   NDArray* output);

SD_LIB_EXPORT void mfcc(LaunchContext* context, NDArray* input,
                         int sampleRate, int fftSize, int hopLength,
                         int numMelBins, int numMfcc,
                         double lowerEdgeHz, double upperEdgeHz,
                         NDArray* output);

SD_LIB_EXPORT void griffinLim(LaunchContext* context, NDArray* magnitudeSpectrogram,
                               int fftSize, int hopLength, int numIterations,
                               NDArray* output);

SD_LIB_EXPORT void spectralCentroid(LaunchContext* context, NDArray* input,
                                     int sampleRate, int fftSize,
                                     NDArray* output);

SD_LIB_EXPORT void spectralRolloff(LaunchContext* context, NDArray* input,
                                    int sampleRate, int fftSize, double rolloffPercent,
                                    NDArray* output);

SD_LIB_EXPORT void chromaFeatures(LaunchContext* context, NDArray* input,
                                   int sampleRate, int fftSize, int numChroma,
                                   NDArray* output);

SD_LIB_EXPORT void zeroCrossingRate(LaunchContext* context, NDArray* input,
                                     int frameLength, int hopLength,
                                     NDArray* output);

SD_LIB_EXPORT void preEmphasis(LaunchContext* context, NDArray* input,
                                double coefficient, NDArray* output);

SD_LIB_EXPORT void audioNormalize(LaunchContext* context, NDArray* input,
                                   double targetLevel, bool useRms,
                                   NDArray* output);

SD_LIB_EXPORT void aWeighting(LaunchContext* context, NDArray* frequencies,
                               NDArray* output);

SD_LIB_EXPORT void pitchDetection(LaunchContext* context, NDArray* input,
                                   int sampleRate, int frameLength, int hopLength,
                                   double minFreq, double maxFreq,
                                   NDArray* output);

SD_LIB_EXPORT void audioResample(LaunchContext* context, NDArray* input,
                                  int origSampleRate, int targetSampleRate,
                                  NDArray* output);

SD_LIB_EXPORT void whisperMelSpectrogram(LaunchContext* context, NDArray* input,
                                          int sampleRate, int fftSize, int hopLength,
                                          int numMelBins, int targetFrames,
                                          double lowerEdgeHz, double upperEdgeHz,
                                          NDArray* output);

}  // namespace helpers
}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HELPERS_AUDIO_H
