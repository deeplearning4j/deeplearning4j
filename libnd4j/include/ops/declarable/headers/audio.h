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

#ifndef LIBND4J_HEADERS_AUDIO_H
#define LIBND4J_HEADERS_AUDIO_H

#include <ops/declarable/headers/common.h>

namespace sd {
namespace ops {

/**
 * Mel Filterbank
 *
 * Creates a mel-scale triangular filterbank matrix.
 *
 * Int arguments:
 *   0 - numMelBins: Number of mel frequency bins
 *   1 - fftSize: FFT size
 *   2 - sampleRate: Audio sample rate in Hz
 *
 * T arguments:
 *   0 - lowerEdgeHz: Lower edge of mel band (default: 0.0)
 *   1 - upperEdgeHz: Upper edge of mel band (default: 8000.0)
 *
 * Output:
 *   0 - Filterbank matrix [numMelBins, fftSize/2+1]
 */
#if NOT_EXCLUDED(OP_mel_filterbank)
DECLARE_CUSTOM_OP(mel_filterbank, 0, 1, false, -2, -2);
#endif

/**
 * Mel Spectrogram
 *
 * Compute mel spectrogram from audio waveform (STFT + mel filterbank + power).
 *
 * Input arrays:
 *   0 - input: Audio waveform [batch, samples] or [samples]
 *
 * Int arguments:
 *   0 - sampleRate (default: 22050)
 *   1 - fftSize (default: 2048)
 *   2 - hopLength (default: 512)
 *   3 - numMelBins (default: 128)
 *
 * T arguments:
 *   0 - lowerEdgeHz (default: 0.0)
 *   1 - upperEdgeHz (default: 8000.0)
 *   2 - power (default: 2.0)
 *
 * Output:
 *   0 - Mel spectrogram [batch, numMelBins, numFrames]
 */
#if NOT_EXCLUDED(OP_mel_spectrogram)
DECLARE_CUSTOM_OP(mel_spectrogram, 1, 1, false, -2, -2);
#endif

/**
 * MFCC - Mel-Frequency Cepstral Coefficients
 *
 * Input arrays:
 *   0 - input: Audio waveform [batch, samples] or [samples]
 *
 * Int arguments:
 *   0 - sampleRate (default: 22050)
 *   1 - fftSize (default: 2048)
 *   2 - hopLength (default: 512)
 *   3 - numMelBins (default: 128)
 *   4 - numMfcc (default: 13)
 *
 * T arguments:
 *   0 - lowerEdgeHz (default: 0.0)
 *   1 - upperEdgeHz (default: 8000.0)
 *
 * Output:
 *   0 - MFCC coefficients [batch, numMfcc, numFrames]
 */
#if NOT_EXCLUDED(OP_mfcc)
DECLARE_CUSTOM_OP(mfcc, 1, 1, false, -2, -2);
#endif

/**
 * Griffin-Lim Algorithm
 *
 * Reconstruct audio waveform from magnitude spectrogram.
 *
 * Input arrays:
 *   0 - magnitudeSpectrogram: [batch, freqBins, numFrames]
 *
 * Int arguments:
 *   0 - fftSize (default: 2048)
 *   1 - hopLength (default: 512)
 *   2 - numIterations (default: 32)
 *
 * Output:
 *   0 - Reconstructed waveform [batch, samples]
 */
#if NOT_EXCLUDED(OP_griffin_lim)
DECLARE_CUSTOM_OP(griffin_lim, 1, 1, false, 0, -2);
#endif

/**
 * Spectral Centroid
 *
 * Weighted mean of frequencies by magnitude.
 *
 * Input arrays:
 *   0 - input: Magnitude spectrogram [batch, freqBins, numFrames]
 *
 * Int arguments:
 *   0 - sampleRate (default: 22050)
 *   1 - fftSize (default: 2048)
 *
 * Output:
 *   0 - Spectral centroid per frame [batch, numFrames]
 */
#if NOT_EXCLUDED(OP_spectral_centroid)
DECLARE_CUSTOM_OP(spectral_centroid, 1, 1, false, 0, -2);
#endif

/**
 * Spectral Rolloff
 *
 * Frequency below which a given percentage of spectral energy falls.
 *
 * Input arrays:
 *   0 - input: Magnitude spectrogram [batch, freqBins, numFrames]
 *
 * Int arguments:
 *   0 - sampleRate (default: 22050)
 *   1 - fftSize (default: 2048)
 *
 * T arguments:
 *   0 - rolloffPercent (default: 0.85)
 *
 * Output:
 *   0 - Rolloff frequency per frame [batch, numFrames]
 */
#if NOT_EXCLUDED(OP_spectral_rolloff)
DECLARE_CUSTOM_OP(spectral_rolloff, 1, 1, false, -2, -2);
#endif

/**
 * Chroma Features
 *
 * 12-bin pitch class profile from spectrogram.
 *
 * Input arrays:
 *   0 - input: Magnitude spectrogram [batch, freqBins, numFrames]
 *
 * Int arguments:
 *   0 - sampleRate (default: 22050)
 *   1 - fftSize (default: 2048)
 *   2 - numChroma (default: 12)
 *
 * Output:
 *   0 - Chroma features [batch, numChroma, numFrames]
 */
#if NOT_EXCLUDED(OP_chroma_features)
DECLARE_CUSTOM_OP(chroma_features, 1, 1, false, 0, -2);
#endif

/**
 * Zero Crossing Rate
 *
 * Rate of sign changes in the signal per frame.
 *
 * Input arrays:
 *   0 - input: Audio waveform [batch, samples] or [samples]
 *
 * Int arguments:
 *   0 - frameLength (default: 2048)
 *   1 - hopLength (default: 512)
 *
 * Output:
 *   0 - Zero crossing rate per frame [batch, numFrames]
 */
#if NOT_EXCLUDED(OP_zero_crossing_rate)
DECLARE_CUSTOM_OP(zero_crossing_rate, 1, 1, false, 0, -2);
#endif

/**
 * Pre-Emphasis Filter
 *
 * y[n] = x[n] - coeff * x[n-1]
 *
 * Input arrays:
 *   0 - input: Audio waveform [batch, samples] or [samples]
 *
 * T arguments:
 *   0 - coefficient (default: 0.97)
 *
 * Output:
 *   0 - Pre-emphasized signal with same shape as input
 */
#if NOT_EXCLUDED(OP_pre_emphasis)
DECLARE_CUSTOM_OP(pre_emphasis, 1, 1, false, -2, 0);
#endif

/**
 * Audio Normalize
 *
 * Normalize audio to target peak or RMS level.
 *
 * Input arrays:
 *   0 - input: Audio waveform [batch, samples] or [samples]
 *
 * T arguments:
 *   0 - targetLevel (default: 1.0)
 *
 * Int arguments:
 *   0 - useRms: 0 for peak, 1 for RMS (default: 0)
 *
 * Output:
 *   0 - Normalized audio with same shape as input
 */
#if NOT_EXCLUDED(OP_audio_normalize)
DECLARE_CUSTOM_OP(audio_normalize, 1, 1, false, -2, -2);
#endif

/**
 * A-Weighting
 *
 * A-weighting filter values (IEC 61672) for given frequencies.
 *
 * Input arrays:
 *   0 - frequencies: Frequency values in Hz
 *
 * Output:
 *   0 - A-weighting values in dB
 */
#if NOT_EXCLUDED(OP_a_weighting)
DECLARE_CUSTOM_OP(a_weighting, 1, 1, false, 0, 0);
#endif

/**
 * Pitch Detection
 *
 * Fundamental frequency estimation using autocorrelation.
 *
 * Input arrays:
 *   0 - input: Audio waveform [batch, samples] or [samples]
 *
 * Int arguments:
 *   0 - sampleRate (default: 22050)
 *   1 - frameLength (default: 2048)
 *   2 - hopLength (default: 512)
 *
 * T arguments:
 *   0 - minFreq (default: 80.0)
 *   1 - maxFreq (default: 1000.0)
 *
 * Output:
 *   0 - Detected fundamental frequency per frame [batch, numFrames]
 */
#if NOT_EXCLUDED(OP_pitch_detection)
DECLARE_CUSTOM_OP(pitch_detection, 1, 1, false, -2, -2);
#endif

/**
 * Audio Resample
 *
 * Resample audio using sinc interpolation.
 *
 * Input arrays:
 *   0 - input: Audio waveform [batch, samples] or [samples]
 *
 * Int arguments:
 *   0 - origSampleRate: Original sample rate in Hz
 *   1 - targetSampleRate: Target sample rate in Hz
 *
 * Output:
 *   0 - Resampled waveform
 */
#if NOT_EXCLUDED(OP_audio_resample)
DECLARE_CUSTOM_OP(audio_resample, 1, 1, false, 0, 2);
#endif

/**
 * Whisper Mel Spectrogram
 *
 * Combined mel spectrogram extraction with Whisper-specific log normalization.
 * Computes: mel → log10(max(mel, 1e-10)) → clamp(max - 8.0) → (x + 4.0) / 4.0
 * Pads or trims frames to targetFrames (default 3000 for 30-second chunks).
 *
 * Input arrays:
 *   0 - input: Audio waveform [batch, samples] or [samples]
 *
 * Int arguments:
 *   0 - sampleRate (default: 16000)
 *   1 - fftSize (default: 400)
 *   2 - hopLength (default: 160)
 *   3 - numMelBins (default: 80)
 *   4 - targetFrames (default: 3000)
 *
 * T arguments:
 *   0 - lowerEdgeHz (default: 0.0)
 *   1 - upperEdgeHz (default: 8000.0)
 *
 * Output:
 *   0 - Whisper mel features [batch, numMelBins, targetFrames]
 */
#if NOT_EXCLUDED(OP_whisper_mel_spectrogram)
DECLARE_CUSTOM_OP(whisper_mel_spectrogram, 1, 1, false, -2, -2);
#endif

}  // namespace ops
}  // namespace sd

#endif  // LIBND4J_HEADERS_AUDIO_H
