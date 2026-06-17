/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * See the NOTICE file distributed with this work for additional
 *  * information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.codegen.ops

import org.nd4j.codegen.api.Language
import org.nd4j.codegen.api.doc.DocScope
import org.nd4j.codegen.dsl.*
import org.nd4j.codegen.api.DataType.*

fun SDAudio() = Namespace("Audio") {
    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.impl.audio"

    Op("melFilterbank") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "MelFilterbank"
        Arg(INT, "numMelBins") { description = "Number of mel frequency bins" }
        Arg(INT, "fftSize") { description = "FFT size (number of frequency bins in the spectrogram)" }
        Arg(INT, "sampleRate") { description = "Audio sample rate in Hz" }
        Arg(FLOATING_POINT, "lowerEdgeHz") { description = "Lower edge of the mel band in Hz"; defaultValue = 0.0 }
        Arg(FLOATING_POINT, "upperEdgeHz") { description = "Upper edge of the mel band in Hz"; defaultValue = 8000.0 }

        Output(NUMERIC, "output") { description = "Mel filterbank matrix of shape [numMelBins, fftSize/2+1]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Create a mel-scale triangular filterbank matrix.
             Maps linear frequency spectrogram bins to mel-frequency bins using
             overlapping triangular filters. Used as a component of mel spectrogram and MFCC extraction.
            """.trimIndent()
        }
    }

    Op("melSpectrogram") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "MelSpectrogram"
        Input(NUMERIC, "input") { description = "Audio waveform tensor of shape [batch, samples] or [samples]" }
        Arg(INT, "sampleRate") { description = "Audio sample rate in Hz"; defaultValue = 22050 }
        Arg(INT, "fftSize") { description = "FFT window size"; defaultValue = 2048 }
        Arg(INT, "hopLength") { description = "Number of samples between successive frames"; defaultValue = 512 }
        Arg(INT, "numMelBins") { description = "Number of mel frequency bins"; defaultValue = 128 }
        Arg(FLOATING_POINT, "lowerEdgeHz") { description = "Lower edge of the mel band in Hz"; defaultValue = 0.0 }
        Arg(FLOATING_POINT, "upperEdgeHz") { description = "Upper edge of the mel band in Hz"; defaultValue = 8000.0 }
        Arg(FLOATING_POINT, "power") { description = "Exponent for the magnitude spectrogram (1=amplitude, 2=power)"; defaultValue = 2.0 }

        Output(NUMERIC, "output") { description = "Mel spectrogram of shape [batch, numMelBins, numFrames]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Compute mel spectrogram from audio waveform.
             Applies STFT, converts to power/amplitude spectrogram, then maps through
             a mel filterbank. This is the standard front-end for many audio ML models.
            """.trimIndent()
        }
    }

    Op("mfcc") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "MFCC"
        Input(NUMERIC, "input") { description = "Audio waveform tensor of shape [batch, samples] or [samples]" }
        Arg(INT, "sampleRate") { description = "Audio sample rate in Hz"; defaultValue = 22050 }
        Arg(INT, "fftSize") { description = "FFT window size"; defaultValue = 2048 }
        Arg(INT, "hopLength") { description = "Number of samples between successive frames"; defaultValue = 512 }
        Arg(INT, "numMelBins") { description = "Number of mel frequency bins"; defaultValue = 128 }
        Arg(INT, "numMfcc") { description = "Number of MFCC coefficients to return"; defaultValue = 13 }
        Arg(FLOATING_POINT, "lowerEdgeHz") { description = "Lower edge of the mel band in Hz"; defaultValue = 0.0 }
        Arg(FLOATING_POINT, "upperEdgeHz") { description = "Upper edge of the mel band in Hz"; defaultValue = 8000.0 }

        Output(NUMERIC, "output") { description = "MFCC coefficients of shape [batch, numMfcc, numFrames]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Compute Mel-Frequency Cepstral Coefficients (MFCCs) from audio waveform.
             Applies mel spectrogram extraction, log scaling, and DCT-II to produce
             cepstral coefficients. MFCCs are widely used features for speech and audio recognition.
            """.trimIndent()
        }
    }

    Op("griffinLim") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "GriffinLim"
        Input(NUMERIC, "magnitudeSpectrogram") { description = "Magnitude spectrogram of shape [batch, freqBins, numFrames]" }
        Arg(INT, "fftSize") { description = "FFT window size"; defaultValue = 2048 }
        Arg(INT, "hopLength") { description = "Number of samples between successive frames"; defaultValue = 512 }
        Arg(INT, "numIterations") { description = "Number of Griffin-Lim iterations"; defaultValue = 32 }

        Output(NUMERIC, "output") { description = "Reconstructed waveform of shape [batch, samples]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Reconstruct an audio waveform from a magnitude spectrogram using the Griffin-Lim algorithm.
             Iteratively estimates phase information to invert the STFT. Used in audio synthesis
             and vocoder applications (e.g., text-to-speech).
            """.trimIndent()
        }
    }

    Op("spectralCentroid") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "SpectralCentroid"
        Input(NUMERIC, "input") { description = "Magnitude spectrogram of shape [batch, freqBins, numFrames]" }
        Arg(INT, "sampleRate") { description = "Audio sample rate in Hz"; defaultValue = 22050 }
        Arg(INT, "fftSize") { description = "FFT window size used to produce the spectrogram"; defaultValue = 2048 }

        Output(NUMERIC, "output") { description = "Spectral centroid per frame of shape [batch, numFrames]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Compute the spectral centroid of a magnitude spectrogram.
             The spectral centroid is the weighted mean of frequencies by their magnitudes,
             indicating the "center of mass" of the spectrum. It is a measure of spectral brightness.
            """.trimIndent()
        }
    }

    Op("spectralRolloff") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "SpectralRolloff"
        Input(NUMERIC, "input") { description = "Magnitude spectrogram of shape [batch, freqBins, numFrames]" }
        Arg(INT, "sampleRate") { description = "Audio sample rate in Hz"; defaultValue = 22050 }
        Arg(INT, "fftSize") { description = "FFT window size used to produce the spectrogram"; defaultValue = 2048 }
        Arg(FLOATING_POINT, "rolloffPercent") { description = "Percentage of spectral energy (0.0 to 1.0)"; defaultValue = 0.85 }

        Output(NUMERIC, "output") { description = "Spectral rolloff frequency per frame of shape [batch, numFrames]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Compute the spectral rolloff frequency.
             The rolloff frequency is the frequency below which a specified percentage
             of the total spectral energy falls. Useful for distinguishing voiced/unvoiced speech.
            """.trimIndent()
        }
    }

    Op("chromaFeatures") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "ChromaFeatures"
        Input(NUMERIC, "input") { description = "Magnitude spectrogram of shape [batch, freqBins, numFrames]" }
        Arg(INT, "sampleRate") { description = "Audio sample rate in Hz"; defaultValue = 22050 }
        Arg(INT, "fftSize") { description = "FFT window size used to produce the spectrogram"; defaultValue = 2048 }
        Arg(INT, "numChroma") { description = "Number of chroma bins (typically 12 for semitones)"; defaultValue = 12 }

        Output(NUMERIC, "output") { description = "Chroma features of shape [batch, numChroma, numFrames]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Compute chroma features (pitch class profile) from a magnitude spectrogram.
             Maps the spectrogram onto 12 bins representing the 12 distinct semitones (pitch classes)
             of the musical octave. Useful for music analysis and chord recognition.
            """.trimIndent()
        }
    }

    Op("zeroCrossingRate") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "ZeroCrossingRate"
        Input(NUMERIC, "input") { description = "Audio waveform of shape [batch, samples] or [samples]" }
        Arg(INT, "frameLength") { description = "Length of each analysis frame"; defaultValue = 2048 }
        Arg(INT, "hopLength") { description = "Number of samples between successive frames"; defaultValue = 512 }

        Output(NUMERIC, "output") { description = "Zero crossing rate per frame of shape [batch, numFrames]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Compute the zero crossing rate of an audio signal.
             The zero crossing rate is the rate at which the signal changes sign,
             computed per frame. Useful for speech/music discrimination and onset detection.
            """.trimIndent()
        }
    }

    Op("preEmphasis") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "PreEmphasis"
        Input(NUMERIC, "input") { description = "Audio waveform of shape [batch, samples] or [samples]" }
        Arg(FLOATING_POINT, "coefficient") { description = "Pre-emphasis coefficient (typically 0.95-0.97)"; defaultValue = 0.97 }

        Output(NUMERIC, "output") { description = "Pre-emphasized signal with same shape as input" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Apply pre-emphasis filter to an audio signal.
             Computes y[n] = x[n] - coefficient * x[n-1], a first-order high-pass filter
             that amplifies high frequencies. Standard preprocessing for speech recognition.
            """.trimIndent()
        }
    }

    Op("audioNormalize") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AudioNormalize"
        Input(NUMERIC, "input") { description = "Audio waveform of shape [batch, samples] or [samples]" }
        Arg(FLOATING_POINT, "targetLevel") { description = "Target peak or RMS level"; defaultValue = 1.0 }
        Arg(BOOL, "useRms") { description = "If true, normalize to RMS level; if false, normalize to peak level"; defaultValue = false }

        Output(NUMERIC, "output") { description = "Normalized audio with same shape as input" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Normalize audio to a target peak or RMS level.
             Scales the audio signal so that its peak amplitude or RMS value matches the target.
             Essential preprocessing step for consistent audio feature extraction.
            """.trimIndent()
        }
    }

    Op("aWeighting") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AWeighting"
        Input(NUMERIC, "frequencies") { description = "Frequency values in Hz to compute weights for" }

        Output(NUMERIC, "output") { description = "A-weighting values in dB for each input frequency" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Compute A-weighting filter values for given frequencies.
             A-weighting (IEC 61672) approximates the frequency response of human hearing,
             de-emphasizing very low and very high frequencies. Returns weights in dB.
            """.trimIndent()
        }
    }

    Op("pitchDetection") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "PitchDetection"
        Input(NUMERIC, "input") { description = "Audio waveform of shape [batch, samples] or [samples]" }
        Arg(INT, "sampleRate") { description = "Audio sample rate in Hz"; defaultValue = 22050 }
        Arg(INT, "frameLength") { description = "Length of each analysis frame"; defaultValue = 2048 }
        Arg(INT, "hopLength") { description = "Number of samples between successive frames"; defaultValue = 512 }
        Arg(FLOATING_POINT, "minFreq") { description = "Minimum detectable frequency in Hz"; defaultValue = 80.0 }
        Arg(FLOATING_POINT, "maxFreq") { description = "Maximum detectable frequency in Hz"; defaultValue = 1000.0 }

        Output(NUMERIC, "output") { description = "Detected fundamental frequency per frame of shape [batch, numFrames]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Detect fundamental frequency (pitch) using autocorrelation method.
             Estimates the fundamental frequency of an audio signal per frame by finding
             the peak of the autocorrelation function within the expected frequency range.
            """.trimIndent()
        }
    }

    Op("audioResample") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "AudioResample"
        Input(NUMERIC, "input") { description = "Audio waveform of shape [batch, samples] or [samples]" }
        Arg(INT, "origSampleRate") { description = "Original sample rate in Hz" }
        Arg(INT, "targetSampleRate") { description = "Target sample rate in Hz" }

        Output(NUMERIC, "output") { description = "Resampled audio waveform" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Resample audio from one sample rate to another using sinc interpolation.
             Uses a windowed sinc (Lanczos) kernel for high-quality sample rate conversion.
             Supports both upsampling and downsampling.
            """.trimIndent()
        }
    }
}
