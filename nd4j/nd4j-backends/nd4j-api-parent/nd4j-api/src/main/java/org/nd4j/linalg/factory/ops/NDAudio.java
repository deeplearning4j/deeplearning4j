/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

//================== GENERATED CODE - DO NOT MODIFY THIS FILE ==================

package org.nd4j.linalg.factory.ops;

import static org.nd4j.linalg.factory.NDValidation.isSameType;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.NDValidation;
import org.nd4j.linalg.factory.Nd4j;

public class NDAudio {
  public NDAudio() {
  }

  /**
   * Compute A-weighting filter values for given frequencies.<br>
   * A-weighting (IEC 61672) approximates the frequency response of human hearing,<br>
   * de-emphasizing very low and very high frequencies. Returns weights in dB.<br>
   *
   * @param frequencies Frequency values in Hz to compute weights for (NUMERIC type)
   * @return output A-weighting values in dB for each input frequency (NUMERIC type)
   */
  public INDArray aWeighting(INDArray frequencies) {
    NDValidation.validateNumerical("aWeighting", "frequencies", frequencies);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.AWeighting(frequencies));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Normalize audio to a target peak or RMS level.<br>
   * Scales the audio signal so that its peak amplitude or RMS value matches the target.<br>
   * Essential preprocessing step for consistent audio feature extraction.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param targetLevel Target peak or RMS level
   * @param useRms If true, normalize to RMS level; if false, normalize to peak level
   * @return output Normalized audio with same shape as input (NUMERIC type)
   */
  public INDArray audioNormalize(INDArray input, double targetLevel, boolean useRms) {
    NDValidation.validateNumerical("audioNormalize", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.AudioNormalize(input, targetLevel, useRms));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Normalize audio to a target peak or RMS level.<br>
   * Scales the audio signal so that its peak amplitude or RMS value matches the target.<br>
   * Essential preprocessing step for consistent audio feature extraction.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Normalized audio with same shape as input (NUMERIC type)
   */
  public INDArray audioNormalize(INDArray input) {
    NDValidation.validateNumerical("audioNormalize", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.AudioNormalize(input, 1.0, false));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Resample audio from one sample rate to another using sinc interpolation.<br>
   * Uses a windowed sinc (Lanczos) kernel for high-quality sample rate conversion.<br>
   * Supports both upsampling and downsampling.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param origSampleRate Original sample rate in Hz
   * @param targetSampleRate Target sample rate in Hz
   * @return output Resampled audio waveform (NUMERIC type)
   */
  public INDArray audioResample(INDArray input, int origSampleRate, int targetSampleRate) {
    NDValidation.validateNumerical("audioResample", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.AudioResample(input, origSampleRate, targetSampleRate));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute chroma features (pitch class profile) from a magnitude spectrogram.<br>
   * Maps the spectrogram onto 12 bins representing the 12 distinct semitones (pitch classes)<br>
   * of the musical octave. Useful for music analysis and chord recognition.<br>
   *
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param fftSize FFT window size used to produce the spectrogram
   * @param numChroma Number of chroma bins (typically 12 for semitones)
   * @return output Chroma features of shape [batch, numChroma, numFrames] (NUMERIC type)
   */
  public INDArray chromaFeatures(INDArray input, int sampleRate, int fftSize, int numChroma) {
    NDValidation.validateNumerical("chromaFeatures", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.ChromaFeatures(input, sampleRate, fftSize, numChroma));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute chroma features (pitch class profile) from a magnitude spectrogram.<br>
   * Maps the spectrogram onto 12 bins representing the 12 distinct semitones (pitch classes)<br>
   * of the musical octave. Useful for music analysis and chord recognition.<br>
   *
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Chroma features of shape [batch, numChroma, numFrames] (NUMERIC type)
   */
  public INDArray chromaFeatures(INDArray input) {
    NDValidation.validateNumerical("chromaFeatures", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.ChromaFeatures(input, 22050, 2048, 12));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Reconstruct an audio waveform from a magnitude spectrogram using the Griffin-Lim algorithm.<br>
   * Iteratively estimates phase information to invert the STFT. Used in audio synthesis<br>
   * and vocoder applications (e.g., text-to-speech).<br>
   *
   * @param magnitudeSpectrogram Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @param fftSize FFT window size
   * @param hopLength Number of samples between successive frames
   * @param numIterations Number of Griffin-Lim iterations
   * @return output Reconstructed waveform of shape [batch, samples] (NUMERIC type)
   */
  public INDArray griffinLim(INDArray magnitudeSpectrogram, int fftSize, int hopLength,
      int numIterations) {
    NDValidation.validateNumerical("griffinLim", "magnitudeSpectrogram", magnitudeSpectrogram);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.GriffinLim(magnitudeSpectrogram, fftSize, hopLength, numIterations));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Reconstruct an audio waveform from a magnitude spectrogram using the Griffin-Lim algorithm.<br>
   * Iteratively estimates phase information to invert the STFT. Used in audio synthesis<br>
   * and vocoder applications (e.g., text-to-speech).<br>
   *
   * @param magnitudeSpectrogram Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Reconstructed waveform of shape [batch, samples] (NUMERIC type)
   */
  public INDArray griffinLim(INDArray magnitudeSpectrogram) {
    NDValidation.validateNumerical("griffinLim", "magnitudeSpectrogram", magnitudeSpectrogram);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.GriffinLim(magnitudeSpectrogram, 2048, 512, 32));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Create a mel-scale triangular filterbank matrix.<br>
   * Maps linear frequency spectrogram bins to mel-frequency bins using<br>
   * overlapping triangular filters. Used as a component of mel spectrogram and MFCC extraction.<br>
   *
   * @param numMelBins Number of mel frequency bins
   * @param fftSize FFT size (number of frequency bins in the spectrogram)
   * @param sampleRate Audio sample rate in Hz
   * @param lowerEdgeHz Lower edge of the mel band in Hz
   * @param upperEdgeHz Upper edge of the mel band in Hz
   * @return output Mel filterbank matrix of shape [numMelBins, fftSize/2+1] (NUMERIC type)
   */
  public INDArray melFilterbank(int numMelBins, int fftSize, int sampleRate, double lowerEdgeHz,
      double upperEdgeHz) {
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.MelFilterbank(numMelBins, fftSize, sampleRate, lowerEdgeHz, upperEdgeHz));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Create a mel-scale triangular filterbank matrix.<br>
   * Maps linear frequency spectrogram bins to mel-frequency bins using<br>
   * overlapping triangular filters. Used as a component of mel spectrogram and MFCC extraction.<br>
   *
   * @param numMelBins Number of mel frequency bins
   * @param fftSize FFT size (number of frequency bins in the spectrogram)
   * @param sampleRate Audio sample rate in Hz
   * @return output Mel filterbank matrix of shape [numMelBins, fftSize/2+1] (NUMERIC type)
   */
  public INDArray melFilterbank(int numMelBins, int fftSize, int sampleRate) {
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.MelFilterbank(numMelBins, fftSize, sampleRate, 0.0, 8000.0));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute mel spectrogram from audio waveform.<br>
   * Applies STFT, converts to power/amplitude spectrogram, then maps through<br>
   * a mel filterbank. This is the standard front-end for many audio ML models.<br>
   *
   * @param input Audio waveform tensor of shape [batch, samples] or [samples] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param fftSize FFT window size
   * @param hopLength Number of samples between successive frames
   * @param numMelBins Number of mel frequency bins
   * @param lowerEdgeHz Lower edge of the mel band in Hz
   * @param upperEdgeHz Upper edge of the mel band in Hz
   * @param power Exponent for the magnitude spectrogram (1=amplitude, 2=power)
   * @return output Mel spectrogram of shape [batch, numMelBins, numFrames] (NUMERIC type)
   */
  public INDArray melSpectrogram(INDArray input, int sampleRate, int fftSize, int hopLength,
      int numMelBins, double lowerEdgeHz, double upperEdgeHz, double power) {
    NDValidation.validateNumerical("melSpectrogram", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.MelSpectrogram(input, sampleRate, fftSize, hopLength, numMelBins, lowerEdgeHz, upperEdgeHz, power));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute mel spectrogram from audio waveform.<br>
   * Applies STFT, converts to power/amplitude spectrogram, then maps through<br>
   * a mel filterbank. This is the standard front-end for many audio ML models.<br>
   *
   * @param input Audio waveform tensor of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Mel spectrogram of shape [batch, numMelBins, numFrames] (NUMERIC type)
   */
  public INDArray melSpectrogram(INDArray input) {
    NDValidation.validateNumerical("melSpectrogram", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.MelSpectrogram(input, 22050, 2048, 512, 128, 0.0, 8000.0, 2.0));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute Mel-Frequency Cepstral Coefficients (MFCCs) from audio waveform.<br>
   * Applies mel spectrogram extraction, log scaling, and DCT-II to produce<br>
   * cepstral coefficients. MFCCs are widely used features for speech and audio recognition.<br>
   *
   * @param input Audio waveform tensor of shape [batch, samples] or [samples] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param fftSize FFT window size
   * @param hopLength Number of samples between successive frames
   * @param numMelBins Number of mel frequency bins
   * @param numMfcc Number of MFCC coefficients to return
   * @param lowerEdgeHz Lower edge of the mel band in Hz
   * @param upperEdgeHz Upper edge of the mel band in Hz
   * @return output MFCC coefficients of shape [batch, numMfcc, numFrames] (NUMERIC type)
   */
  public INDArray mfcc(INDArray input, int sampleRate, int fftSize, int hopLength, int numMelBins,
      int numMfcc, double lowerEdgeHz, double upperEdgeHz) {
    NDValidation.validateNumerical("mfcc", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.MFCC(input, sampleRate, fftSize, hopLength, numMelBins, numMfcc, lowerEdgeHz, upperEdgeHz));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute Mel-Frequency Cepstral Coefficients (MFCCs) from audio waveform.<br>
   * Applies mel spectrogram extraction, log scaling, and DCT-II to produce<br>
   * cepstral coefficients. MFCCs are widely used features for speech and audio recognition.<br>
   *
   * @param input Audio waveform tensor of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output MFCC coefficients of shape [batch, numMfcc, numFrames] (NUMERIC type)
   */
  public INDArray mfcc(INDArray input) {
    NDValidation.validateNumerical("mfcc", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.MFCC(input, 22050, 2048, 512, 128, 13, 0.0, 8000.0));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Detect fundamental frequency (pitch) using autocorrelation method.<br>
   * Estimates the fundamental frequency of an audio signal per frame by finding<br>
   * the peak of the autocorrelation function within the expected frequency range.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param frameLength Length of each analysis frame
   * @param hopLength Number of samples between successive frames
   * @param minFreq Minimum detectable frequency in Hz
   * @param maxFreq Maximum detectable frequency in Hz
   * @return output Detected fundamental frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public INDArray pitchDetection(INDArray input, int sampleRate, int frameLength, int hopLength,
      double minFreq, double maxFreq) {
    NDValidation.validateNumerical("pitchDetection", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.PitchDetection(input, sampleRate, frameLength, hopLength, minFreq, maxFreq));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Detect fundamental frequency (pitch) using autocorrelation method.<br>
   * Estimates the fundamental frequency of an audio signal per frame by finding<br>
   * the peak of the autocorrelation function within the expected frequency range.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Detected fundamental frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public INDArray pitchDetection(INDArray input) {
    NDValidation.validateNumerical("pitchDetection", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.PitchDetection(input, 22050, 2048, 512, 80.0, 1000.0));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Apply pre-emphasis filter to an audio signal.<br>
   * Computes y[n] = x[n] - coefficient * x[n-1], a first-order high-pass filter<br>
   * that amplifies high frequencies. Standard preprocessing for speech recognition.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param coefficient Pre-emphasis coefficient (typically 0.95-0.97)
   * @return output Pre-emphasized signal with same shape as input (NUMERIC type)
   */
  public INDArray preEmphasis(INDArray input, double coefficient) {
    NDValidation.validateNumerical("preEmphasis", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.PreEmphasis(input, coefficient));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Apply pre-emphasis filter to an audio signal.<br>
   * Computes y[n] = x[n] - coefficient * x[n-1], a first-order high-pass filter<br>
   * that amplifies high frequencies. Standard preprocessing for speech recognition.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Pre-emphasized signal with same shape as input (NUMERIC type)
   */
  public INDArray preEmphasis(INDArray input) {
    NDValidation.validateNumerical("preEmphasis", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.PreEmphasis(input, 0.97));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the spectral centroid of a magnitude spectrogram.<br>
   * The spectral centroid is the weighted mean of frequencies by their magnitudes,<br>
   * indicating the "center of mass" of the spectrum. It is a measure of spectral brightness.<br>
   *
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param fftSize FFT window size used to produce the spectrogram
   * @return output Spectral centroid per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public INDArray spectralCentroid(INDArray input, int sampleRate, int fftSize) {
    NDValidation.validateNumerical("spectralCentroid", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.SpectralCentroid(input, sampleRate, fftSize));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the spectral centroid of a magnitude spectrogram.<br>
   * The spectral centroid is the weighted mean of frequencies by their magnitudes,<br>
   * indicating the "center of mass" of the spectrum. It is a measure of spectral brightness.<br>
   *
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Spectral centroid per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public INDArray spectralCentroid(INDArray input) {
    NDValidation.validateNumerical("spectralCentroid", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.SpectralCentroid(input, 22050, 2048));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the spectral rolloff frequency.<br>
   * The rolloff frequency is the frequency below which a specified percentage<br>
   * of the total spectral energy falls. Useful for distinguishing voiced/unvoiced speech.<br>
   *
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param fftSize FFT window size used to produce the spectrogram
   * @param rolloffPercent Percentage of spectral energy (0.0 to 1.0)
   * @return output Spectral rolloff frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public INDArray spectralRolloff(INDArray input, int sampleRate, int fftSize,
      double rolloffPercent) {
    NDValidation.validateNumerical("spectralRolloff", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.SpectralRolloff(input, sampleRate, fftSize, rolloffPercent));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the spectral rolloff frequency.<br>
   * The rolloff frequency is the frequency below which a specified percentage<br>
   * of the total spectral energy falls. Useful for distinguishing voiced/unvoiced speech.<br>
   *
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Spectral rolloff frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public INDArray spectralRolloff(INDArray input) {
    NDValidation.validateNumerical("spectralRolloff", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.SpectralRolloff(input, 22050, 2048, 0.85));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the zero crossing rate of an audio signal.<br>
   * The zero crossing rate is the rate at which the signal changes sign,<br>
   * computed per frame. Useful for speech/music discrimination and onset detection.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param frameLength Length of each analysis frame
   * @param hopLength Number of samples between successive frames
   * @return output Zero crossing rate per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public INDArray zeroCrossingRate(INDArray input, int frameLength, int hopLength) {
    NDValidation.validateNumerical("zeroCrossingRate", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.ZeroCrossingRate(input, frameLength, hopLength));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }

  /**
   * Compute the zero crossing rate of an audio signal.<br>
   * The zero crossing rate is the rate at which the signal changes sign,<br>
   * computed per frame. Useful for speech/music discrimination and onset detection.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Zero crossing rate per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public INDArray zeroCrossingRate(INDArray input) {
    NDValidation.validateNumerical("zeroCrossingRate", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.audio.ZeroCrossingRate(input, 2048, 512));
    try {
      return __tmp[0];
    } finally {
      if(__tmp != null) {
        for(int __i = 1; __i < __tmp.length; __i++) {
          if(__tmp[__i] != null) {
            __tmp[__i].close();
          }
        }
      }
    }
  }
}
