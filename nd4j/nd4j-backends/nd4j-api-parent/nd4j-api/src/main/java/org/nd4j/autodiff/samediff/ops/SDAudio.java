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

package org.nd4j.autodiff.samediff.ops;

import static org.nd4j.autodiff.samediff.ops.SDValidation.isSameType;

import java.lang.String;
import org.nd4j.autodiff.samediff.SDVariable;
import org.nd4j.autodiff.samediff.SameDiff;

public class SDAudio extends SDOps {
  public SDAudio(SameDiff sameDiff) {
    super(sameDiff);
  }

  /**
   * Compute A-weighting filter values for given frequencies.<br>
   * A-weighting (IEC 61672) approximates the frequency response of human hearing,<br>
   * de-emphasizing very low and very high frequencies. Returns weights in dB.<br>
   *
   * @param frequencies Frequency values in Hz to compute weights for (NUMERIC type)
   * @return output A-weighting values in dB for each input frequency (NUMERIC type)
   */
  public SDVariable aWeighting(SDVariable frequencies) {
    SDValidation.validateNumerical("aWeighting", "frequencies", frequencies);
    return new org.nd4j.linalg.api.ops.impl.audio.AWeighting(sd,frequencies).outputVariable();
  }

  /**
   * Compute A-weighting filter values for given frequencies.<br>
   * A-weighting (IEC 61672) approximates the frequency response of human hearing,<br>
   * de-emphasizing very low and very high frequencies. Returns weights in dB.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param frequencies Frequency values in Hz to compute weights for (NUMERIC type)
   * @return output A-weighting values in dB for each input frequency (NUMERIC type)
   */
  public SDVariable aWeighting(String name, SDVariable frequencies) {
    SDValidation.validateNumerical("aWeighting", "frequencies", frequencies);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.AWeighting(sd,frequencies).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable audioNormalize(SDVariable input, double targetLevel, boolean useRms) {
    SDValidation.validateNumerical("audioNormalize", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.AudioNormalize(sd,input, targetLevel, useRms).outputVariable();
  }

  /**
   * Normalize audio to a target peak or RMS level.<br>
   * Scales the audio signal so that its peak amplitude or RMS value matches the target.<br>
   * Essential preprocessing step for consistent audio feature extraction.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param targetLevel Target peak or RMS level
   * @param useRms If true, normalize to RMS level; if false, normalize to peak level
   * @return output Normalized audio with same shape as input (NUMERIC type)
   */
  public SDVariable audioNormalize(String name, SDVariable input, double targetLevel,
      boolean useRms) {
    SDValidation.validateNumerical("audioNormalize", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.AudioNormalize(sd,input, targetLevel, useRms).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Normalize audio to a target peak or RMS level.<br>
   * Scales the audio signal so that its peak amplitude or RMS value matches the target.<br>
   * Essential preprocessing step for consistent audio feature extraction.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Normalized audio with same shape as input (NUMERIC type)
   */
  public SDVariable audioNormalize(SDVariable input) {
    SDValidation.validateNumerical("audioNormalize", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.AudioNormalize(sd,input, 1.0, false).outputVariable();
  }

  /**
   * Normalize audio to a target peak or RMS level.<br>
   * Scales the audio signal so that its peak amplitude or RMS value matches the target.<br>
   * Essential preprocessing step for consistent audio feature extraction.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Normalized audio with same shape as input (NUMERIC type)
   */
  public SDVariable audioNormalize(String name, SDVariable input) {
    SDValidation.validateNumerical("audioNormalize", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.AudioNormalize(sd,input, 1.0, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable audioResample(SDVariable input, int origSampleRate, int targetSampleRate) {
    SDValidation.validateNumerical("audioResample", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.AudioResample(sd,input, origSampleRate, targetSampleRate).outputVariable();
  }

  /**
   * Resample audio from one sample rate to another using sinc interpolation.<br>
   * Uses a windowed sinc (Lanczos) kernel for high-quality sample rate conversion.<br>
   * Supports both upsampling and downsampling.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param origSampleRate Original sample rate in Hz
   * @param targetSampleRate Target sample rate in Hz
   * @return output Resampled audio waveform (NUMERIC type)
   */
  public SDVariable audioResample(String name, SDVariable input, int origSampleRate,
      int targetSampleRate) {
    SDValidation.validateNumerical("audioResample", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.AudioResample(sd,input, origSampleRate, targetSampleRate).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable chromaFeatures(SDVariable input, int sampleRate, int fftSize, int numChroma) {
    SDValidation.validateNumerical("chromaFeatures", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.ChromaFeatures(sd,input, sampleRate, fftSize, numChroma).outputVariable();
  }

  /**
   * Compute chroma features (pitch class profile) from a magnitude spectrogram.<br>
   * Maps the spectrogram onto 12 bins representing the 12 distinct semitones (pitch classes)<br>
   * of the musical octave. Useful for music analysis and chord recognition.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param fftSize FFT window size used to produce the spectrogram
   * @param numChroma Number of chroma bins (typically 12 for semitones)
   * @return output Chroma features of shape [batch, numChroma, numFrames] (NUMERIC type)
   */
  public SDVariable chromaFeatures(String name, SDVariable input, int sampleRate, int fftSize,
      int numChroma) {
    SDValidation.validateNumerical("chromaFeatures", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.ChromaFeatures(sd,input, sampleRate, fftSize, numChroma).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Compute chroma features (pitch class profile) from a magnitude spectrogram.<br>
   * Maps the spectrogram onto 12 bins representing the 12 distinct semitones (pitch classes)<br>
   * of the musical octave. Useful for music analysis and chord recognition.<br>
   *
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Chroma features of shape [batch, numChroma, numFrames] (NUMERIC type)
   */
  public SDVariable chromaFeatures(SDVariable input) {
    SDValidation.validateNumerical("chromaFeatures", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.ChromaFeatures(sd,input, 22050, 2048, 12).outputVariable();
  }

  /**
   * Compute chroma features (pitch class profile) from a magnitude spectrogram.<br>
   * Maps the spectrogram onto 12 bins representing the 12 distinct semitones (pitch classes)<br>
   * of the musical octave. Useful for music analysis and chord recognition.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Chroma features of shape [batch, numChroma, numFrames] (NUMERIC type)
   */
  public SDVariable chromaFeatures(String name, SDVariable input) {
    SDValidation.validateNumerical("chromaFeatures", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.ChromaFeatures(sd,input, 22050, 2048, 12).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable griffinLim(SDVariable magnitudeSpectrogram, int fftSize, int hopLength,
      int numIterations) {
    SDValidation.validateNumerical("griffinLim", "magnitudeSpectrogram", magnitudeSpectrogram);
    return new org.nd4j.linalg.api.ops.impl.audio.GriffinLim(sd,magnitudeSpectrogram, fftSize, hopLength, numIterations).outputVariable();
  }

  /**
   * Reconstruct an audio waveform from a magnitude spectrogram using the Griffin-Lim algorithm.<br>
   * Iteratively estimates phase information to invert the STFT. Used in audio synthesis<br>
   * and vocoder applications (e.g., text-to-speech).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param magnitudeSpectrogram Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @param fftSize FFT window size
   * @param hopLength Number of samples between successive frames
   * @param numIterations Number of Griffin-Lim iterations
   * @return output Reconstructed waveform of shape [batch, samples] (NUMERIC type)
   */
  public SDVariable griffinLim(String name, SDVariable magnitudeSpectrogram, int fftSize,
      int hopLength, int numIterations) {
    SDValidation.validateNumerical("griffinLim", "magnitudeSpectrogram", magnitudeSpectrogram);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.GriffinLim(sd,magnitudeSpectrogram, fftSize, hopLength, numIterations).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Reconstruct an audio waveform from a magnitude spectrogram using the Griffin-Lim algorithm.<br>
   * Iteratively estimates phase information to invert the STFT. Used in audio synthesis<br>
   * and vocoder applications (e.g., text-to-speech).<br>
   *
   * @param magnitudeSpectrogram Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Reconstructed waveform of shape [batch, samples] (NUMERIC type)
   */
  public SDVariable griffinLim(SDVariable magnitudeSpectrogram) {
    SDValidation.validateNumerical("griffinLim", "magnitudeSpectrogram", magnitudeSpectrogram);
    return new org.nd4j.linalg.api.ops.impl.audio.GriffinLim(sd,magnitudeSpectrogram, 2048, 512, 32).outputVariable();
  }

  /**
   * Reconstruct an audio waveform from a magnitude spectrogram using the Griffin-Lim algorithm.<br>
   * Iteratively estimates phase information to invert the STFT. Used in audio synthesis<br>
   * and vocoder applications (e.g., text-to-speech).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param magnitudeSpectrogram Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Reconstructed waveform of shape [batch, samples] (NUMERIC type)
   */
  public SDVariable griffinLim(String name, SDVariable magnitudeSpectrogram) {
    SDValidation.validateNumerical("griffinLim", "magnitudeSpectrogram", magnitudeSpectrogram);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.GriffinLim(sd,magnitudeSpectrogram, 2048, 512, 32).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable melFilterbank(int numMelBins, int fftSize, int sampleRate, double lowerEdgeHz,
      double upperEdgeHz) {
    return new org.nd4j.linalg.api.ops.impl.audio.MelFilterbank(sd,numMelBins, fftSize, sampleRate, lowerEdgeHz, upperEdgeHz).outputVariable();
  }

  /**
   * Create a mel-scale triangular filterbank matrix.<br>
   * Maps linear frequency spectrogram bins to mel-frequency bins using<br>
   * overlapping triangular filters. Used as a component of mel spectrogram and MFCC extraction.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param numMelBins Number of mel frequency bins
   * @param fftSize FFT size (number of frequency bins in the spectrogram)
   * @param sampleRate Audio sample rate in Hz
   * @param lowerEdgeHz Lower edge of the mel band in Hz
   * @param upperEdgeHz Upper edge of the mel band in Hz
   * @return output Mel filterbank matrix of shape [numMelBins, fftSize/2+1] (NUMERIC type)
   */
  public SDVariable melFilterbank(String name, int numMelBins, int fftSize, int sampleRate,
      double lowerEdgeHz, double upperEdgeHz) {
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.MelFilterbank(sd,numMelBins, fftSize, sampleRate, lowerEdgeHz, upperEdgeHz).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable melFilterbank(int numMelBins, int fftSize, int sampleRate) {
    return new org.nd4j.linalg.api.ops.impl.audio.MelFilterbank(sd,numMelBins, fftSize, sampleRate, 0.0, 8000.0).outputVariable();
  }

  /**
   * Create a mel-scale triangular filterbank matrix.<br>
   * Maps linear frequency spectrogram bins to mel-frequency bins using<br>
   * overlapping triangular filters. Used as a component of mel spectrogram and MFCC extraction.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param numMelBins Number of mel frequency bins
   * @param fftSize FFT size (number of frequency bins in the spectrogram)
   * @param sampleRate Audio sample rate in Hz
   * @return output Mel filterbank matrix of shape [numMelBins, fftSize/2+1] (NUMERIC type)
   */
  public SDVariable melFilterbank(String name, int numMelBins, int fftSize, int sampleRate) {
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.MelFilterbank(sd,numMelBins, fftSize, sampleRate, 0.0, 8000.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable melSpectrogram(SDVariable input, int sampleRate, int fftSize, int hopLength,
      int numMelBins, double lowerEdgeHz, double upperEdgeHz, double power) {
    SDValidation.validateNumerical("melSpectrogram", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.MelSpectrogram(sd,input, sampleRate, fftSize, hopLength, numMelBins, lowerEdgeHz, upperEdgeHz, power).outputVariable();
  }

  /**
   * Compute mel spectrogram from audio waveform.<br>
   * Applies STFT, converts to power/amplitude spectrogram, then maps through<br>
   * a mel filterbank. This is the standard front-end for many audio ML models.<br>
   *
   * @param name name May be null. Name for the output variable
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
  public SDVariable melSpectrogram(String name, SDVariable input, int sampleRate, int fftSize,
      int hopLength, int numMelBins, double lowerEdgeHz, double upperEdgeHz, double power) {
    SDValidation.validateNumerical("melSpectrogram", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.MelSpectrogram(sd,input, sampleRate, fftSize, hopLength, numMelBins, lowerEdgeHz, upperEdgeHz, power).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Compute mel spectrogram from audio waveform.<br>
   * Applies STFT, converts to power/amplitude spectrogram, then maps through<br>
   * a mel filterbank. This is the standard front-end for many audio ML models.<br>
   *
   * @param input Audio waveform tensor of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Mel spectrogram of shape [batch, numMelBins, numFrames] (NUMERIC type)
   */
  public SDVariable melSpectrogram(SDVariable input) {
    SDValidation.validateNumerical("melSpectrogram", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.MelSpectrogram(sd,input, 22050, 2048, 512, 128, 0.0, 8000.0, 2.0).outputVariable();
  }

  /**
   * Compute mel spectrogram from audio waveform.<br>
   * Applies STFT, converts to power/amplitude spectrogram, then maps through<br>
   * a mel filterbank. This is the standard front-end for many audio ML models.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform tensor of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Mel spectrogram of shape [batch, numMelBins, numFrames] (NUMERIC type)
   */
  public SDVariable melSpectrogram(String name, SDVariable input) {
    SDValidation.validateNumerical("melSpectrogram", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.MelSpectrogram(sd,input, 22050, 2048, 512, 128, 0.0, 8000.0, 2.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable mfcc(SDVariable input, int sampleRate, int fftSize, int hopLength,
      int numMelBins, int numMfcc, double lowerEdgeHz, double upperEdgeHz) {
    SDValidation.validateNumerical("mfcc", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.MFCC(sd,input, sampleRate, fftSize, hopLength, numMelBins, numMfcc, lowerEdgeHz, upperEdgeHz).outputVariable();
  }

  /**
   * Compute Mel-Frequency Cepstral Coefficients (MFCCs) from audio waveform.<br>
   * Applies mel spectrogram extraction, log scaling, and DCT-II to produce<br>
   * cepstral coefficients. MFCCs are widely used features for speech and audio recognition.<br>
   *
   * @param name name May be null. Name for the output variable
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
  public SDVariable mfcc(String name, SDVariable input, int sampleRate, int fftSize, int hopLength,
      int numMelBins, int numMfcc, double lowerEdgeHz, double upperEdgeHz) {
    SDValidation.validateNumerical("mfcc", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.MFCC(sd,input, sampleRate, fftSize, hopLength, numMelBins, numMfcc, lowerEdgeHz, upperEdgeHz).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Compute Mel-Frequency Cepstral Coefficients (MFCCs) from audio waveform.<br>
   * Applies mel spectrogram extraction, log scaling, and DCT-II to produce<br>
   * cepstral coefficients. MFCCs are widely used features for speech and audio recognition.<br>
   *
   * @param input Audio waveform tensor of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output MFCC coefficients of shape [batch, numMfcc, numFrames] (NUMERIC type)
   */
  public SDVariable mfcc(SDVariable input) {
    SDValidation.validateNumerical("mfcc", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.MFCC(sd,input, 22050, 2048, 512, 128, 13, 0.0, 8000.0).outputVariable();
  }

  /**
   * Compute Mel-Frequency Cepstral Coefficients (MFCCs) from audio waveform.<br>
   * Applies mel spectrogram extraction, log scaling, and DCT-II to produce<br>
   * cepstral coefficients. MFCCs are widely used features for speech and audio recognition.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform tensor of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output MFCC coefficients of shape [batch, numMfcc, numFrames] (NUMERIC type)
   */
  public SDVariable mfcc(String name, SDVariable input) {
    SDValidation.validateNumerical("mfcc", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.MFCC(sd,input, 22050, 2048, 512, 128, 13, 0.0, 8000.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable pitchDetection(SDVariable input, int sampleRate, int frameLength, int hopLength,
      double minFreq, double maxFreq) {
    SDValidation.validateNumerical("pitchDetection", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.PitchDetection(sd,input, sampleRate, frameLength, hopLength, minFreq, maxFreq).outputVariable();
  }

  /**
   * Detect fundamental frequency (pitch) using autocorrelation method.<br>
   * Estimates the fundamental frequency of an audio signal per frame by finding<br>
   * the peak of the autocorrelation function within the expected frequency range.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param frameLength Length of each analysis frame
   * @param hopLength Number of samples between successive frames
   * @param minFreq Minimum detectable frequency in Hz
   * @param maxFreq Maximum detectable frequency in Hz
   * @return output Detected fundamental frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable pitchDetection(String name, SDVariable input, int sampleRate, int frameLength,
      int hopLength, double minFreq, double maxFreq) {
    SDValidation.validateNumerical("pitchDetection", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.PitchDetection(sd,input, sampleRate, frameLength, hopLength, minFreq, maxFreq).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Detect fundamental frequency (pitch) using autocorrelation method.<br>
   * Estimates the fundamental frequency of an audio signal per frame by finding<br>
   * the peak of the autocorrelation function within the expected frequency range.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Detected fundamental frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable pitchDetection(SDVariable input) {
    SDValidation.validateNumerical("pitchDetection", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.PitchDetection(sd,input, 22050, 2048, 512, 80.0, 1000.0).outputVariable();
  }

  /**
   * Detect fundamental frequency (pitch) using autocorrelation method.<br>
   * Estimates the fundamental frequency of an audio signal per frame by finding<br>
   * the peak of the autocorrelation function within the expected frequency range.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Detected fundamental frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable pitchDetection(String name, SDVariable input) {
    SDValidation.validateNumerical("pitchDetection", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.PitchDetection(sd,input, 22050, 2048, 512, 80.0, 1000.0).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable preEmphasis(SDVariable input, double coefficient) {
    SDValidation.validateNumerical("preEmphasis", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.PreEmphasis(sd,input, coefficient).outputVariable();
  }

  /**
   * Apply pre-emphasis filter to an audio signal.<br>
   * Computes y[n] = x[n] - coefficient * x[n-1], a first-order high-pass filter<br>
   * that amplifies high frequencies. Standard preprocessing for speech recognition.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param coefficient Pre-emphasis coefficient (typically 0.95-0.97)
   * @return output Pre-emphasized signal with same shape as input (NUMERIC type)
   */
  public SDVariable preEmphasis(String name, SDVariable input, double coefficient) {
    SDValidation.validateNumerical("preEmphasis", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.PreEmphasis(sd,input, coefficient).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Apply pre-emphasis filter to an audio signal.<br>
   * Computes y[n] = x[n] - coefficient * x[n-1], a first-order high-pass filter<br>
   * that amplifies high frequencies. Standard preprocessing for speech recognition.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Pre-emphasized signal with same shape as input (NUMERIC type)
   */
  public SDVariable preEmphasis(SDVariable input) {
    SDValidation.validateNumerical("preEmphasis", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.PreEmphasis(sd,input, 0.97).outputVariable();
  }

  /**
   * Apply pre-emphasis filter to an audio signal.<br>
   * Computes y[n] = x[n] - coefficient * x[n-1], a first-order high-pass filter<br>
   * that amplifies high frequencies. Standard preprocessing for speech recognition.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Pre-emphasized signal with same shape as input (NUMERIC type)
   */
  public SDVariable preEmphasis(String name, SDVariable input) {
    SDValidation.validateNumerical("preEmphasis", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.PreEmphasis(sd,input, 0.97).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable spectralCentroid(SDVariable input, int sampleRate, int fftSize) {
    SDValidation.validateNumerical("spectralCentroid", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.SpectralCentroid(sd,input, sampleRate, fftSize).outputVariable();
  }

  /**
   * Compute the spectral centroid of a magnitude spectrogram.<br>
   * The spectral centroid is the weighted mean of frequencies by their magnitudes,<br>
   * indicating the "center of mass" of the spectrum. It is a measure of spectral brightness.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param fftSize FFT window size used to produce the spectrogram
   * @return output Spectral centroid per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable spectralCentroid(String name, SDVariable input, int sampleRate, int fftSize) {
    SDValidation.validateNumerical("spectralCentroid", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.SpectralCentroid(sd,input, sampleRate, fftSize).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Compute the spectral centroid of a magnitude spectrogram.<br>
   * The spectral centroid is the weighted mean of frequencies by their magnitudes,<br>
   * indicating the "center of mass" of the spectrum. It is a measure of spectral brightness.<br>
   *
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Spectral centroid per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable spectralCentroid(SDVariable input) {
    SDValidation.validateNumerical("spectralCentroid", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.SpectralCentroid(sd,input, 22050, 2048).outputVariable();
  }

  /**
   * Compute the spectral centroid of a magnitude spectrogram.<br>
   * The spectral centroid is the weighted mean of frequencies by their magnitudes,<br>
   * indicating the "center of mass" of the spectrum. It is a measure of spectral brightness.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Spectral centroid per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable spectralCentroid(String name, SDVariable input) {
    SDValidation.validateNumerical("spectralCentroid", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.SpectralCentroid(sd,input, 22050, 2048).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable spectralRolloff(SDVariable input, int sampleRate, int fftSize,
      double rolloffPercent) {
    SDValidation.validateNumerical("spectralRolloff", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.SpectralRolloff(sd,input, sampleRate, fftSize, rolloffPercent).outputVariable();
  }

  /**
   * Compute the spectral rolloff frequency.<br>
   * The rolloff frequency is the frequency below which a specified percentage<br>
   * of the total spectral energy falls. Useful for distinguishing voiced/unvoiced speech.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @param sampleRate Audio sample rate in Hz
   * @param fftSize FFT window size used to produce the spectrogram
   * @param rolloffPercent Percentage of spectral energy (0.0 to 1.0)
   * @return output Spectral rolloff frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable spectralRolloff(String name, SDVariable input, int sampleRate, int fftSize,
      double rolloffPercent) {
    SDValidation.validateNumerical("spectralRolloff", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.SpectralRolloff(sd,input, sampleRate, fftSize, rolloffPercent).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Compute the spectral rolloff frequency.<br>
   * The rolloff frequency is the frequency below which a specified percentage<br>
   * of the total spectral energy falls. Useful for distinguishing voiced/unvoiced speech.<br>
   *
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Spectral rolloff frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable spectralRolloff(SDVariable input) {
    SDValidation.validateNumerical("spectralRolloff", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.SpectralRolloff(sd,input, 22050, 2048, 0.85).outputVariable();
  }

  /**
   * Compute the spectral rolloff frequency.<br>
   * The rolloff frequency is the frequency below which a specified percentage<br>
   * of the total spectral energy falls. Useful for distinguishing voiced/unvoiced speech.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Magnitude spectrogram of shape [batch, freqBins, numFrames] (NUMERIC type)
   * @return output Spectral rolloff frequency per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable spectralRolloff(String name, SDVariable input) {
    SDValidation.validateNumerical("spectralRolloff", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.SpectralRolloff(sd,input, 22050, 2048, 0.85).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
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
  public SDVariable zeroCrossingRate(SDVariable input, int frameLength, int hopLength) {
    SDValidation.validateNumerical("zeroCrossingRate", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.ZeroCrossingRate(sd,input, frameLength, hopLength).outputVariable();
  }

  /**
   * Compute the zero crossing rate of an audio signal.<br>
   * The zero crossing rate is the rate at which the signal changes sign,<br>
   * computed per frame. Useful for speech/music discrimination and onset detection.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @param frameLength Length of each analysis frame
   * @param hopLength Number of samples between successive frames
   * @return output Zero crossing rate per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable zeroCrossingRate(String name, SDVariable input, int frameLength,
      int hopLength) {
    SDValidation.validateNumerical("zeroCrossingRate", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.ZeroCrossingRate(sd,input, frameLength, hopLength).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Compute the zero crossing rate of an audio signal.<br>
   * The zero crossing rate is the rate at which the signal changes sign,<br>
   * computed per frame. Useful for speech/music discrimination and onset detection.<br>
   *
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Zero crossing rate per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable zeroCrossingRate(SDVariable input) {
    SDValidation.validateNumerical("zeroCrossingRate", "input", input);
    return new org.nd4j.linalg.api.ops.impl.audio.ZeroCrossingRate(sd,input, 2048, 512).outputVariable();
  }

  /**
   * Compute the zero crossing rate of an audio signal.<br>
   * The zero crossing rate is the rate at which the signal changes sign,<br>
   * computed per frame. Useful for speech/music discrimination and onset detection.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Audio waveform of shape [batch, samples] or [samples] (NUMERIC type)
   * @return output Zero crossing rate per frame of shape [batch, numFrames] (NUMERIC type)
   */
  public SDVariable zeroCrossingRate(String name, SDVariable input) {
    SDValidation.validateNumerical("zeroCrossingRate", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.audio.ZeroCrossingRate(sd,input, 2048, 512).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}
