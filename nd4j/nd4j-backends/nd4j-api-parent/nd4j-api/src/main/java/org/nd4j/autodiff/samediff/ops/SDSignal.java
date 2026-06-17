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

public class SDSignal extends SDOps {
  public SDSignal(SameDiff sameDiff) {
    super(sameDiff);
  }

  /**
   * Generates a Blackman window function.<br>
   * The Blackman window is defined as: w(n) = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @param periodic If true, generate a periodic window for spectral analysis
   * @return output Blackman window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable blackmanWindow(SDVariable size, boolean periodic) {
    SDValidation.validateInteger("blackmanWindow", "size", size);
    return new org.nd4j.linalg.api.ops.impl.signal.BlackmanWindow(sd,size, periodic).outputVariable();
  }

  /**
   * Generates a Blackman window function.<br>
   * The Blackman window is defined as: w(n) = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param size Window size (INT type)
   * @param periodic If true, generate a periodic window for spectral analysis
   * @return output Blackman window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable blackmanWindow(String name, SDVariable size, boolean periodic) {
    SDValidation.validateInteger("blackmanWindow", "size", size);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.BlackmanWindow(sd,size, periodic).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Generates a Blackman window function.<br>
   * The Blackman window is defined as: w(n) = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @return output Blackman window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable blackmanWindow(SDVariable size) {
    SDValidation.validateInteger("blackmanWindow", "size", size);
    return new org.nd4j.linalg.api.ops.impl.signal.BlackmanWindow(sd,size, true).outputVariable();
  }

  /**
   * Generates a Blackman window function.<br>
   * The Blackman window is defined as: w(n) = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param size Window size (INT type)
   * @return output Blackman window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable blackmanWindow(String name, SDVariable size) {
    SDValidation.validateInteger("blackmanWindow", "size", size);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.BlackmanWindow(sd,size, true).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Discrete Fourier Transform operation.<br>
   * Computes the DFT of the input tensor along the specified axis.<br>
   * For real input, can optionally return only positive frequencies (onesided=true).<br>
   *
   * @param input Complex input tensor. Last dimension should be 2 for [real, imag] or treated as real-only (NUMERIC type)
   * @param axis Axis along which to compute the DFT
   * @param inverse If true, compute inverse DFT (IDFT)
   * @param onesided If true, return only the positive frequencies (for real input)
   * @return output DFT output - complex tensor with last dimension 2 for [real, imag] (NUMERIC type)
   */
  public SDVariable dft(SDVariable input, int axis, boolean inverse, boolean onesided) {
    SDValidation.validateNumerical("dft", "input", input);
    return new org.nd4j.linalg.api.ops.impl.signal.DFT(sd,input, axis, inverse, onesided).outputVariable();
  }

  /**
   * Discrete Fourier Transform operation.<br>
   * Computes the DFT of the input tensor along the specified axis.<br>
   * For real input, can optionally return only positive frequencies (onesided=true).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Complex input tensor. Last dimension should be 2 for [real, imag] or treated as real-only (NUMERIC type)
   * @param axis Axis along which to compute the DFT
   * @param inverse If true, compute inverse DFT (IDFT)
   * @param onesided If true, return only the positive frequencies (for real input)
   * @return output DFT output - complex tensor with last dimension 2 for [real, imag] (NUMERIC type)
   */
  public SDVariable dft(String name, SDVariable input, int axis, boolean inverse,
      boolean onesided) {
    SDValidation.validateNumerical("dft", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.DFT(sd,input, axis, inverse, onesided).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Discrete Fourier Transform operation.<br>
   * Computes the DFT of the input tensor along the specified axis.<br>
   * For real input, can optionally return only positive frequencies (onesided=true).<br>
   *
   * @param input Complex input tensor. Last dimension should be 2 for [real, imag] or treated as real-only (NUMERIC type)
   * @return output DFT output - complex tensor with last dimension 2 for [real, imag] (NUMERIC type)
   */
  public SDVariable dft(SDVariable input) {
    SDValidation.validateNumerical("dft", "input", input);
    return new org.nd4j.linalg.api.ops.impl.signal.DFT(sd,input, -2, false, false).outputVariable();
  }

  /**
   * Discrete Fourier Transform operation.<br>
   * Computes the DFT of the input tensor along the specified axis.<br>
   * For real input, can optionally return only positive frequencies (onesided=true).<br>
   *
   * @param name name May be null. Name for the output variable
   * @param input Complex input tensor. Last dimension should be 2 for [real, imag] or treated as real-only (NUMERIC type)
   * @return output DFT output - complex tensor with last dimension 2 for [real, imag] (NUMERIC type)
   */
  public SDVariable dft(String name, SDVariable input) {
    SDValidation.validateNumerical("dft", "input", input);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.DFT(sd,input, -2, false, false).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Generates a Hamming window function.<br>
   * The Hamming window is defined as: w(n) = 0.54 - 0.46 * cos(2*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @param periodic If true, generate a periodic window for spectral analysis
   * @return output Hamming window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable hammingWindow(SDVariable size, boolean periodic) {
    SDValidation.validateInteger("hammingWindow", "size", size);
    return new org.nd4j.linalg.api.ops.impl.signal.HammingWindow(sd,size, periodic).outputVariable();
  }

  /**
   * Generates a Hamming window function.<br>
   * The Hamming window is defined as: w(n) = 0.54 - 0.46 * cos(2*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param size Window size (INT type)
   * @param periodic If true, generate a periodic window for spectral analysis
   * @return output Hamming window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable hammingWindow(String name, SDVariable size, boolean periodic) {
    SDValidation.validateInteger("hammingWindow", "size", size);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.HammingWindow(sd,size, periodic).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Generates a Hamming window function.<br>
   * The Hamming window is defined as: w(n) = 0.54 - 0.46 * cos(2*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @return output Hamming window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable hammingWindow(SDVariable size) {
    SDValidation.validateInteger("hammingWindow", "size", size);
    return new org.nd4j.linalg.api.ops.impl.signal.HammingWindow(sd,size, true).outputVariable();
  }

  /**
   * Generates a Hamming window function.<br>
   * The Hamming window is defined as: w(n) = 0.54 - 0.46 * cos(2*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param size Window size (INT type)
   * @return output Hamming window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable hammingWindow(String name, SDVariable size) {
    SDValidation.validateInteger("hammingWindow", "size", size);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.HammingWindow(sd,size, true).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Generates a Hann window function.<br>
   * The Hann window is defined as: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @param periodic If true, generate a periodic window for spectral analysis
   * @return output Hann window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable hannWindow(SDVariable size, boolean periodic) {
    SDValidation.validateInteger("hannWindow", "size", size);
    return new org.nd4j.linalg.api.ops.impl.signal.HannWindow(sd,size, periodic).outputVariable();
  }

  /**
   * Generates a Hann window function.<br>
   * The Hann window is defined as: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param size Window size (INT type)
   * @param periodic If true, generate a periodic window for spectral analysis
   * @return output Hann window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable hannWindow(String name, SDVariable size, boolean periodic) {
    SDValidation.validateInteger("hannWindow", "size", size);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.HannWindow(sd,size, periodic).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Generates a Hann window function.<br>
   * The Hann window is defined as: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @return output Hann window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable hannWindow(SDVariable size) {
    SDValidation.validateInteger("hannWindow", "size", size);
    return new org.nd4j.linalg.api.ops.impl.signal.HannWindow(sd,size, true).outputVariable();
  }

  /**
   * Generates a Hann window function.<br>
   * The Hann window is defined as: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param size Window size (INT type)
   * @return output Hann window tensor of shape [size] (NUMERIC type)
   */
  public SDVariable hannWindow(String name, SDVariable size) {
    SDValidation.validateInteger("hannWindow", "size", size);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.HannWindow(sd,size, true).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Short-Time Fourier Transform operation.<br>
   * Computes STFT by applying DFT to windowed overlapping segments of the input signal.<br>
   * Used for time-frequency analysis of signals.<br>
   *
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @param window Window function to apply to each frame (optional) (NUMERIC type)
   * @param frameLength Length of each frame (optional, defaults to window length or FFT size) (INT type)
   * @param onesided If true, return only positive frequencies
   * @return output STFT output - complex spectrogram with shape [batch, frames, freq_bins, 2] (NUMERIC type)
   */
  public SDVariable stft(SDVariable signal, SDVariable frameStep, SDVariable window,
      SDVariable frameLength, boolean onesided) {
    SDValidation.validateNumerical("stft", "signal", signal);
    SDValidation.validateInteger("stft", "frameStep", frameStep);
    SDValidation.validateNumerical("stft", "window", window);
    SDValidation.validateInteger("stft", "frameLength", frameLength);
    return new org.nd4j.linalg.api.ops.impl.signal.STFT(sd,signal, frameStep, window, frameLength, onesided).outputVariable();
  }

  /**
   * Short-Time Fourier Transform operation.<br>
   * Computes STFT by applying DFT to windowed overlapping segments of the input signal.<br>
   * Used for time-frequency analysis of signals.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @param window Window function to apply to each frame (optional) (NUMERIC type)
   * @param frameLength Length of each frame (optional, defaults to window length or FFT size) (INT type)
   * @param onesided If true, return only positive frequencies
   * @return output STFT output - complex spectrogram with shape [batch, frames, freq_bins, 2] (NUMERIC type)
   */
  public SDVariable stft(String name, SDVariable signal, SDVariable frameStep, SDVariable window,
      SDVariable frameLength, boolean onesided) {
    SDValidation.validateNumerical("stft", "signal", signal);
    SDValidation.validateInteger("stft", "frameStep", frameStep);
    SDValidation.validateNumerical("stft", "window", window);
    SDValidation.validateInteger("stft", "frameLength", frameLength);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.STFT(sd,signal, frameStep, window, frameLength, onesided).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Short-Time Fourier Transform operation.<br>
   * Computes STFT by applying DFT to windowed overlapping segments of the input signal.<br>
   * Used for time-frequency analysis of signals.<br>
   *
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @param window Window function to apply to each frame (optional) (NUMERIC type)
   * @param frameLength Length of each frame (optional, defaults to window length or FFT size) (INT type)
   * @return output STFT output - complex spectrogram with shape [batch, frames, freq_bins, 2] (NUMERIC type)
   */
  public SDVariable stft(SDVariable signal, SDVariable frameStep, SDVariable window,
      SDVariable frameLength) {
    SDValidation.validateNumerical("stft", "signal", signal);
    SDValidation.validateInteger("stft", "frameStep", frameStep);
    SDValidation.validateNumerical("stft", "window", window);
    SDValidation.validateInteger("stft", "frameLength", frameLength);
    return new org.nd4j.linalg.api.ops.impl.signal.STFT(sd,signal, frameStep, window, frameLength, true).outputVariable();
  }

  /**
   * Short-Time Fourier Transform operation.<br>
   * Computes STFT by applying DFT to windowed overlapping segments of the input signal.<br>
   * Used for time-frequency analysis of signals.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @param window Window function to apply to each frame (optional) (NUMERIC type)
   * @param frameLength Length of each frame (optional, defaults to window length or FFT size) (INT type)
   * @return output STFT output - complex spectrogram with shape [batch, frames, freq_bins, 2] (NUMERIC type)
   */
  public SDVariable stft(String name, SDVariable signal, SDVariable frameStep, SDVariable window,
      SDVariable frameLength) {
    SDValidation.validateNumerical("stft", "signal", signal);
    SDValidation.validateInteger("stft", "frameStep", frameStep);
    SDValidation.validateNumerical("stft", "window", window);
    SDValidation.validateInteger("stft", "frameLength", frameLength);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.STFT(sd,signal, frameStep, window, frameLength, true).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Short-Time Fourier Transform operation (simplified version without window/frameLength).<br>
   * Computes STFT by applying DFT to overlapping segments of the input signal.<br>
   *
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @param onesided If true, return only positive frequencies
   * @return output STFT output - complex spectrogram (NUMERIC type)
   */
  public SDVariable stftSimple(SDVariable signal, SDVariable frameStep, boolean onesided) {
    SDValidation.validateNumerical("stftSimple", "signal", signal);
    SDValidation.validateInteger("stftSimple", "frameStep", frameStep);
    return new org.nd4j.linalg.api.ops.impl.signal.STFT(sd,signal, frameStep, onesided).outputVariable();
  }

  /**
   * Short-Time Fourier Transform operation (simplified version without window/frameLength).<br>
   * Computes STFT by applying DFT to overlapping segments of the input signal.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @param onesided If true, return only positive frequencies
   * @return output STFT output - complex spectrogram (NUMERIC type)
   */
  public SDVariable stftSimple(String name, SDVariable signal, SDVariable frameStep,
      boolean onesided) {
    SDValidation.validateNumerical("stftSimple", "signal", signal);
    SDValidation.validateInteger("stftSimple", "frameStep", frameStep);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.STFT(sd,signal, frameStep, onesided).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }

  /**
   * Short-Time Fourier Transform operation (simplified version without window/frameLength).<br>
   * Computes STFT by applying DFT to overlapping segments of the input signal.<br>
   *
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @return output STFT output - complex spectrogram (NUMERIC type)
   */
  public SDVariable stftSimple(SDVariable signal, SDVariable frameStep) {
    SDValidation.validateNumerical("stftSimple", "signal", signal);
    SDValidation.validateInteger("stftSimple", "frameStep", frameStep);
    return new org.nd4j.linalg.api.ops.impl.signal.STFT(sd,signal, frameStep, true).outputVariable();
  }

  /**
   * Short-Time Fourier Transform operation (simplified version without window/frameLength).<br>
   * Computes STFT by applying DFT to overlapping segments of the input signal.<br>
   *
   * @param name name May be null. Name for the output variable
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @return output STFT output - complex spectrogram (NUMERIC type)
   */
  public SDVariable stftSimple(String name, SDVariable signal, SDVariable frameStep) {
    SDValidation.validateNumerical("stftSimple", "signal", signal);
    SDValidation.validateInteger("stftSimple", "frameStep", frameStep);
    SDVariable out =  new org.nd4j.linalg.api.ops.impl.signal.STFT(sd,signal, frameStep, true).outputVariable();
    return sd.updateVariableNameAndReference(out, name);
  }
}
