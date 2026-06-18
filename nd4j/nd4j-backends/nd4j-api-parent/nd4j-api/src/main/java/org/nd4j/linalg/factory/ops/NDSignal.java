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

public class NDSignal {
  public NDSignal() {
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
  public INDArray blackmanWindow(INDArray size, boolean periodic) {
    NDValidation.validateInteger("blackmanWindow", "size", size);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.BlackmanWindow(size, periodic));
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
   * Generates a Blackman window function.<br>
   * The Blackman window is defined as: w(n) = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @return output Blackman window tensor of shape [size] (NUMERIC type)
   */
  public INDArray blackmanWindow(INDArray size) {
    NDValidation.validateInteger("blackmanWindow", "size", size);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.BlackmanWindow(size, true));
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
  public INDArray dft(INDArray input, int axis, boolean inverse, boolean onesided) {
    NDValidation.validateNumerical("dft", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.DFT(input, axis, inverse, onesided));
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
   * Discrete Fourier Transform operation.<br>
   * Computes the DFT of the input tensor along the specified axis.<br>
   * For real input, can optionally return only positive frequencies (onesided=true).<br>
   *
   * @param input Complex input tensor. Last dimension should be 2 for [real, imag] or treated as real-only (NUMERIC type)
   * @return output DFT output - complex tensor with last dimension 2 for [real, imag] (NUMERIC type)
   */
  public INDArray dft(INDArray input) {
    NDValidation.validateNumerical("dft", "input", input);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.DFT(input, -2, false, false));
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
   * Generates a Hamming window function.<br>
   * The Hamming window is defined as: w(n) = 0.54 - 0.46 * cos(2*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @param periodic If true, generate a periodic window for spectral analysis
   * @return output Hamming window tensor of shape [size] (NUMERIC type)
   */
  public INDArray hammingWindow(INDArray size, boolean periodic) {
    NDValidation.validateInteger("hammingWindow", "size", size);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.HammingWindow(size, periodic));
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
   * Generates a Hamming window function.<br>
   * The Hamming window is defined as: w(n) = 0.54 - 0.46 * cos(2*pi*n/(N-1))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @return output Hamming window tensor of shape [size] (NUMERIC type)
   */
  public INDArray hammingWindow(INDArray size) {
    NDValidation.validateInteger("hammingWindow", "size", size);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.HammingWindow(size, true));
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
   * Generates a Hann window function.<br>
   * The Hann window is defined as: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @param periodic If true, generate a periodic window for spectral analysis
   * @return output Hann window tensor of shape [size] (NUMERIC type)
   */
  public INDArray hannWindow(INDArray size, boolean periodic) {
    NDValidation.validateInteger("hannWindow", "size", size);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.HannWindow(size, periodic));
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
   * Generates a Hann window function.<br>
   * The Hann window is defined as: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))<br>
   * Used for spectral analysis and STFT preprocessing.<br>
   *
   * @param size Window size (INT type)
   * @return output Hann window tensor of shape [size] (NUMERIC type)
   */
  public INDArray hannWindow(INDArray size) {
    NDValidation.validateInteger("hannWindow", "size", size);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.HannWindow(size, true));
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
  public INDArray stft(INDArray signal, INDArray frameStep, INDArray window, INDArray frameLength,
      boolean onesided) {
    NDValidation.validateNumerical("stft", "signal", signal);
    NDValidation.validateInteger("stft", "frameStep", frameStep);
    NDValidation.validateNumerical("stft", "window", window);
    NDValidation.validateInteger("stft", "frameLength", frameLength);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.STFT(signal, frameStep, window, frameLength, onesided));
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
  public INDArray stft(INDArray signal, INDArray frameStep, INDArray window, INDArray frameLength) {
    NDValidation.validateNumerical("stft", "signal", signal);
    NDValidation.validateInteger("stft", "frameStep", frameStep);
    NDValidation.validateNumerical("stft", "window", window);
    NDValidation.validateInteger("stft", "frameLength", frameLength);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.STFT(signal, frameStep, window, frameLength, true));
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
   * Short-Time Fourier Transform operation (simplified version without window/frameLength).<br>
   * Computes STFT by applying DFT to overlapping segments of the input signal.<br>
   *
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @param onesided If true, return only positive frequencies
   * @return output STFT output - complex spectrogram (NUMERIC type)
   */
  public INDArray stftSimple(INDArray signal, INDArray frameStep, boolean onesided) {
    NDValidation.validateNumerical("stftSimple", "signal", signal);
    NDValidation.validateInteger("stftSimple", "frameStep", frameStep);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.STFT(signal, frameStep, onesided));
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
   * Short-Time Fourier Transform operation (simplified version without window/frameLength).<br>
   * Computes STFT by applying DFT to overlapping segments of the input signal.<br>
   *
   * @param signal Input signal tensor (NUMERIC type)
   * @param frameStep Number of samples to step between frames (hop length) (INT type)
   * @return output STFT output - complex spectrogram (NUMERIC type)
   */
  public INDArray stftSimple(INDArray signal, INDArray frameStep) {
    NDValidation.validateNumerical("stftSimple", "signal", signal);
    NDValidation.validateInteger("stftSimple", "frameStep", frameStep);
    INDArray[] __tmp = Nd4j.exec(new org.nd4j.linalg.api.ops.impl.signal.STFT(signal, frameStep, true));
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
