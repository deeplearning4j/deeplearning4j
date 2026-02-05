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

fun SDSignal() = Namespace("Signal") {
    val namespaceJavaPackage = "org.nd4j.linalg.api.ops.impl.signal"

    Op("dft") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "DFT"
        Input(NUMERIC, "input") { description = "Complex input tensor. Last dimension should be 2 for [real, imag] or treated as real-only" }
        Arg(INT, "axis") { description = "Axis along which to compute the DFT"; defaultValue = -2 }
        Arg(BOOL, "inverse") { description = "If true, compute inverse DFT (IDFT)"; defaultValue = false }
        Arg(BOOL, "onesided") { description = "If true, return only the positive frequencies (for real input)"; defaultValue = false }

        Output(NUMERIC, "output") { description = "DFT output - complex tensor with last dimension 2 for [real, imag]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Discrete Fourier Transform operation.
             Computes the DFT of the input tensor along the specified axis.
             For real input, can optionally return only positive frequencies (onesided=true).
            """.trimIndent()
        }
    }

    Op("stft") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "STFT"
        Input(NUMERIC, "signal") { description = "Input signal tensor" }
        Input(INT, "frameStep") { description = "Number of samples to step between frames (hop length)" }
        Input(NUMERIC, "window") { description = "Window function to apply to each frame (optional)" }
        Input(INT, "frameLength") { description = "Length of each frame (optional, defaults to window length or FFT size)" }
        Arg(BOOL, "onesided") { description = "If true, return only positive frequencies"; defaultValue = true }

        Output(NUMERIC, "output") { description = "STFT output - complex spectrogram with shape [batch, frames, freq_bins, 2]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Short-Time Fourier Transform operation.
             Computes STFT by applying DFT to windowed overlapping segments of the input signal.
             Used for time-frequency analysis of signals.
            """.trimIndent()
        }
    }

    Op("stftSimple") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "STFT"
        Input(NUMERIC, "signal") { description = "Input signal tensor" }
        Input(INT, "frameStep") { description = "Number of samples to step between frames (hop length)" }
        Arg(BOOL, "onesided") { description = "If true, return only positive frequencies"; defaultValue = true }

        Output(NUMERIC, "output") { description = "STFT output - complex spectrogram" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Short-Time Fourier Transform operation (simplified version without window/frameLength).
             Computes STFT by applying DFT to overlapping segments of the input signal.
            """.trimIndent()
        }
    }

    Op("hannWindow") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "HannWindow"
        Input(INT, "size") { description = "Window size" }
        Arg(BOOL, "periodic") { description = "If true, generate a periodic window for spectral analysis"; defaultValue = true }

        Output(NUMERIC, "output") { description = "Hann window tensor of shape [size]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Generates a Hann window function.
             The Hann window is defined as: w(n) = 0.5 * (1 - cos(2*pi*n/(N-1)))
             Used for spectral analysis and STFT preprocessing.
            """.trimIndent()
        }
    }

    Op("hammingWindow") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "HammingWindow"
        Input(INT, "size") { description = "Window size" }
        Arg(BOOL, "periodic") { description = "If true, generate a periodic window for spectral analysis"; defaultValue = true }

        Output(NUMERIC, "output") { description = "Hamming window tensor of shape [size]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Generates a Hamming window function.
             The Hamming window is defined as: w(n) = 0.54 - 0.46 * cos(2*pi*n/(N-1))
             Used for spectral analysis and STFT preprocessing.
            """.trimIndent()
        }
    }

    Op("blackmanWindow") {
        javaPackage = namespaceJavaPackage
        javaOpClass = "BlackmanWindow"
        Input(INT, "size") { description = "Window size" }
        Arg(BOOL, "periodic") { description = "If true, generate a periodic window for spectral analysis"; defaultValue = true }

        Output(NUMERIC, "output") { description = "Blackman window tensor of shape [size]" }

        Doc(Language.ANY, DocScope.ALL) {
            """
             Generates a Blackman window function.
             The Blackman window is defined as: w(n) = 0.42 - 0.5*cos(2*pi*n/(N-1)) + 0.08*cos(4*pi*n/(N-1))
             Used for spectral analysis and STFT preprocessing.
            """.trimIndent()
        }
    }
}
