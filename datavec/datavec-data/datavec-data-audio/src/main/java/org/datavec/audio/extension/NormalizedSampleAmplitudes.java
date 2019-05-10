/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

package org.datavec.audio.extension;


import org.datavec.audio.Wave;

/**
 * Handles the wave data in amplitude-time domain.
 *
 * @author Jacquet Wong
 */
public class NormalizedSampleAmplitudes {

    private Wave wave;
    private double[] normalizedAmplitudes; // normalizedAmplitudes[sampleNumber]=normalizedAmplitudeInTheFrame

    public NormalizedSampleAmplitudes(Wave wave) {
        this.wave = wave;
    }

    /**
     * 
     * Get normalized amplitude of each frame
     * 
     * @return	array of normalized amplitudes(signed 16 bit): normalizedAmplitudes[frame]=amplitude
     */
    public double[] getNormalizedAmplitudes() {

        if (normalizedAmplitudes == null) {

            boolean signed = true;

            // usually 8bit is unsigned
            if (wave.getWaveHeader().getBitsPerSample() == 8) {
                signed = false;
            }

            short[] amplitudes = wave.getSampleAmplitudes();
            int numSamples = amplitudes.length;
            int maxAmplitude = 1 << (wave.getWaveHeader().getBitsPerSample() - 1);

            if (!signed) { // one more bit for unsigned value
                maxAmplitude <<= 1;
            }

            normalizedAmplitudes = new double[numSamples];
            for (int i = 0; i < numSamples; i++) {
                normalizedAmplitudes[i] = (double) amplitudes[i] / maxAmplitude;
            }
        }
        return normalizedAmplitudes;
    }
}
