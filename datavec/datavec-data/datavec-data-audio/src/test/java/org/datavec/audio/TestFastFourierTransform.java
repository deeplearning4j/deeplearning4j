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

package org.datavec.audio;

import org.datavec.audio.dsp.FastFourierTransform;
import org.junit.Assert;
import org.junit.Test;

public class TestFastFourierTransform {

    @Test
    public void testFastFourierTransformComplex() {
        FastFourierTransform fft = new FastFourierTransform();
        double[] amplitudes = new double[] {3.0, 4.0, 0.5, 7.8, 6.9, -6.5, 8.5, 4.6};
        double[] frequencies = fft.getMagnitudes(amplitudes);

        Assert.assertEquals(2, frequencies.length);
        Assert.assertArrayEquals(new double[] {21.335, 18.513}, frequencies, 0.005);
    }

    @Test
    public void testFastFourierTransformComplexLong() {
        FastFourierTransform fft = new FastFourierTransform();
        double[] amplitudes = new double[] {3.0, 4.0, 0.5, 7.8, 6.9, -6.5, 8.5, 4.6};
        double[] frequencies = fft.getMagnitudes(amplitudes, true);

        Assert.assertEquals(4, frequencies.length);
        Assert.assertArrayEquals(new double[] {21.335, 18.5132, 14.927, 7.527}, frequencies, 0.005);
    }

    @Test
    public void testFastFourierTransformReal() {
        FastFourierTransform fft = new FastFourierTransform();
        double[] amplitudes = new double[] {3.0, 4.0, 0.5, 7.8, 6.9, -6.5, 8.5, 4.6};
        double[] frequencies = fft.getMagnitudes(amplitudes, false);

        Assert.assertEquals(4, frequencies.length);
        Assert.assertArrayEquals(new double[] {28.8, 2.107, 14.927, 19.874}, frequencies, 0.005);
    }

    @Test
    public void testFastFourierTransformRealOddSize() {
        FastFourierTransform fft = new FastFourierTransform();
        double[] amplitudes = new double[] {3.0, 4.0, 0.5, 7.8, 6.9, -6.5, 8.5};
        double[] frequencies = fft.getMagnitudes(amplitudes, false);

        Assert.assertEquals(3, frequencies.length);
        Assert.assertArrayEquals(new double[] {24.2, 3.861, 16.876}, frequencies, 0.005);
    }
}
