/*-
 *  * Copyright 2016 Skymind, Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 */

package org.datavec.audio.dsp;

import org.jtransforms.fft.DoubleFFT_1D;

/**
 * FFT object, transform amplitudes to frequency intensities
 *
 * @author Jacquet Wong
 */
public class FastFourierTransform {

    /**
     * Get the frequency intensities
     *
     * @param amplitudes amplitudes of the signal. Format depends on value of complex
     * @param complex    if true, amplitudes is assumed to be complex interlaced (re = even, im = odd), if false amplitudes
     *                   are assumed to be real valued.
     * @return intensities of each frequency unit: mag[frequency_unit]=intensity
     */
    public double[] getMagnitudes(double[] amplitudes, boolean complex) {

        final int sampleSize = amplitudes.length;
        final int nrofFrequencyBins = sampleSize / 2;


        // call the fft and transform the complex numbers
        if (complex) {
            DoubleFFT_1D fft = new DoubleFFT_1D(nrofFrequencyBins);
            fft.complexForward(amplitudes);
        } else {
            DoubleFFT_1D fft = new DoubleFFT_1D(sampleSize);
            fft.realForward(amplitudes);
            // amplitudes[1] contains re[sampleSize/2] or im[(sampleSize-1) / 2] (depending on whether sampleSize is odd or even)
            // Discard it as it is useless without the other part
            // im part dc bin is always 0 for real input
            amplitudes[1] = 0;
        }
        // end call the fft and transform the complex numbers

        // even indexes (0,2,4,6,...) are real parts
        // odd indexes (1,3,5,7,...) are img parts
        double[] mag = new double[nrofFrequencyBins];
        for (int i = 0; i < nrofFrequencyBins; i++) {
            final int f = 2 * i;
            mag[i] = Math.sqrt(amplitudes[f] * amplitudes[f] + amplitudes[f + 1] * amplitudes[f + 1]);
        }

        return mag;
    }

    /**
     * Get the frequency intensities. Backwards compatible with previous versions w.r.t to number of frequency bins.
     * Use getMagnitudes(amplitudes, true) to get all bins.
     *
     * @param amplitudes complex-valued signal to transform. Even indexes are real and odd indexes are img
     * @return intensities of each frequency unit: mag[frequency_unit]=intensity
     */
    public double[] getMagnitudes(double[] amplitudes) {
        double[] magnitudes = getMagnitudes(amplitudes, true);

        double[] halfOfMagnitudes = new double[magnitudes.length/2];
        System.arraycopy(magnitudes, 0,halfOfMagnitudes, 0, halfOfMagnitudes.length);
        return halfOfMagnitudes;
    }

}
