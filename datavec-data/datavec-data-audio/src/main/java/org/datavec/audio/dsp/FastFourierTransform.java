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

import com.sun.media.sound.FFT;

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
     * @param complex if true, amplitudes is assumed to be complex interlaced (re = even, im = odd), if false amplitudes
     *                are assumed to be real valued.
     * @return intensities of each frequency unit: mag[frequency_unit]=intensity
     */
    public double[] getMagnitudes(double[] amplitudes, boolean complex) {

        if(complex) {
            return getMagnitudes(amplitudes);
        }

        // FFT expects complex input where even indexes are real parts
        // and odd indexes are img parts.
        // Here we assume amplitudes to be real-valued and double the size of
        // the input array and set img indexes to zero
        int sampleSize = 2 * amplitudes.length;
        double[] amplitudesAsComplex = new double[sampleSize];
        for (int j = 0; j < sampleSize; j += 2) {
            amplitudesAsComplex[j] = amplitudes[j / 2];
            amplitudesAsComplex[j + 1] = 0;
        }
        return getMagnitudes(amplitudesAsComplex);
    }

    /**
     * Get the frequency intensities
     *
     * @param amplitudes complex-valued signal to transform. Even indexes are real and odd indexes are img
     * @return intensities of each frequency unit: mag[frequency_unit]=intensity
     */
    public double[] getMagnitudes(double[] amplitudes) {

        int sampleSize = amplitudes.length;

        // call the fft and transform the complex numbers
        FFT fft = new FFT(sampleSize / 2, -1);
        fft.transform(amplitudes);
        // end call the fft and transform the complex numbers

        // even indexes (0,2,4,6,...) are real parts
        // odd indexes (1,3,5,7,...) are img parts
        int indexSize = sampleSize / 2;

        // FFT produces a transformed pair of arrays where the first half of the
        // values represent positive frequency components and the second half
        // represents negative frequency components.
        // we omit the negative ones
        int positiveSize = indexSize / 2;

        double[] mag = new double[positiveSize];
        for (int j = 0; j < indexSize; j += 2) {
            mag[j / 2] = Math.sqrt(amplitudes[j] * amplitudes[j] + amplitudes[j + 1] * amplitudes[j + 1]);
        }

        return mag;
    }

}
