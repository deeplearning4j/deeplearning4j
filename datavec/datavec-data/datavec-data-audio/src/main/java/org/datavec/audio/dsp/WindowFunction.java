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

package org.datavec.audio.dsp;

/**
 * Window functions generator
 * 
 * @author Jacquet Wong
 *
 */
public class WindowFunction {

    public static final int RECTANGULAR = 0;
    public static final int BARTLETT = 1;
    public static final int HANNING = 2;
    public static final int HAMMING = 3;
    public static final int BLACKMAN = 4;

    int windowType = 0; // defaults to rectangular window

    public WindowFunction() {}

    public void setWindowType(int wt) {
        windowType = wt;
    }

    public void setWindowType(String w) {
        if (w.toUpperCase().equals("RECTANGULAR"))
            windowType = RECTANGULAR;
        if (w.toUpperCase().equals("BARTLETT"))
            windowType = BARTLETT;
        if (w.toUpperCase().equals("HANNING"))
            windowType = HANNING;
        if (w.toUpperCase().equals("HAMMING"))
            windowType = HAMMING;
        if (w.toUpperCase().equals("BLACKMAN"))
            windowType = BLACKMAN;
    }

    public int getWindowType() {
        return windowType;
    }

    /**
     * Generate a window
     * 
     * @param nSamples	size of the window
     * @return	window in array
     */
    public double[] generate(int nSamples) {
        // generate nSamples window function values
        // for index values 0 .. nSamples - 1
        int m = nSamples / 2;
        double r;
        double pi = Math.PI;
        double[] w = new double[nSamples];
        switch (windowType) {
            case BARTLETT: // Bartlett (triangular) window
                for (int n = 0; n < nSamples; n++)
                    w[n] = 1.0f - Math.abs(n - m) / m;
                break;
            case HANNING: // Hanning window
                r = pi / (m + 1);
                for (int n = -m; n < m; n++)
                    w[m + n] = 0.5f + 0.5f * Math.cos(n * r);
                break;
            case HAMMING: // Hamming window
                r = pi / m;
                for (int n = -m; n < m; n++)
                    w[m + n] = 0.54f + 0.46f * Math.cos(n * r);
                break;
            case BLACKMAN: // Blackman window
                r = pi / m;
                for (int n = -m; n < m; n++)
                    w[m + n] = 0.42f + 0.5f * Math.cos(n * r) + 0.08f * Math.cos(2 * n * r);
                break;
            default: // Rectangular window function
                for (int n = 0; n < nSamples; n++)
                    w[n] = 1.0f;
        }
        return w;
    }
}
