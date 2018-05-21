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

/**
 * Construct new data points within the range of a discrete set of known data points by linear equation
 * 
 * @author Jacquet Wong
 */
public class LinearInterpolation {

    public LinearInterpolation() {

    }

    /**
     * Do interpolation on the samples according to the original and destinated sample rates
     * 
     * @param oldSampleRate	sample rate of the original samples
     * @param newSampleRate	sample rate of the interpolated samples
     * @param samples	original samples
     * @return interpolated samples
     */
    public short[] interpolate(int oldSampleRate, int newSampleRate, short[] samples) {

        if (oldSampleRate == newSampleRate) {
            return samples;
        }

        int newLength = Math.round(((float) samples.length / oldSampleRate * newSampleRate));
        float lengthMultiplier = (float) newLength / samples.length;
        short[] interpolatedSamples = new short[newLength];

        // interpolate the value by the linear equation y=mx+c        
        for (int i = 0; i < newLength; i++) {

            // get the nearest positions for the interpolated point
            float currentPosition = i / lengthMultiplier;
            int nearestLeftPosition = (int) currentPosition;
            int nearestRightPosition = nearestLeftPosition + 1;
            if (nearestRightPosition >= samples.length) {
                nearestRightPosition = samples.length - 1;
            }

            float slope = samples[nearestRightPosition] - samples[nearestLeftPosition]; // delta x is 1
            float positionFromLeft = currentPosition - nearestLeftPosition;

            interpolatedSamples[i] = (short) (slope * positionFromLeft + samples[nearestLeftPosition]); // y=mx+c
        }

        return interpolatedSamples;
    }
}
