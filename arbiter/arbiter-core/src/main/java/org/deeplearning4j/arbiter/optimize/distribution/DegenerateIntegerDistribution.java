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

package org.deeplearning4j.arbiter.optimize.distribution;

import org.apache.commons.math3.distribution.IntegerDistribution;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.apache.commons.math3.exception.OutOfRangeException;

/**
 * Degenerate distribution: i.e., integer "distribution" that is just a fixed value
 */
public class DegenerateIntegerDistribution implements IntegerDistribution {
    private int value;

    public DegenerateIntegerDistribution(int value) {
        this.value = value;
    }


    @Override
    public double probability(int x) {
        return (x == value ? 1.0 : 0.0);
    }

    @Override
    public double cumulativeProbability(int x) {
        return (x >= value ? 1.0 : 0.0);
    }

    @Override
    public double cumulativeProbability(int x0, int x1) throws NumberIsTooLargeException {
        return (value >= x0 && value <= x1 ? 1.0 : 0.0);
    }

    @Override
    public int inverseCumulativeProbability(double p) throws OutOfRangeException {
        throw new UnsupportedOperationException();
    }

    @Override
    public double getNumericalMean() {
        return value;
    }

    @Override
    public double getNumericalVariance() {
        return 0;
    }

    @Override
    public int getSupportLowerBound() {
        return value;
    }

    @Override
    public int getSupportUpperBound() {
        return value;
    }

    @Override
    public boolean isSupportConnected() {
        return true;
    }

    @Override
    public void reseedRandomGenerator(long seed) {
        //no op
    }

    @Override
    public int sample() {
        return value;
    }

    @Override
    public int[] sample(int sampleSize) {
        int[] out = new int[sampleSize];
        for (int i = 0; i < out.length; i++)
            out[i] = value;
        return out;
    }
}
