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

package org.datavec.api.transform.analysis.histogram;

import org.datavec.api.writable.Writable;

/**
 * A counter for building histograms on a Double column
 *
 * @author Alex Black
 */
public class DoubleHistogramCounter implements HistogramCounter {

    private final double minValue;
    private final double maxValue;
    private final int nBins;
    private final double[] bins;
    private final long[] binCounts;

    public DoubleHistogramCounter(double minValue, double maxValue, int nBins) {
        this.minValue = minValue;
        this.maxValue = maxValue;
        this.nBins = nBins;

        bins = new double[nBins + 1]; //+1 because bins are defined by a range of values: bins[i] to bins[i+1]
        double step = (maxValue - minValue) / nBins;
        for (int i = 0; i < bins.length; i++) {
            if (i == bins.length - 1)
                bins[i] = maxValue;
            else
                bins[i] = minValue + i * step;
        }

        binCounts = new long[nBins];
    }


    @Override
    public HistogramCounter add(Writable w) {
        double d = w.toDouble();

        //Not super efficient, but linear search on 20-50 items should be good enough
        int idx = -1;
        for (int i = 0; i < nBins; i++) {
            if (d >= bins[i] && d < bins[i + 1]) {
                idx = i;
                break;
            }
        }
        if (idx == -1)
            idx = nBins - 1;

        binCounts[idx]++;

        return this;
    }

    @Override
    public DoubleHistogramCounter merge(HistogramCounter other) {
        if (other == null)
            return this;
        if (!(other instanceof DoubleHistogramCounter))
            throw new IllegalArgumentException("Cannot merge " + other);

        DoubleHistogramCounter o = (DoubleHistogramCounter) other;

        if (minValue != o.minValue || maxValue != o.maxValue)
            throw new IllegalStateException("Min/max values differ: (" + minValue + "," + maxValue + ") " + " vs. ("
                            + o.minValue + "," + o.maxValue + ")");
        if (nBins != o.nBins)
            throw new IllegalStateException("Different number of bins: " + nBins + " vs " + o.nBins);

        for (int i = 0; i < nBins; i++) {
            binCounts[i] += o.binCounts[i];
        }

        return this;
    }

    @Override
    public double[] getBins() {
        return bins;
    }

    @Override
    public long[] getCounts() {
        return binCounts;
    }
}
