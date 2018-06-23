/*-
 *  * Copyright 2017 Skymind, Inc.
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

package org.datavec.api.transform.analysis.histogram;

import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.api.writable.Writable;
import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * A counter for building histograms on a NDArray columns.
 * This is a bit of a hack, using DoubleHistogramCounter internally. This should (one day) be optimized to use
 * native ND4J operations
 *
 * @author Alex Black
 */
public class NDArrayHistogramCounter implements HistogramCounter {

    private DoubleHistogramCounter underlying;

    public NDArrayHistogramCounter(double minValue, double maxValue, int nBins) {
        this.underlying = new DoubleHistogramCounter(minValue, maxValue, nBins);
    }


    @Override
    public HistogramCounter add(Writable w) {
        INDArray arr = ((NDArrayWritable) w).get();
        if (arr == null) {
            return this;
        }

        long length = arr.length();
        DoubleWritable dw = new DoubleWritable();
        for (int i = 0; i < length; i++) {
            dw.set(arr.getDouble(i));
            underlying.add(dw);
        }

        return this;
    }

    @Override
    public NDArrayHistogramCounter merge(HistogramCounter other) {
        if (other == null)
            return this;
        if (!(other instanceof NDArrayHistogramCounter))
            throw new IllegalArgumentException("Cannot merge " + other.getClass());

        NDArrayHistogramCounter o = (NDArrayHistogramCounter) other;

        if (this.underlying == null) {
            this.underlying = o.underlying;
        } else {
            if (o.underlying == null) {
                return this;
            }
            this.underlying.merge(o.underlying);
        }

        return this;
    }

    @Override
    public double[] getBins() {
        return underlying.getBins();
    }

    @Override
    public long[] getCounts() {
        return underlying.getCounts();
    }
}
