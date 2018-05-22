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

package org.datavec.spark.transform.analysis.columns;

import com.tdunning.math.stats.TDigest;
import lombok.AllArgsConstructor;
import lombok.Data;
import org.apache.spark.util.StatCounter;
import org.datavec.api.writable.Writable;
import org.datavec.spark.transform.analysis.AnalysisCounter;

/**
 * A counter function for doing analysis on integer columns, on Spark
 *
 * @author Alex Black
 */
@AllArgsConstructor
@Data
public class IntegerAnalysisCounter implements AnalysisCounter<IntegerAnalysisCounter> {

    private StatCounter counter = new StatCounter();
    private long countZero = 0;
    private long countMinValue = 0;
    private long countMaxValue = 0;
    private long countPositive = 0;
    private long countNegative = 0;
    /**
     * A histogram structure that will record a sketch of a distribution.
     *
     * The compression argument regulates how accuracy should be traded for size? A value of N here
     * will give quantile errors almost always less than 3/N with considerably smaller errors expected
     * for extreme quantiles. Conversely, you should expect to track about 5 N centroids for this
     * accuracy.
     */
    private TDigest digest = TDigest.createDigest(100);

    public IntegerAnalysisCounter() {};

    public int getMinValueSeen() {
        return (int) counter.min();
    };

    public int getMaxValueSeen() {
        return (int) counter.max();
    };

    public long getSum() {
        return (long) counter.sum();
    };

    public long getCountTotal() {
        return counter.count();
    };

    public double getSampleStdev() {
        return counter.sampleStdev();
    };

    public double getMean() {
        return counter.mean();
    }

    public double getSampleVariance() {
        return counter.sampleVariance();
    }

    @Override
    public IntegerAnalysisCounter add(Writable writable) {
        int value = writable.toInt();

        if (value == 0)
            countZero++;

        if (value == getMinValueSeen())
            countMinValue++;
        else if (value < getMinValueSeen()) {
            countMinValue = 1;
        }

        if (value == getMaxValueSeen())
            countMaxValue++;
        else if (value > getMaxValueSeen()) {
            countMaxValue = 1;
        }

        if (value >= 0) {
            countPositive++;
        } else {
            countNegative++;
        } ;

        digest.add((double) value);

        counter.merge((double) value);

        return this;
    }

    public IntegerAnalysisCounter merge(IntegerAnalysisCounter other) {
        int otherMin = other.getMinValueSeen();
        long newCountMinValue;
        if (getMinValueSeen() == otherMin) {
            newCountMinValue = countMinValue + other.getCountMinValue();
        } else if (getMinValueSeen() > otherMin) {
            //Keep other, take count from other
            newCountMinValue = other.getCountMinValue();
        } else {
            //Keep this min, no change to count
            newCountMinValue = countMinValue;
        }

        int otherMax = other.getMaxValueSeen();
        long newCountMaxValue;
        if (getMaxValueSeen() == otherMax) {
            newCountMaxValue = countMaxValue + other.getCountMaxValue();
        } else if (getMaxValueSeen() < otherMax) {
            //Keep other, take count from other
            newCountMaxValue = other.getCountMaxValue();
        } else {
            //Keep this max, no change to count
            newCountMaxValue = countMaxValue;
        }

        digest.add(other.getDigest());

        return new IntegerAnalysisCounter(counter.merge(other.getCounter()), countZero + other.getCountZero(),
                        newCountMinValue, newCountMaxValue, countPositive + other.getCountPositive(),
                        countNegative + other.getCountNegative(), digest);
    }
}
