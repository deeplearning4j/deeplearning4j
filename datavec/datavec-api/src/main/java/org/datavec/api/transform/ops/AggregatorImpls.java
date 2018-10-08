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

package org.datavec.api.transform.ops;

import com.clearspring.analytics.stream.cardinality.CardinalityMergeException;
import com.clearspring.analytics.stream.cardinality.HyperLogLogPlus;
import lombok.Getter;
import lombok.NoArgsConstructor;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.UnsafeWritableInjector;
import org.datavec.api.writable.Writable;

/**
 * Created by huitseeker on 4/28/17.
 */
public class AggregatorImpls {

    public static class AggregableFirst<T> implements IAggregableReduceOp<T, Writable> {

        private T elem = null;

        @Override
        public void accept(T element) {
            if (elem == null)
                elem = element;
        }

        @Override
        public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
            // left-favoring for first
            if (!(accu instanceof IAggregableReduceOp))
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        @Override
        public Writable get() {
            return UnsafeWritableInjector.inject(elem);
        }
    }

    public static class AggregableLast<T> implements IAggregableReduceOp<T, Writable> {

        private T elem = null;
        private Writable override = null;

        @Override
        public void accept(T element) {
            if (element != null)
                elem = element;
        }

        @Override
        public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
            if (accu instanceof AggregableLast)
                override = accu.get(); // right-favoring for last
            else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        @Override
        public Writable get() {
            if (override == null)
                return UnsafeWritableInjector.inject(elem);
            else
                return override;
        }
    }

    public static class AggregableSum<T extends Number> implements IAggregableReduceOp<T, Writable> {

        @Getter
        private Number sum;
        @Getter
        private T initialElement; // this value is ignored and jut serves as a subtype indicator

        private static <U extends Number> Number addNumbers(U a, U b) {
            if (a instanceof Double || b instanceof Double) {
                return new Double(a.doubleValue() + b.doubleValue());
            } else if (a instanceof Float || b instanceof Float) {
                return new Float(a.floatValue() + b.floatValue());
            } else if (a instanceof Long || b instanceof Long) {
                return new Long(a.longValue() + b.longValue());
            } else {
                return new Integer(a.intValue() + b.intValue());
            }
        }

        @Override
        public void accept(T element) {
            if (sum == null) {
                sum = element;
                initialElement = element;
            } else {
                if (initialElement.getClass().isAssignableFrom(element.getClass()))
                    sum = addNumbers(sum, element);
            }
        }

        @Override
        public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
            if (accu instanceof AggregableSum) {
                AggregableSum<T> accumulator = (AggregableSum<T>) accu;
                // the type of this now becomes that of the union of initialelement
                if (accumulator.getInitialElement().getClass().isAssignableFrom(initialElement.getClass()))
                    initialElement = accumulator.initialElement;
                sum = addNumbers(sum, accumulator.getSum());
            } else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        @Override
        public Writable get() {
            return UnsafeWritableInjector.inject(sum);
        }
    }

    public static class AggregableProd<T extends Number> implements IAggregableReduceOp<T, Writable> {

        @Getter
        private Number prod;
        @Getter
        private T initialElement; // this value is ignored and jut serves as a subtype indicator

        private static <U extends Number> Number multiplyNumbers(U a, U b) {
            if (a instanceof Double || b instanceof Double) {
                return new Double(a.doubleValue() * b.doubleValue());
            } else if (a instanceof Float || b instanceof Float) {
                return new Float(a.floatValue() * b.floatValue());
            } else if (a instanceof Long || b instanceof Long) {
                return new Long(a.longValue() * b.longValue());
            } else {
                return new Integer(a.intValue() * b.intValue());
            }
        }

        @Override
        public void accept(T element) {
            if (prod == null) {
                prod = element;
                initialElement = element;
            } else {
                if (initialElement.getClass().isAssignableFrom(element.getClass()))
                    prod = multiplyNumbers(prod, element);
            }
        }

        @Override
        public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
            if (accu instanceof AggregableSum) {
                AggregableSum<T> accumulator = (AggregableSum<T>) accu;
                // the type of this now becomes that of the union of initialelement
                if (accumulator.getInitialElement().getClass().isAssignableFrom(initialElement.getClass()))
                    initialElement = accumulator.initialElement;
                prod = multiplyNumbers(prod, accumulator.getSum());
            } else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        @Override
        public Writable get() {
            return UnsafeWritableInjector.inject(prod);
        }
    }

    public static class AggregableMax<T extends Number & Comparable<T>> implements IAggregableReduceOp<T, Writable> {

        @Getter
        private T max = null;

        @Override
        public void accept(T element) {
            if (max == null || max.compareTo(element) < 0)
                max = element;
        }

        @Override
        public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
            if (max == null || (accu instanceof AggregableMax && max.compareTo(((AggregableMax<T>) accu).getMax()) < 0))
                max = ((AggregableMax<T>) accu).getMax();
            else if (!(accu instanceof AggregableMax))
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        @Override
        public Writable get() {
            return UnsafeWritableInjector.inject(max);
        }
    }


    public static class AggregableMin<T extends Number & Comparable<T>> implements IAggregableReduceOp<T, Writable> {

        @Getter
        private T min = null;

        @Override
        public void accept(T element) {
            if (min == null || min.compareTo(element) > 0)
                min = element;
        }

        @Override
        public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
            if (min == null || (accu instanceof AggregableMin && min.compareTo(((AggregableMin<T>) accu).getMin()) > 0))
                min = ((AggregableMin<T>) accu).getMin();
            else if (!(accu instanceof AggregableMin))
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        @Override
        public Writable get() {
            return UnsafeWritableInjector.inject(min);
        }
    }

    public static class AggregableRange<T extends Number & Comparable<T>> implements IAggregableReduceOp<T, Writable> {

        @Getter
        private T min = null;
        @Getter
        private T max = null;

        @Override
        public void accept(T element) {
            if (min == null || min.compareTo(element) > 0)
                min = element;
            if (max == null || max.compareTo(element) < 0)
                max = element;
        }

        @Override
        public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
            if (max == null || (accu instanceof AggregableRange
                            && max.compareTo(((AggregableRange<T>) accu).getMax()) < 0))
                max = ((AggregableRange<T>) accu).getMax();
            if (min == null || (accu instanceof AggregableRange
                            && min.compareTo(((AggregableRange<T>) accu).getMin()) > 0))
                min = ((AggregableRange<T>) accu).getMin();
            if (!(accu instanceof AggregableRange))
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }


        @Override
        public Writable get() {
            if (min.getClass() == Long.class)
                return UnsafeWritableInjector.inject(max.longValue() - min.longValue());
            else if (min.getClass() == Integer.class)
                return UnsafeWritableInjector.inject(max.intValue() - min.intValue());
            else if (min.getClass() == Float.class)
                return UnsafeWritableInjector.inject(max.floatValue() - min.floatValue());
            else if (min.getClass() == Double.class)
                return UnsafeWritableInjector.inject(max.doubleValue() - min.doubleValue());
            else if (min.getClass() == Byte.class)
                return UnsafeWritableInjector.inject(max.byteValue() - min.byteValue());
            else
                throw new IllegalArgumentException(
                                "Wrong type for Aggregable Range operation " + min.getClass().getName());
        }
    }


    public static class AggregableCount<T> implements IAggregableReduceOp<T, Writable> {

        private Long count = 0L;

        @Override
        public void accept(T element) {
            count += 1L;
        }

        @Override
        public <W extends IAggregableReduceOp<T, Writable>> void combine(W accu) {
            if (accu instanceof AggregableCount)
                count = count + accu.get().toLong();
            else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + accu.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        @Override
        public Writable get() {
            return new LongWritable(count);
        }
    }

    public static class AggregableMean<T extends Number> implements IAggregableReduceOp<T, Writable> {

        @Getter
        private Long count = 0L;
        private Double mean = 0D;


        public void accept(T n) {

            // See Knuth TAOCP vol 2, 3rd edition, page 232
            if (count == 0) {
                count = 1L;
                mean = n.doubleValue();
            } else {
                count = count + 1;
                mean = mean + (n.doubleValue() - mean) / count;
            }
        }

        public <U extends IAggregableReduceOp<T, Writable>> void combine(U acc) {
            if (acc instanceof AggregableMean) {
                Long cnt = ((AggregableMean<T>) acc).getCount();
                Long newCount = count + cnt;
                mean = (mean * count + (acc.get().toDouble() * cnt)) / newCount;
                count = newCount;
            } else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + acc.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        public Writable get() {
            return new DoubleWritable(mean);
        }
    }

    /**
     * This class offers an aggregable reduce operation for the unbiased standard deviation, i.e. the estimator
     * of the square root of the arithmetic mean of squared differences to the mean, corrected with Bessel's correction.
     *
     * See <a href="https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation">https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation</a>
     * This is computed with Welford's method for increased numerical stability & aggregability.
     */
    public static class AggregableStdDev<T extends Number> implements IAggregableReduceOp<T, Writable> {

        @Getter
        private Long count = 0L;
        @Getter
        private Double mean = 0D;
        @Getter
        private Double variation = 0D;


        public void accept(T n) {
            if (count == 0) {
                count = 1L;
                mean = n.doubleValue();
                variation = 0D;
            } else {
                Long newCount = count + 1;
                Double newMean = mean + (n.doubleValue() - mean) / newCount;
                Double newvariation = variation + (n.doubleValue() - mean) * (n.doubleValue() - newMean);
                count = newCount;
                mean = newMean;
                variation = newvariation;
            }
        }

        public <U extends IAggregableReduceOp<T, Writable>> void combine(U acc) {
            if (this.getClass().isAssignableFrom(acc.getClass())) {
                AggregableStdDev<T> accu = (AggregableStdDev<T>) acc;

                Long totalCount = count + accu.getCount();
                Double totalMean = (accu.getMean() * accu.getCount() + mean * count) / totalCount;
                // the variance of the union is the sum of variances
                Double variance = variation / (count - 1);
                Double otherVariance = accu.getVariation() / (accu.getCount() - 1);
                Double totalVariation = (variance + otherVariance) * (totalCount - 1);
                count = totalCount;
                mean = totalMean;
                variation = variation;
            } else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + acc.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        public Writable get() {
            return new DoubleWritable(Math.sqrt(variation / (count - 1)));
        }
    }

    /**
     * This class offers an aggregable reduce operation for the biased standard deviation, i.e. the estimator
     * of the square root of the arithmetic mean of squared differences to the mean.
     *
     * See <a href="https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation">https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation</a>
     * This is computed with Welford's method for increased numerical stability & aggregability.
     */
    public static class AggregableUncorrectedStdDev<T extends Number> extends AggregableStdDev<T> {

        @Override
        public Writable get() {
            return new DoubleWritable(Math.sqrt(this.getVariation() / this.getCount()));
        }
    }


    /**
     * This class offers an aggregable reduce operation for the unbiased variance, i.e. the estimator
     * of the arithmetic mean of squared differences to the mean, corrected with Bessel's correction.
     *
     * See <a href="https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation">https://en.wikipedia.org/wiki/Unbiased_estimation_of_standard_deviation</a>
     * This is computed with Welford's method for increased numerical stability & aggregability.
     */
    public static class AggregableVariance<T extends Number> implements IAggregableReduceOp<T, Writable> {

        @Getter
        private Long count = 0L;
        @Getter
        private Double mean = 0D;
        @Getter
        private Double variation = 0D;


        public void accept(T n) {
            if (count == 0) {
                count = 1L;
                mean = n.doubleValue();
                variation = 0D;
            } else {
                Long newCount = count + 1;
                Double newMean = mean + (n.doubleValue() - mean) / newCount;
                Double newvariation = variation + (n.doubleValue() - mean) * (n.doubleValue() - newMean);
                count = newCount;
                mean = newMean;
                variation = newvariation;
            }
        }

        public <U extends IAggregableReduceOp<T, Writable>> void combine(U acc) {
            if (this.getClass().isAssignableFrom(acc.getClass())) {
                AggregableVariance<T> accu = (AggregableVariance<T>) acc;

                Long totalCount = count + accu.getCount();
                Double totalMean = (accu.getMean() * accu.getCount() + mean * count) / totalCount;
                // the variance of the union is the sum of variances
                Double variance = variation / (count - 1);
                Double otherVariance = accu.getVariation() / (accu.getCount() - 1);
                Double totalVariation = (variance + otherVariance) * (totalCount - 1);
                count = totalCount;
                mean = totalMean;
                variation = variation;
            } else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + acc.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        public Writable get() {
            return new DoubleWritable(variation / (count - 1));
        }
    }


    /**
     * This class offers an aggregable reduce operation for the population variance, i.e. the uncorrected estimator
     * of the arithmetic mean of squared differences to the mean.
     *
     * See <a href="https://en.wikipedia.org/wiki/Variance#Population_variance_and_sample_variance">https://en.wikipedia.org/wiki/Variance#Population_variance_and_sample_variance</a>
     * This is computed with Welford's method for increased numerical stability & aggregability.
     */
    public static class AggregablePopulationVariance<T extends Number> extends AggregableVariance<T> {

        @Override
        public Writable get() {
            return new DoubleWritable(this.getVariation() / this.getCount());
        }
    }

    /**
     *
     * This distinct count is based on streamlib's implementation of "HyperLogLog in Practice:
     * Algorithmic Engineering of a State of The Art Cardinality Estimation Algorithm", available
     * <a href="http://dx.doi.org/10.1145/2452376.2452456">here</a>.
     *
     * The relative accuracy is approximately `1.054 / sqrt(2^p)`. Setting
     * a nonzero `sp > p` in HyperLogLogPlus(p, sp) would trigger sparse
     * representation of registers, which may reduce the memory consumption
     * and increase accuracy when the cardinality is small.
     * @param <T>
     */
    @NoArgsConstructor
    public static class AggregableCountUnique<T> implements IAggregableReduceOp<T, Writable> {

        private float p = 0.05f;
        @Getter
        private HyperLogLogPlus hll = new HyperLogLogPlus((int) Math.ceil(2.0 * Math.log(1.054 / p) / Math.log(2)), 0);

        public AggregableCountUnique(float precision) {
            this.p = precision;
        }

        @Override
        public void accept(T element) {
            hll.offer(element);
        }

        @Override
        public <U extends IAggregableReduceOp<T, Writable>> void combine(U acc) {
            if (acc instanceof AggregableCountUnique) {
                try {
                    hll.addAll(((AggregableCountUnique<T>) acc).getHll());
                } catch (CardinalityMergeException e) {
                    throw new RuntimeException(e);
                }
            } else
                throw new UnsupportedOperationException("Tried to combine() incompatible " + acc.getClass().getName()
                                + " operator where " + this.getClass().getName() + " expected");
        }

        @Override
        public Writable get() {
            return new LongWritable(hll.cardinality());
        }
    }
}
