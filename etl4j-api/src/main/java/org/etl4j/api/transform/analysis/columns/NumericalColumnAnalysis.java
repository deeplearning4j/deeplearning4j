package io.skymind.echidna.api.analysis.columns;

import lombok.Data;

/**
 * Abstract class for numerical column analysis
 *
 * @author Alex Black
 */
@Data
public abstract class NumericalColumnAnalysis implements ColumnAnalysis {

    protected final double mean;
    protected final double sampleStdev;
    protected final double sampleVariance;
    protected final long countZero;
    protected final long countNegative;
    protected final long countPositive;
    protected final long countMinValue;
    protected final long countMaxValue;
    protected final long countTotal;
    protected double[] histogramBuckets;
    protected long[] histogramBucketCounts;

    protected NumericalColumnAnalysis(Builder builder) {
        this.mean = builder.mean;
        this.sampleStdev = builder.sampleStdev;
        this.sampleVariance = builder.sampleVariance;
        this.countZero = builder.countZero;
        this.countNegative = builder.countNegative;
        this.countPositive = builder.countPositive;
        this.countMinValue = builder.countMinValue;
        this.countMaxValue = builder.countMaxValue;
        this.countTotal = builder.countTotal;
        this.histogramBuckets = builder.histogramBuckets;
        this.histogramBucketCounts = builder.histogramBucketCounts;
    }

    @Override
    public String toString() {
        return "mean=" + mean + ",sampleStDev=" + sampleStdev + ",sampleVariance=" + sampleVariance + ",countZero=" + countZero
                + ",countNegative=" + countNegative + ",countPositive=" + countPositive + ",countMinValue=" + countMinValue
                + ",countMaxValue=" + countMaxValue + ",count=" + countTotal;
    }

    public abstract double getMinDouble();

    public abstract double getMaxDouble();

    @SuppressWarnings("unchecked")
    public abstract static class Builder<T extends Builder<T>> {
        protected double mean;
        protected double sampleStdev;
        protected double sampleVariance;
        protected long countZero;
        protected long countNegative;
        protected long countPositive;
        protected long countMinValue;
        protected long countMaxValue;
        protected long countTotal;
        protected double[] histogramBuckets;
        protected long[] histogramBucketCounts;

        public T mean(double mean) {
            this.mean = mean;
            return (T) this;
        }

        public T sampleStdev(double sampleStdev) {
            this.sampleStdev = sampleStdev;
            return (T) this;
        }

        public T sampleVariance(double sampleVariance) {
            this.sampleVariance = sampleVariance;
            return (T) this;
        }

        public T countZero(long countZero) {
            this.countZero = countZero;
            return (T) this;
        }

        public T countNegative(long countNegative) {
            this.countNegative = countNegative;
            return (T) this;
        }

        public T countPositive(long countPositive) {
            this.countPositive = countPositive;
            return (T) this;
        }

        public T countMinValue(long countMinValue) {
            this.countMinValue = countMinValue;
            return (T) this;
        }

        public T countMaxValue(long countMaxValue) {
            this.countMaxValue = countMaxValue;
            return (T) this;
        }

        public T countTotal(long countTotal) {
            this.countTotal = countTotal;
            return (T) this;
        }

        public T histogramBuckets(double[] histogramBuckets) {
            this.histogramBuckets = histogramBuckets;
            return (T) this;
        }

        public T histogramBucketCounts(long[] histogramBucketCounts) {
            this.histogramBucketCounts = histogramBucketCounts;
            return (T) this;
        }
    }

}
