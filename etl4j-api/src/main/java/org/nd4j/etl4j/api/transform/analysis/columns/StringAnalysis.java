package org.nd4j.etl4j.api.transform.analysis.columns;

import io.skymind.echidna.api.ColumnType;
import lombok.Data;

/**
 * Analysis for String columns
 *
 * @author Alex Black
 */
@Data
public class StringAnalysis implements ColumnAnalysis {

    private final long countUnique;
    private final int minLength;
    private final int maxLength;
    private final double meanLength;
    private final double sampleStdevLength;
    private final double sampleVarianceLength;
    private final long countTotal;
    private double[] histogramBuckets;
    private long[] histogramBucketCounts;

    private StringAnalysis(Builder builder) {
        this.countUnique = builder.countUnique;
        this.minLength = builder.minLength;
        this.maxLength = builder.maxLength;
        this.meanLength = builder.meanLength;
        this.sampleStdevLength = builder.sampleStdevLength;
        this.sampleVarianceLength = builder.sampleVarianceLength;
        this.countTotal = builder.countTotal;
        this.histogramBuckets = builder.histogramBuckets;
        this.histogramBucketCounts = builder.histogramBucketCounts;
    }

    @Override
    public String toString() {
        return "StringAnalysis(unique=" + countUnique + ",minLen=" + minLength + ",maxLen=" + maxLength + ",meanLen=" + meanLength +
                ",sampleStDevLen=" + sampleStdevLength + ",sampleVarianceLen=" + sampleVarianceLength + ",count=" + countTotal + ")";
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.String;
    }

    public static class Builder {

        private long countUnique;
        private int minLength;
        private int maxLength;
        private double meanLength;
        private double sampleStdevLength;
        private double sampleVarianceLength;
        private long countTotal;
        private double[] histogramBuckets;
        private long[] histogramBucketCounts;

        public Builder countUnique(long countUnique) {
            this.countUnique = countUnique;
            return this;
        }

        public Builder minLength(int minLength) {
            this.minLength = minLength;
            return this;
        }

        public Builder maxLength(int maxLength) {
            this.maxLength = maxLength;
            return this;
        }

        public Builder meanLength(double meanLength) {
            this.meanLength = meanLength;
            return this;
        }

        public Builder sampleStdevLength(double sampleStdevLength) {
            this.sampleStdevLength = sampleStdevLength;
            return this;
        }

        public Builder sampleVarianceLength(double sampleVarianceLength) {
            this.sampleVarianceLength = sampleVarianceLength;
            return this;
        }

        public Builder countTotal(long countTotal) {
            this.countTotal = countTotal;
            return this;
        }

        public Builder histogramBuckets(double[] histogramBuckets) {
            this.histogramBuckets = histogramBuckets;
            return this;
        }

        public Builder histogramBucketCounts(long[] histogramBucketCounts) {
            this.histogramBucketCounts = histogramBucketCounts;
            return this;
        }

        public StringAnalysis build() {
            return new StringAnalysis(this);
        }
    }

}
