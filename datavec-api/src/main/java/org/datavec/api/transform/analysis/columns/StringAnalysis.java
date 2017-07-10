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

package org.datavec.api.transform.analysis.columns;

import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;
import org.datavec.api.transform.ColumnType;

/**
 * Analysis for String columns
 *
 * @author Alex Black
 */
@Data
@AllArgsConstructor
@NoArgsConstructor //For Jackson deserialization
public class StringAnalysis implements ColumnAnalysis {
    private int minLength;
    private int maxLength;
    private double meanLength;
    private double sampleStdevLength;
    private double sampleVarianceLength;
    private long countTotal;
    private double[] histogramBuckets;
    private long[] histogramBucketCounts;

    private StringAnalysis(Builder builder) {
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
        return "StringAnalysis(minLen=" + minLength + ",maxLen=" + maxLength + ",meanLen=" + meanLength
                        + ",sampleStDevLen=" + sampleStdevLength + ",sampleVarianceLen=" + sampleVarianceLength
                        + ",count=" + countTotal + ")";
    }

    @Override
    public ColumnType getColumnType() {
        return ColumnType.String;
    }

    public static class Builder {
        private int minLength;
        private int maxLength;
        private double meanLength;
        private double sampleStdevLength;
        private double sampleVarianceLength;
        private long countTotal;
        private double[] histogramBuckets;
        private long[] histogramBucketCounts;

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
