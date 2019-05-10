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

package org.datavec.api.transform.analysis.columns;

import com.tdunning.math.stats.TDigest;
import lombok.Data;
import lombok.EqualsAndHashCode;
import org.datavec.api.transform.analysis.json.TDigestDeserializer;
import org.datavec.api.transform.analysis.json.TDigestSerializer;
import org.nd4j.shade.jackson.databind.annotation.JsonDeserialize;
import org.nd4j.shade.jackson.databind.annotation.JsonSerialize;

/**
 * Abstract class for numerical column analysis
 *
 * @author Alex Black
 */
@Data
@EqualsAndHashCode(exclude = {"digest"})
public abstract class NumericalColumnAnalysis implements ColumnAnalysis {

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
    @JsonSerialize(using = TDigestSerializer.class)
    @JsonDeserialize(using = TDigestDeserializer.class)
    protected TDigest digest;

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
        this.digest = builder.digest;
    }

    protected NumericalColumnAnalysis() {
        //No arg for Jackson
    }

    @Override
    public String toString() {
        String q = "";
        if(digest != null) {
            StringBuilder quantiles = new StringBuilder(", quantiles=[");
            double[] printReports = new double[]{0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999};
            for (int i = 0; i < printReports.length; i++) {
                quantiles.append(printReports[i]).append(" -> ").append(digest.quantile(printReports[i]));
                if (i < printReports.length - 1) quantiles.append(",");
            }
            quantiles.append("]");
            q = quantiles.toString();
        }
        return "mean=" + mean + ",sampleStDev=" + sampleStdev + ",sampleVariance=" + sampleVariance + ",countZero="
                        + countZero + ",countNegative=" + countNegative + ",countPositive=" + countPositive
                        + ",countMinValue=" + countMinValue + ",countMaxValue=" + countMaxValue + ",count="
                        + countTotal + q;
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
        protected TDigest digest;

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

        public T digest(TDigest digest){
            this.digest = digest;
            return (T) this;
        }

    }

}
