/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.arbiter.util;


import org.nd4j.linalg.api.ndarray.INDArray;

/**
 * @author Adam Gibson
 */
public class SummaryStatistics {

    private double mean ,sum,min,max;

    private SummaryStatistics() {}

    private SummaryStatistics(INDArray mean,INDArray sum,INDArray min,INDArray max) {
        this.mean = (double) mean.element();
        this.sum = (double) sum.element();
        this.min = (double) min.element();
        this.max = (double) max.element();
    }

    private SummaryStatistics(double mean,double sum,double min,double max) {
        this.mean = mean;
        this.sum = sum;
        this.min = min;
        this.max = max;
    }


    public static String summaryStatsString(INDArray d) {
        return new SummaryStatistics(d.mean(Integer.MAX_VALUE),d.sum(Integer.MAX_VALUE),
            d.min(Integer.MAX_VALUE),d.max(Integer.MAX_VALUE)).toString();
    }

    public static SummaryStatistics summaryStats(INDArray d) {
        return new SummaryStatistics(d.mean(Integer.MAX_VALUE),d.sum(Integer.MAX_VALUE),
            d.min(Integer.MAX_VALUE),d.max(Integer.MAX_VALUE));
    }


    public double getMean() {
        return mean;
    }

    public void setMean(double mean) {
        this.mean = mean;
    }

    public double getSum() {
        return sum;
    }

    public void setSum(double sum) {
        this.sum = sum;
    }

    public double getMin() {
        return min;
    }

    public void setMin(double min) {
        this.min = min;
    }

    public double getMax() {
        return max;
    }

    public void setMax(double max) {
        this.max = max;
    }

    @Override
    public String toString() {
        return "SummaryStatistics{" +
                "mean=" + mean +
                ", sum=" + sum +
                ", min=" + min +
                ", max=" + max +
                '}';
    }
}
