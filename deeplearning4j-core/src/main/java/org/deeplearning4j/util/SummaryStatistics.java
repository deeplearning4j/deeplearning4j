package org.deeplearning4j.util;

import org.jblas.DoubleMatrix;

/**
 * @author Adam Gibson
 */
public class SummaryStatistics {

    private double mean ,sum,min,max;

    private SummaryStatistics() {}



    private SummaryStatistics(double mean,double sum,double min,double max) {
        this.mean = mean;
        this.sum = sum;
        this.min = min;
        this.max = max;
    }


    public static String summaryStatsString(DoubleMatrix d) {
        return new SummaryStatistics(d.mean(),d.sum(),d.min(),d.max()).toString();
    }

    public static SummaryStatistics summaryStats(DoubleMatrix d) {
         return new SummaryStatistics(d.mean(),d.sum(),d.min(),d.max());
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
