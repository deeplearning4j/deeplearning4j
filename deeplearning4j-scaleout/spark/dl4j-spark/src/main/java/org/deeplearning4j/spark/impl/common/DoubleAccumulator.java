package org.deeplearning4j.spark.impl.common;

import org.apache.spark.Accumulator;
import org.apache.spark.AccumulatorParam;
import org.apache.spark.SparkContext;

/**
 * Created by nyghtowl on 1/14/16.
 */
public class DoubleAccumulator implements AccumulatorParam<Double> {

    @Override
    public Double addAccumulator(Double num, Double num1) {
        num += num1;
        return num;
        }

    @Override
    public Double addInPlace(Double num, Double num1) {
        num += num1;
        return num;
    }

    @Override
    public Double zero(Double num) {
            return num;
        }

    public static Accumulator<Double> create(SparkContext sc, String name) {
        return sc.accumulator(Double.POSITIVE_INFINITY, name, new DoubleAccumulator());
    }
}
