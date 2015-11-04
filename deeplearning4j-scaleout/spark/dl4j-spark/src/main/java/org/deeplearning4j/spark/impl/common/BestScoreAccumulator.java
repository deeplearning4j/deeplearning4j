package org.deeplearning4j.spark.impl.common;

import org.apache.spark.Accumulator;
import org.apache.spark.AccumulatorParam;
import org.apache.spark.SparkContext;

/**
 * Accumulator which tracks best score seen.
 *
 * Created by mlapan on 27/10/15.
 */
public class BestScoreAccumulator implements AccumulatorParam<Double> {
    public static final String NAME = "Best score";

    @Override
    public Double addAccumulator(Double v1, Double v2) {
        return Math.min(v1, v2);
    }

    @Override
    public Double addInPlace(Double v1, Double v2) {
        return Math.min(v1, v2);
    }

    @Override
    public Double zero(Double init) {
        return Double.POSITIVE_INFINITY;
    }
    
    public static Accumulator<Double> create(SparkContext sc) {
        return sc.accumulator(Double.POSITIVE_INFINITY, NAME, new BestScoreAccumulator());
    }
}
