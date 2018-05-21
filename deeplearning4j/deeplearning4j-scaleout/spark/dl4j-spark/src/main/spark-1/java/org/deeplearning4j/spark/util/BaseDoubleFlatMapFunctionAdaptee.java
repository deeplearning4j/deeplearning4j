package org.deeplearning4j.spark.util;

import org.apache.spark.api.java.function.DoubleFlatMapFunction;
import org.datavec.spark.functions.FlatMapFunctionAdapter;

/**
 * DoubleFlatMapFunction adapter to hide incompatibilities between Spark 1.x and Spark 2.x
 *
 * This class should be used instead of direct referral to DoubleFlatMapFunction
 *
 */
public class BaseDoubleFlatMapFunctionAdaptee<T> implements DoubleFlatMapFunction<T> {

    protected final FlatMapFunctionAdapter<T, Double> adapter;

    public BaseDoubleFlatMapFunctionAdaptee(FlatMapFunctionAdapter<T, Double> adapter) {
        this.adapter = adapter;
    }

    @Override
    public Iterable<Double> call(T t) throws Exception {
        return adapter.call(t);
    }
}
