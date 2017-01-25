package org.deeplearning4j.spark.util;

import org.apache.spark.api.java.function.DoubleFlatMapFunction;

import java.util.Iterator;

/**
 * DoubleFlatMapFunction adapter to hide incompatibilities between Spark 1.x and Spark 2.x
 *
 * This class should be used instead of direct referral to DoubleFlatMapFunction
 *
 */
public class BaseDoubleFlatMapFunctionAdaptee<T> implements DoubleFlatMapFunction<T> {

    protected final DoubleFlatMapFunctionAdapter<T> adapter;

    public BaseDoubleFlatMapFunctionAdaptee(DoubleFlatMapFunctionAdapter<T> adapter) {
        this.adapter = adapter;
    }

    @Override
    public Iterator<Double> call(T t) throws Exception {
        return adapter.call(t).iterator();
    }
}
