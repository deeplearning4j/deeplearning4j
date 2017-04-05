package org.datavec.spark.transform;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.spark.functions.FlatMapFunctionAdapter;

/**
 * FlatMapFunction adapter to
 * hide incompatibilities between Spark 1.x and Spark 2.x
 *
 * This class should be used instead of direct referral to FlatMapFunction
 *
 */
public class BaseFlatMapFunctionAdaptee<K, V> implements FlatMapFunction<K, V> {

    protected final FlatMapFunctionAdapter<K, V> adapter;

    public BaseFlatMapFunctionAdaptee(FlatMapFunctionAdapter<K, V> adapter) {
        this.adapter = adapter;
    }

    @Override
    public Iterable<V> call(K k) throws Exception {
        return adapter.call(k);
    }
}
