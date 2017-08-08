package org.deeplearning4j.spark.util;

import org.apache.spark.api.java.function.PairFlatMapFunction;
import org.datavec.spark.functions.FlatMapFunctionAdapter;
import scala.Tuple2;

/**
 * PairFlatMapFunction adapter to hide incompatibilities between Spark 1.x and Spark 2.x
 *
 * This class should be used instead of direct referral to PairFlatMapFunction
 *
 */
public class BasePairFlatMapFunctionAdaptee<T, K, V> implements PairFlatMapFunction<T, K, V> {

    protected final FlatMapFunctionAdapter<T, Tuple2<K, V>> adapter;

    public BasePairFlatMapFunctionAdaptee(FlatMapFunctionAdapter<T, Tuple2<K, V>> adapter) {
        this.adapter = adapter;
    }

    @Override
    public Iterable<Tuple2<K, V>> call(T t) throws Exception {
        return adapter.call(t);
    }
}
