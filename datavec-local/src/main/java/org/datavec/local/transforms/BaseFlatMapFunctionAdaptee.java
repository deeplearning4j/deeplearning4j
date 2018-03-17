package org.datavec.local.transforms;


import org.datavec.local.transforms.functions.FlatMapFunctionAdapter;

/**
 * FlatMapFunction adapter to
 * hide incompatibilities between Spark 1.x and Spark 2.x
 *
 * This class should be used instead of direct referral to FlatMapFunction
 *
 */
public class BaseFlatMapFunctionAdaptee<K, V>  {

    protected final FlatMapFunctionAdapter<K, V> adapter;

    public BaseFlatMapFunctionAdaptee(FlatMapFunctionAdapter<K, V> adapter) {
        this.adapter = adapter;
    }

    public Iterable<V> call(K k)  {
        try {
            return adapter.call(k);
        } catch (Exception e) {
            e.printStackTrace();
        }

        throw new IllegalStateException("");
    }
}
