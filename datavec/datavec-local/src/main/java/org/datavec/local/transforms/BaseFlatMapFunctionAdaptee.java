package org.datavec.local.transforms;


import org.datavec.local.transforms.functions.FlatMapFunctionAdapter;
import org.nd4j.linalg.exception.ND4JIllegalStateException;

import java.util.List;

/**
 *
 * This class should be used instead of direct referral to FlatMapFunction
 *
 */
public class BaseFlatMapFunctionAdaptee<K, V>  {

    protected final FlatMapFunctionAdapter<K, V> adapter;

    public BaseFlatMapFunctionAdaptee(FlatMapFunctionAdapter<K, V> adapter) {
        this.adapter = adapter;
    }

    public List<V> call(K k)  {
        try {
            return adapter.call(k);
        } catch (Exception e) {
            throw new IllegalStateException(e);
        }

    }
}
