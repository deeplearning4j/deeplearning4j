package org.datavec.spark.transform.utils.adapter;

import org.apache.spark.api.java.function.Function2;
import org.nd4j.linalg.function.BiFunction;

public class BiFunctionAdapter<A,B,R> implements Function2<A,B,R> {

    private final BiFunction<A,B,R> fn;

    public BiFunctionAdapter(BiFunction<A,B,R> fn){
        this.fn = fn;
    }

    @Override
    public R call(A v1, B v2) throws Exception {
        return fn.apply(v1, v2);
    }
}
