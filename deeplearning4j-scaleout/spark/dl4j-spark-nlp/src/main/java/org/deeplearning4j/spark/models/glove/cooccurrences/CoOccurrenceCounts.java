package org.deeplearning4j.spark.models.glove.cooccurrences;

import org.apache.spark.api.java.function.Function2;
import org.deeplearning4j.berkeley.CounterMap;


/**
 * Co occurrence count reduction
 * @author Adam Gibson
 */
public class CoOccurrenceCounts implements Function2<CounterMap<String,String>,CounterMap<String,String>,CounterMap<String,String>> {


    @Override
    public CounterMap<String,String> call(CounterMap<String,String> v1, CounterMap<String,String> v2) throws Exception {
        v1.incrementAll(v2);
        return v1;
    }
}
