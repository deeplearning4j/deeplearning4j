package org.datavec.spark.transform.analysis;

import org.apache.spark.api.java.function.Function;
import org.datavec.api.writable.Writable;

/**
 * Created by Alex on 4/03/2016.
 */
public class WritableToStringFunction implements Function<Writable,String> {
    @Override
    public String call(Writable writable) throws Exception {
        return writable.toString();
    }
}
