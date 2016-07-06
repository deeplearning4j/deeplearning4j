package io.skymind.echidna.spark.analysis;

import org.apache.spark.api.java.function.DoubleFunction;
import org.canova.api.writable.Writable;

/**
 * Created by Alex on 4/03/2016.
 */
public class WritableToDoubleFunction implements DoubleFunction<Writable> {

    @Override
    public double call(Writable writable) throws Exception {
        return writable.toDouble();
    }
}
