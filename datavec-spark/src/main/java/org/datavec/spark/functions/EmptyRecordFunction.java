package org.datavec.spark.functions;

import org.apache.spark.api.java.function.Function;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Used for filtering empty records
 *
 * @author Adam Gibson
 */
public class EmptyRecordFunction implements Function<List<Writable>, Boolean> {
    @Override
    public Boolean call(List<Writable> v1) throws Exception {
        return v1.isEmpty();
    }
}
