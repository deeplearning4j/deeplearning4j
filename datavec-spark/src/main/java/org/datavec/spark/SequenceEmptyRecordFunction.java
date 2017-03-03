package org.datavec.spark;

import org.apache.spark.api.java.function.Function;
import org.datavec.api.writable.Writable;

import java.util.List;

/**
 * Used for filtering empty records
 *
 * @author Adam Gibson
 */
public class SequenceEmptyRecordFunction implements Function<List<List<Writable>>, Boolean> {
    @Override
    public Boolean call(List<List<Writable>> v1) throws Exception {
        return v1.isEmpty();
    }
}
