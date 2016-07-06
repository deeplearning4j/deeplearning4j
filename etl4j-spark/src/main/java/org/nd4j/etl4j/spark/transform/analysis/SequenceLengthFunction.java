package org.nd4j.etl4j.spark.transform.analysis;

import org.apache.spark.api.java.function.Function;
import org.canova.api.writable.Writable;

import java.util.List;

/**
 * Map a sequence to the size of that sequence
 */
public class SequenceLengthFunction implements Function<List<List<Writable>>,Integer> {
    @Override
    public Integer call(List<List<Writable>> v1) throws Exception {
        return v1.size();
    }
}
