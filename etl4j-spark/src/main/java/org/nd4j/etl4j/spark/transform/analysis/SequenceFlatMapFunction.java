package org.nd4j.etl4j.spark.transform.analysis;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.canova.api.writable.Writable;

import java.util.List;

/**
 * SequenceFlatMapFunction: very simple function used to flatten a sequence
 * Typically used only internally for certain analysis operations
 *
 * @author Alex Black
 */
public class SequenceFlatMapFunction implements FlatMapFunction<List<List<Writable>>, List<Writable>> {
    @Override
    public Iterable<List<Writable>> call(List<List<Writable>> collections) throws Exception {
        return collections;
    }

}
