package org.datavec.spark.transform.analysis;

import org.datavec.api.writable.Writable;
import org.datavec.spark.functions.FlatMapFunctionAdapter;

import java.util.List;

/**
 * SequenceFlatMapFunction: very simple function used to flatten a sequence
 * Typically used only internally for certain analysis operations
 *
 * @author Alex Black
 */
public class SequenceFlatMapFunctionAdapter implements FlatMapFunctionAdapter<List<List<Writable>>, List<Writable>> {
    @Override
    public Iterable<List<Writable>> call(List<List<Writable>> collections) throws Exception {
        return collections;
    }

}
