package org.datavec.spark.transform.transform;

import org.apache.spark.api.java.function.FlatMapFunction;
import org.datavec.api.writable.Writable;
import org.datavec.api.transform.sequence.SequenceSplit;

import java.util.Iterator;
import java.util.List;

/**
 *
 */
public class SequenceSplitFunction implements FlatMapFunction<List<List<Writable>>,List<List<Writable>>>{

    private final SequenceSplitFunctionAdapter adapter;

    public SequenceSplitFunction(SequenceSplit split) {
        this.adapter = new SequenceSplitFunctionAdapter(split);
    }

    @Override
    public Iterator<List<List<Writable>>> call(List<List<Writable>> collections) throws Exception {
        return adapter.call(collections).iterator();
    }
}
