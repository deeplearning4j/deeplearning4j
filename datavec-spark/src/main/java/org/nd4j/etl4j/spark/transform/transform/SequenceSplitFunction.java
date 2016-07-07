package org.nd4j.etl4j.spark.transform.transform;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.FlatMapFunction;
import org.nd4j.etl4j.api.writable.Writable;
import org.nd4j.etl4j.api.transform.sequence.SequenceSplit;

import java.util.List;

/**
 * Created by Alex on 17/03/2016.
 */
@AllArgsConstructor
public class SequenceSplitFunction implements FlatMapFunction<List<List<Writable>>,List<List<Writable>>>{

    private final SequenceSplit split;

    @Override
    public Iterable<List<List<Writable>>> call(List<List<Writable>> collections) throws Exception {
        return split.split(collections);
    }
}
