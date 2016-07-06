package org.nd4j.etl4j.spark.transform.misc;

import org.apache.spark.api.java.function.Function;
import org.canova.api.writable.Writable;
import org.nd4j.etl4j.api.transform.sequence.merge.SequenceMerge;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.List;

/**
 * Spark function for merging multiple sequences, using a {@link SequenceMerge} instance.<br>
 *
 * Typical usage:<br>
 * <pre>
 * {@code
 * JavaPairRDD<SomeKey,List<List<Writable>>> myData = ...;
 * SequenceComparator comparator = ...;
 * SequenceMergeFunction<String> sequenceMergeFunction = new SequenceMergeFunction<>(new SequenceMerge(comparator));
 * JavaRDD<List<List<Writable>>> merged = myData.groupByKey().map(sequenceMergeFunction);
 * }
 * </pre>
 *
 * @author Alex Black
 */
public class SequenceMergeFunction <T> implements Function<Tuple2<T,Iterable<List<List<Writable>>>>,List<List<Writable>>> {

    private SequenceMerge sequenceMerge;

    public SequenceMergeFunction(SequenceMerge sequenceMerge){
        this.sequenceMerge = sequenceMerge;
    }

    @Override
    public List<List<Writable>> call(Tuple2<T, Iterable<List<List<Writable>>>> t2) throws Exception {
        List<List<List<Writable>>> sequences = new ArrayList<>();
        for(List<List<Writable>> l : t2._2()){
            sequences.add(l);
        }

        return sequenceMerge.mergeSequences(sequences);
    }
}
