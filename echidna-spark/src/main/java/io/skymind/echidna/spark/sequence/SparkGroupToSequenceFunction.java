package io.skymind.echidna.spark.sequence;

import lombok.AllArgsConstructor;
import org.apache.spark.api.java.function.Function;
import org.canova.api.writable.Writable;
import io.skymind.echidna.api.sequence.SequenceComparator;
import scala.Tuple2;

import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 * Spark function for grouping independent values/examples into a sequence, and then sorting them
 * using a provided {@link SequenceComparator}
 *
 * @author Alex Black
 */
@AllArgsConstructor
public class SparkGroupToSequenceFunction implements Function<Tuple2<Writable,Iterable<List<Writable>>>,List<List<Writable>>> {

    private final SequenceComparator comparator;

    @Override
    public List<List<Writable>> call(Tuple2<Writable, Iterable<List<Writable>>> tuple) throws Exception {

        List<List<Writable>> list = new ArrayList<>();
        for (List<Writable> writables : tuple._2()) list.add(writables);

        Collections.sort(list,comparator);

        return list;
    }
}
