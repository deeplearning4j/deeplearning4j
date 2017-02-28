package org.deeplearning4j.spark.models.sequencevectors.functions;

import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.util.List;

/**
 * Simple function to convert List<T extends SequenceElement> to Sequence<T>
 *
 * @author raver119@gmail.com
 */
public class ListSequenceConvertFunction<T extends SequenceElement> implements Function<List<T>, Sequence<T>> {
    @Override
    public Sequence<T> call(List<T> ts) throws Exception {
        Sequence<T> sequence = new Sequence<>(ts);
        return sequence;
    }
}
