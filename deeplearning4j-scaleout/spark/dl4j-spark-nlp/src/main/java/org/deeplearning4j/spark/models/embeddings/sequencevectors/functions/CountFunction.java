package org.deeplearning4j.spark.models.embeddings.sequencevectors.functions;

import lombok.NonNull;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.Function;
import org.apache.spark.api.java.function.PairFunction;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import scala.Tuple2;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This accumulator function does count individual elements, using provided Accumulator
 *
 * @author raver119@gmail.com
 */
public class CountFunction<T extends SequenceElement> implements Function<Sequence<T>, Pair<Sequence<T>, Long>>{
    protected Accumulator<Counter<Long>> accumulator;
    protected boolean fetchLabels;

    public CountFunction(@NonNull  Accumulator<Counter<Long>>  accumulator, boolean fetchLabels) {
        this.accumulator = accumulator;
        this.fetchLabels = fetchLabels;
    }

    @Override
    public Pair<Sequence<T>, Long> call(Sequence<T> sequence) throws Exception {
        // since we can't be 100% sure that sequence size is ok itself, or it's not overflow through int limits, we'll recalculate it.
        // anyway we're going to loop through it for elements frequencies
        Counter<Long> localCounter = new Counter<>();
        long seqLen = 0;

        for (T element: sequence.getElements()) {
            if (element == null)
                continue;

            // FIXME: hashcode is bad idea here. we need Long id
            localCounter.incrementCount((long) element.hashCode(), 1.0);
            seqLen++;
        }

        accumulator.add(localCounter);

        return Pair.makePair(sequence, seqLen);
    }
}
