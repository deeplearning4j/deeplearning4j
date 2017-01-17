package org.deeplearning4j.spark.models.sequencevectors.functions;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.Accumulator;
import org.apache.spark.api.java.function.Function;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * This accumulator function does count individual elements, using provided Accumulator
 *
 * @author raver119@gmail.com
 */
@Slf4j
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

        log.info("Count function executed");
        System.out.println("Cnt function executed");

        for (T element: sequence.getElements()) {
            if (element == null)
                continue;

            // FIXME: hashcode is bad idea here. we need Long id
            localCounter.incrementCount(element.getStorageId(), 1.0);
            seqLen++;
        }

        // FIXME: we're missing label information here due to shallow vocab mechanics
        if (sequence.getSequenceLabels() != null)
            for (T label: sequence.getSequenceLabels()) {
                localCounter.incrementCount(label.getStorageId(), 1.0);
            }

        accumulator.add(localCounter);

        return Pair.makePair(sequence, seqLen);
    }
}
