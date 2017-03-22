package org.deeplearning4j.models.sequencevectors.listeners;

import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent;
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Simple listener, to monitor similarity between selected elements during training
 *
 * @author raver119@gmail.com
 */
public class SimilarityListener<T extends SequenceElement> implements VectorsListener<T> {
    protected static final Logger logger = LoggerFactory.getLogger(SimilarityListener.class);
    private final ListenerEvent targetEvent;
    private final int frequency;
    private final String element1;
    private final String element2;
    private final AtomicLong counter = new AtomicLong(0);

    public SimilarityListener(ListenerEvent targetEvent, int frequency, String label1, String label2) {
        this.targetEvent = targetEvent;
        this.frequency = frequency;
        this.element1 = label1;
        this.element2 = label2;
    }

    @Override
    public boolean validateEvent(ListenerEvent event, long argument) {
        return event == targetEvent;
    }

    @Override
    public void processEvent(ListenerEvent event, SequenceVectors<T> sequenceVectors, long argument) {
        if (event != targetEvent)
            return;

        long cnt = counter.getAndIncrement();

        if (cnt % frequency != 0)
            return;

        double similarity = sequenceVectors.similarity(element1, element2);

        logger.info("Invocation: {}, similarity: {}", cnt, similarity);
    }
}
