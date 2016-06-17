package org.deeplearning4j.models.sequencevectors.listeners;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent;
import org.deeplearning4j.models.sequencevectors.interfaces.VectorsListener;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple VectorsListener implementation that prints out model score.
 *
 * @author raver119@gmail.com
 */
public class ScoreListener<T extends SequenceElement> implements VectorsListener<T> {
    protected static final Logger logger = LoggerFactory.getLogger(ScoreListener.class);
    private final ListenerEvent targetEvent;

    public ScoreListener(@NonNull ListenerEvent targetEvent) {
        this.targetEvent = targetEvent;
    }

    @Override
    public boolean validateEvent(ListenerEvent event, long argument) {
        if (event == targetEvent)
            return true;

        return false;
    }

    @Override
    public void processEvent(ListenerEvent event, SequenceVectors<T> sequenceVectors, long argument) {
        if (event != targetEvent)
            return;

        logger.info("Average score for last batch: {}", sequenceVectors.getScore());
    }
}
