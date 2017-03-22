package org.deeplearning4j.models.sequencevectors.interfaces;

import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.sequencevectors.enums.ListenerEvent;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * This interface describes Listeners to SequenceVectors and its derivatives.
 *
 * @author raver119@gmail.com
 */
public interface VectorsListener<T extends SequenceElement> {

    /**
     * This method is called prior each processEvent call, to check if this specific VectorsListener implementation is viable for specific event
     *
     * @param event
     * @param argument
     * @return TRUE, if this event can and should be processed with this listener, FALSE otherwise
     */
    boolean validateEvent(ListenerEvent event, long argument);

    /**
     * This method is called at each epoch end
     *
     * @param event
     * @param sequenceVectors
     */
    void processEvent(ListenerEvent event, SequenceVectors<T> sequenceVectors, long argument);
}
