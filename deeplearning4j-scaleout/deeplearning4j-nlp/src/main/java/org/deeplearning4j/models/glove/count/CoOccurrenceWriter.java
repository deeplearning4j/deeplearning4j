package org.deeplearning4j.models.glove.count;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * Created by fartovii on 25.12.15.
 */
public interface CoOccurrenceWriter<T extends SequenceElement> {

    /**
     * This method implementations should write out objects immediately
     * @param object
     */
    void writeObject(CoOccurrenceWeight<T> object);

    /**
     * This method implementations should queue objects for writing out.
     *
     * @param object
     */
    void queueObject(CoOccurrenceWeight<T> object);

    /**
     * Implementations of this method should close everything they use, before eradication
     */
    void finish();
}
