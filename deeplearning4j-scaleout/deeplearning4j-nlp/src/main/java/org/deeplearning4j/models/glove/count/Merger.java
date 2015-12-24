package org.deeplearning4j.models.glove.count;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * Created by raver on 24.12.2015.
 */
public interface Merger<T extends SequenceElement> {
    /*
        Storage->Memory merging part
     */
    boolean hasMoreObjects();


    CoOccurrenceWeight<T> nextObject();
    /*
        Memory -> Storage part
     */
    void writeObject(CoOccurrenceWeight<T> object);
}
