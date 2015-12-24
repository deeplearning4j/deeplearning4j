package org.deeplearning4j.models.glove.count;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * Created by raver on 24.12.2015.
 */
public interface CoOccurenceReader<T extends SequenceElement> {
    /*
        Storage->Memory merging part
     */
    boolean hasMoreObjects();


    CoOccurrenceWeight<T> nextObject();

    void finish();
}
