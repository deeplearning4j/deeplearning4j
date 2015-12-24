package org.deeplearning4j.models.glove.count;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * Created by fartovii on 25.12.15.
 */
public interface CoOccurrenceWriter<T extends SequenceElement> {

    /*
    Memory -> Storage part
 */
    void writeObject(CoOccurrenceWeight<T> object);

    void finish();
}
