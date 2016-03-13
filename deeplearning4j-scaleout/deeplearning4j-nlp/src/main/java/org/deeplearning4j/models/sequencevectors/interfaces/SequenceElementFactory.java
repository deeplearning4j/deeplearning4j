package org.deeplearning4j.models.sequencevectors.interfaces;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * This is interface for JSON -> SequenceElement deserialziation
 *
 * @author raver119@gmail.com
 */
public interface SequenceElementFactory<T extends SequenceElement> {
    /**
     * This method builds object from provided JSON
     *
     * @param json JSON for restored object
     * @return restored object
     */
    T deserialize(String json);

    /**
     * This method serializaes object  into JSON string
     * @param element
     * @return
     */
    String serialize(T element);
}
