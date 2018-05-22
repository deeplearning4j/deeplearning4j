package org.deeplearning4j.models.sequencevectors.sequence;

/**
 * This is special shallow SequenceElement implementation, that doesn't hold labels or any other custom user-defined data
 *
 * @author raver119@gmail.com
 */
public class ShallowSequenceElement extends SequenceElement {

    public ShallowSequenceElement(double frequency, long id) {
        this.storageId = id;
        this.elementFrequency.set(frequency);
    }

    @Override
    public String getLabel() {
        return null;
    }

    @Override
    public String toJSON() {
        return null;
    }
}
