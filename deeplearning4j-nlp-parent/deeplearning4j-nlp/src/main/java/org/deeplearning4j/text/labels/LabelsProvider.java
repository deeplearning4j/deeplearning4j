package org.deeplearning4j.text.labels;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * @author raver119@gmail.com
 */
public interface LabelsProvider<T extends SequenceElement> {

    T nextLabel();

    T getLabel(int index);
}
