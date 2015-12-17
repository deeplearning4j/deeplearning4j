package org.deeplearning4j.models.embeddings.training;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * Implementations of this interface should contain sequence-related learning algorithms. Like dbow or dm.
 *
 * @author raver119@gmail.com
 */
public interface SequenceLearningAlgorithm<T extends SequenceElement> {

    String getCodeName();

    void learnSequence(Sequence<T> sequence);
}
