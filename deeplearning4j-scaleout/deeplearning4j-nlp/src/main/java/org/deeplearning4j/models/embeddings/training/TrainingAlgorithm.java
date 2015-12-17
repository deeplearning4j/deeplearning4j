package org.deeplearning4j.models.embeddings.training;

import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

/**
 * This interface implementations will hold implementations for skip-gram and glove
 *
 * @author raver119@gmail.com
 */
public interface TrainingAlgorithm {
    <T extends SequenceElement>void iterateSample(T word1, T word2);
}
