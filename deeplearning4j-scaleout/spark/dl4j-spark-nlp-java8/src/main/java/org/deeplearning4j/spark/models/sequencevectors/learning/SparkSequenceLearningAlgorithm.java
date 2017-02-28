package org.deeplearning4j.spark.models.sequencevectors.learning;

import org.deeplearning4j.models.embeddings.learning.SequenceLearningAlgorithm;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;

/**
 * Identification layer for Spark-ready implementations of LearningAlgorithms
 *
 * @author raver119@gmail.com
 */
public interface SparkSequenceLearningAlgorithm extends SequenceLearningAlgorithm<ShallowSequenceElement> {
}
