package org.deeplearning4j.spark.models.sequencevectors.learning;

import org.deeplearning4j.models.embeddings.learning.ElementsLearningAlgorithm;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;

import java.util.concurrent.atomic.AtomicLong;

/**
 * Identification layer for Spark-ready implementations of LearningAlgorithms
 *
 * @author raver119@gmail.com
 */
public interface SparkElementsLearningAlgorithm extends ElementsLearningAlgorithm<ShallowSequenceElement> {
    TrainingDriver<? extends TrainingMessage> getTrainingDriver();

    Frame<? extends TrainingMessage> frameSequence(Sequence<ShallowSequenceElement> sequence, AtomicLong nextRandom,
                    double learningRate);
}
