package org.deeplearning4j.spark.models.sequencevectors.learning.sequence;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.spark.models.sequencevectors.learning.elements.BaseSparkLearningAlgorithm;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class SparkDBOW extends BaseSparkLearningAlgorithm {
    @Override
    public String getCodeName() {
        return "Spark-DBOW";
    }


    @Override
    public Frame<? extends TrainingMessage> frameSequence(Sequence<ShallowSequenceElement> sequence, AtomicLong nextRandom, double learningRate) {
        return null;
    }

    @Override
    public TrainingDriver<? extends TrainingMessage> getTrainingDriver() {
        return null;
    }
}
