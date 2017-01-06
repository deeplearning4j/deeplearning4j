package org.deeplearning4j.spark.models.sequencevectors.learning.elements;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.training.impl.SkipGramTrainer;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class SparkCBOW extends BaseSparkLearningAlgorithm {
    @Override
    public String getCodeName() {
        return "Spark-CBOW";
    }

    @Override
    public double learnSequence(Sequence<ShallowSequenceElement> sequence, AtomicLong nextRandom, double learningRate) {
        return 0;
    }

    @Override
    public TrainingDriver<? extends TrainingMessage> getTrainingDriver() {
        return null;
    }
}
