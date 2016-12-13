package org.deeplearning4j.spark.models.sequencevectors.learning.sequence;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.spark.models.sequencevectors.learning.elements.BaseSparkLearningAlgorithm;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class SparkDM extends BaseSparkLearningAlgorithm {
    @Override
    public String getCodeName() {
        return "Spark-DM";
    }

    @Override
    public double learnSequence(Sequence<ShallowSequenceElement> sequence, AtomicLong nextRandom, double learningRate) {
        return 0;
    }
}
