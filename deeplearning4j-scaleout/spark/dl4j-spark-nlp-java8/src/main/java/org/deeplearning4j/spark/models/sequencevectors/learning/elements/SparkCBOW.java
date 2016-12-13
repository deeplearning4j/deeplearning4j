package org.deeplearning4j.spark.models.sequencevectors.learning.elements;

import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;

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
}
