package org.deeplearning4j.spark.models.sequencevectors.learning.elements;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.learning.impl.elements.RandomUtils;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class SparkSkipGram extends BaseSparkLearningAlgorithm {
    @Override
    public String getCodeName() {
        return "Spark-SkipGram";
    }

    @Override
    public double learnSequence(Sequence<ShallowSequenceElement> sequence, AtomicLong nextRandom, double learningRate) {
        if (vectorsConfiguration.getSampling() > 0)
            sequence = BaseSparkLearningAlgorithm.applySubsampling(sequence, nextRandom, 10L, vectorsConfiguration.getSampling());

        int currentWindow = vectorsConfiguration.getWindow();

        if (vectorsConfiguration.getVariableWindows() != null && vectorsConfiguration.getVariableWindows().length != 0) {
            currentWindow = vectorsConfiguration.getVariableWindows()[RandomUtils.nextInt(vectorsConfiguration.getVariableWindows().length)];
        }

        for (int i = 0; i < sequence.size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));

            ShallowSequenceElement word = sequence.getElementByIndex(i);
            if(word == null || sequence.isEmpty())
                return 0.0;

            int b = (int) (nextRandom.get() % currentWindow);
            int end =  currentWindow * 2 + 1 - b;
            for(int a = b; a < end; a++) {
                if(a != currentWindow) {
                    int c = i - currentWindow + a;
                    if(c >= 0 && c < sequence.size()) {
                        ShallowSequenceElement lastWord = sequence.getElementByIndex(c);
                        iterateSample(word,lastWord,nextRandom, learningRate);
                        nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
                    }
                }
            }
        }

        return 0;
    }


    protected void iterateSample(ShallowSequenceElement word, ShallowSequenceElement lastWord, AtomicLong nextRandom, double lr) {
        // TODO: to be implemented
    }
}
