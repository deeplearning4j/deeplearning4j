package org.deeplearning4j.spark.models.sequencevectors.learning.elements;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.embeddings.learning.impl.elements.RandomUtils;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.nd4j.parameterserver.distributed.logic.sequence.BasicSequenceProvider;
import org.nd4j.parameterserver.distributed.messages.Frame;
import org.nd4j.parameterserver.distributed.messages.TrainingMessage;
import org.nd4j.parameterserver.distributed.messages.requests.SkipGramRequestMessage;
import org.nd4j.parameterserver.distributed.training.TrainingDriver;
import org.nd4j.parameterserver.distributed.training.impl.SkipGramTrainer;

import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
@Slf4j
public class SparkSkipGram extends BaseSparkLearningAlgorithm {
    @Override
    public String getCodeName() {
        return "Spark-SkipGram";
    }

    protected transient AtomicLong counter;
    protected transient ThreadLocal<Frame<SkipGramRequestMessage>> frame;

    protected TrainingDriver<SkipGramRequestMessage> driver = new SkipGramTrainer();

    @Override
    public Frame<? extends TrainingMessage> frameSequence(Sequence<ShallowSequenceElement> sequence,
                    AtomicLong nextRandom, double learningRate) {

        // FIXME: totalElementsCount should have real value
        if (vectorsConfiguration.getSampling() > 0)
            sequence = BaseSparkLearningAlgorithm.applySubsampling(sequence, nextRandom, 10L,
                            vectorsConfiguration.getSampling());

        int currentWindow = vectorsConfiguration.getWindow();

        if (vectorsConfiguration.getVariableWindows() != null
                        && vectorsConfiguration.getVariableWindows().length != 0) {
            currentWindow = vectorsConfiguration.getVariableWindows()[RandomUtils
                            .nextInt(vectorsConfiguration.getVariableWindows().length)];
        }
        if (frame == null)
            synchronized (this) {
                if (frame == null)
                    frame = new ThreadLocal<>();
            }

        if (frame.get() == null)
            frame.set(new Frame<SkipGramRequestMessage>(BasicSequenceProvider.getInstance().getNextValue()));

        for (int i = 0; i < sequence.size(); i++) {
            nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));

            ShallowSequenceElement word = sequence.getElementByIndex(i);
            if (word == null)
                continue;

            int b = (int) (nextRandom.get() % currentWindow);
            int end = currentWindow * 2 + 1 - b;
            for (int a = b; a < end; a++) {
                if (a != currentWindow) {
                    int c = i - currentWindow + a;
                    if (c >= 0 && c < sequence.size()) {
                        ShallowSequenceElement lastWord = sequence.getElementByIndex(c);
                        iterateSample(word, lastWord, nextRandom, learningRate);
                        nextRandom.set(Math.abs(nextRandom.get() * 25214903917L + 11));
                    }
                }
            }
        }

        // at this moment we should have something in ThreadLocal Frame, so we'll send it to VoidParameterServer for processing

        Frame<SkipGramRequestMessage> currentFrame = frame.get();
        frame.set(new Frame<SkipGramRequestMessage>(BasicSequenceProvider.getInstance().getNextValue()));

        return currentFrame;
    }



    protected void iterateSample(ShallowSequenceElement word, ShallowSequenceElement lastWord, AtomicLong nextRandom,
                    double lr) {
        if (word == null || lastWord == null || lastWord.getIndex() < 0 || word.getIndex() == lastWord.getIndex())
            return;
        /**
         * all we want here, is actually very simple:
         * we just build simple SkipGram frame, and send it over network
         */

        int[] idxSyn1 = new int[0];
        byte[] codes = new byte[0];
        if (vectorsConfiguration.isUseHierarchicSoftmax()) {
            idxSyn1 = new int[word.getCodeLength()];
            codes = new byte[word.getCodeLength()];
            for (int i = 0; i < word.getCodeLength(); i++) {
                byte code = word.getCodes().get(i);
                int point = word.getPoints().get(i);
                if (point >= vocabCache.numWords() || point < 0)
                    continue;

                codes[i] = code;
                idxSyn1[i] = point;
            }
        }

        short neg = (short) vectorsConfiguration.getNegative();
        SkipGramRequestMessage sgrm = new SkipGramRequestMessage(word.getIndex(), lastWord.getIndex(), idxSyn1, codes,
                        neg, lr, nextRandom.get());

        // we just stackfor now
        frame.get().stackMessage(sgrm);
    }

    @Override
    public TrainingDriver<? extends TrainingMessage> getTrainingDriver() {
        return driver;
    }
}
