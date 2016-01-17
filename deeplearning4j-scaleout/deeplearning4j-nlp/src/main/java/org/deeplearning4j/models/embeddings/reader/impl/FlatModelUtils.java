package org.deeplearning4j.models.embeddings.reader.impl;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;

/**
 * This model reader is suited for tests, and for cases where flat scan against elements is required
 *
 * @author raver119@gmail.com
 */
public class FlatModelUtils<T extends SequenceElement> extends BasicModelUtils<T> {
    private static final Logger log = LoggerFactory.getLogger(FlatModelUtils.class);

    public FlatModelUtils() {

    }

    /**
     * This method does full scan against whole vocabulary, building descending list of similar words
     * @param label
     * @param n
     * @return
     */
    @Override
    public Collection<String> wordsNearest(String label, int n) {
        log.info("Flat scan against '"+label+"' word...");
        return wordsNearest(lookupTable.vector(label), n);
    }

    /**
     * This method does full scan against whole vocabulary, building descending list of similar words
     *
     * @param words
     * @param top
     * @return the words nearest the mean of the words
     */
    @Override
    public Collection<String> wordsNearest(INDArray words, int top) {
        log.info("Flat scan against vector...");
        Counter<String> distances = new Counter<>();

        for(String s : vocabCache.words()) {
            INDArray otherVec = lookupTable.vector(s);
            double sim = Transforms.cosineSim(words, otherVec);
            distances.incrementCount(s, sim);
        }

        distances.keepTopNKeys(top);
        return distances.keySet();
    }
}
