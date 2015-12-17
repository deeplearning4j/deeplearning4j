package org.deeplearning4j.models.embeddings.training.impl.elements;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.training.ElementsLearningAlgorithm;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * GloVe implementation for SequenceVectors
 *
 * @author raver119@gmail.com
 */
public abstract class GloVe<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {
/*
    @Override
    public String getCodeName() {
        return null;
    }

    @Override
    public void configure(VocabCache<T> vocabCache, WeightLookupTable<T> lookupTable) {

    }

    @Override
    public void learnSequence(Sequence<T> sequence) {

    }
    */
}
