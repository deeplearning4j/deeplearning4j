package org.deeplearning4j.models.embeddings.training.impl;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.training.ElementsLearningAlgorithm;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * Created by raver on 17.12.2015.
 */
public class GloVe<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {

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
}
