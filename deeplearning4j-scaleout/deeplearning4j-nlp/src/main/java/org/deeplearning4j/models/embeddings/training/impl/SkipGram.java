package org.deeplearning4j.models.embeddings.training.impl;

import lombok.NonNull;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.training.ElementsLearningAlgorithm;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

/**
 * Skip-Gram implementation for dl4j SequenceVectors
 *
 * @author raver119@gmail.com
 */
public class SkipGram<T extends SequenceElement> implements ElementsLearningAlgorithm<T> {
    protected VocabCache<T> vocabCache;
    protected WeightLookupTable<T> lookupTable;

    public SkipGram() {

    }

    @Override
    public String getCodeName() {
        return "skipgram";
    }

    @Override
    public void configure(@NonNull VocabCache<T> vocabCache, @NonNull WeightLookupTable<T> lookupTable) {
        this.vocabCache = vocabCache;
        this.lookupTable = lookupTable;
    }

    @Override
    public void learnSequence(Sequence<T> sequence) {

    }
}
