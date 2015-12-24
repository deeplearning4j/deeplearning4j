package org.deeplearning4j.models.glove.count;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.io.File;

/**
 * Binary implementation of Merger interface, used to provide off-memory storage for cooccurrence maps generated for GloVe
 *
 * @author raver119@gmail.com
 */
public class BinaryMerger<T extends SequenceElement> implements Merger<T> {
    private VocabCache<T> vocabCache;

    public BinaryMerger(@NonNull File file, @NonNull VocabCache<T> vocabCache) {
        this.vocabCache = vocabCache;
    }

    @Override
    public boolean hasMoreObjects() {
        return false;
    }

    @Override
    public CoOccurrenceWeight<T> nextObject() {
        return null;
    }

    @Override
    public void writeObject(@NonNull CoOccurrenceWeight<T> object) {

    }
}
