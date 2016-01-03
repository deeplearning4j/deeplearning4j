package org.deeplearning4j.models.embeddings.reader;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.Collection;

/**
 * Instances implementing this interface should be responsible for SequenceVectors model access
 *
 * @author raver119@gmail.com
 */
public interface ModelReader<T extends SequenceElement> {

    /**
     * This method implementations should accept given params, and use them in further calls to interface methods
     *
     * @param vocabCache
     * @param lookupTable
     */
    void init(VocabCache<T> vocabCache, WeightLookupTable<T> lookupTable);

    /**
     * This method implementations should return distance between two given elements
     *
     * @param label1
     * @param label2
     * @return
     */
    double similarity(String label1, String label2);

    /**
     * This method implementations should return N nearest elements labels to given element's label
     *
     * @param label label to return nearest elements for
     * @param n number of nearest words to return
     * @return
     */
    Collection<String> wordsNearest(String label, int n);
}

