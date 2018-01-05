package org.deeplearning4j.iterator;

import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 *
 * LabeledSentenceProvider: a simple iterator interface over sentences/documents that have a label.<br>
 *
 * This is intended for use with {@link CnnSentenceDataSetIterator}
 *
 * @author Alex Black
 */
public interface LabeledSentenceProvider {

    /**
     * Are there more sentences/documents available?
     */
    boolean hasNext();

    /**
     *
     * @return Pair: sentence/document text and label
     */
    Pair<String, String> nextSentence();

    /**
     * Reset the iterator - including shuffling the order, if necessary/appropriate
     */
    void reset();

    /**
     * Return the total number of sentences, or -1 if not available
     */
    int totalNumSentences();

    /**
     * Return the list of labels - this also defines the class/integer label assignment order
     */
    List<String> allLabels();

    /**
     * Equivalent to allLabels().size()
     */
    int numLabelClasses();

}
