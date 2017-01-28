package org.deeplearning4j.iterator;

import org.deeplearning4j.berkeley.Pair;

import java.util.List;

/**
 * Created by Alex on 27/01/2017.
 */
public interface LabeledSentenceProvider {


    boolean hasNext();

    /**
     *
     * @return Pair: sentence text and label
     */
    Pair<String,String> nextSentence();

    void reset();

    /**
     * Return the total number of sentences, or -1 if not available
     */
    int totalNumSentences();

    List<String> allLabels();

    int numLabelClasses();

}
