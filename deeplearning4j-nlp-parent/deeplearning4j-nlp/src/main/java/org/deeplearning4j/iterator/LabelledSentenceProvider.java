package org.deeplearning4j.iterator;

import org.deeplearning4j.berkeley.Pair;

import java.util.List;

/**
 * Created by Alex on 27/01/2017.
 */
public interface LabelledSentenceProvider {


    boolean hasNext();

    /**
     *
     * @return Pair: sentence text and label
     */
    Pair<String,String> nextSentence();

    void reset();

    List<String> allLabels();

    int numLabelClasses();

}
