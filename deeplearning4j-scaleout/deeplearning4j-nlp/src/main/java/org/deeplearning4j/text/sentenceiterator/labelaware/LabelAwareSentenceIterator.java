package org.deeplearning4j.text.sentenceiterator.labelaware;

import org.deeplearning4j.text.sentenceiterator.SentenceIterator;

/**
 * SentenceIterator that is aware of its label. This is useful
 * for creating datasets all at once or on the fly.
 *
 * @author Adam Gibson
 */
public interface LabelAwareSentenceIterator extends SentenceIterator {
    /**
     * Returns the current label for nextSentence()
     * @return the label for nextSentence()
     */
    String currentLabel();

}
