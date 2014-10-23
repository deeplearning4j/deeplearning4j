package org.deeplearning4j.text.documentiterator;

/**
 * Created by agibsonccc on 10/18/14.
 */
public interface LabelAwareDocumentIterator extends DocumentIterator {


    /**
     * Returns the current label
     * @return
     */
    public String currentLabel();

}
