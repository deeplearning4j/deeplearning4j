package org.deeplearning4j.text.documentiterator;

/**
 * This simple iterator interface assumes, that all documents are packed into strings OR into references to VocabWords.
 * Basic idea is: for tasks like ParagraphVectors we need unified interface for reading Sentences (read: lines of text) or Documents (read: set of lines) with label support.
 *
 * There's 2 interoperbility implementations of this interfaces: SentenceIteratorConverter and DocumentIteratorConverter.
 * After conversion is done, they can be wrapped by BasicLabelAwareIterator, that accepts all 5 current interfaces (including this one) as source for labelled documents.
 * This way 100% backward compatibility is provided, as well as additional functionality is delivered via LabelsSource.
 *
 * @author raver119@gmail.com
 */
public interface LabelAwareIterator {

    boolean hasNextDocument();

    LabelledDocument nextDocument();

    void reset();

    LabelsSource getLabelsSource();
}
