package org.deeplearning4j.text.sentenceiterator.interoperability;

import lombok.NonNull;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Simple class providing compatibility between SentenceIterator/LabelAwareSentenceIterator and LabelAwareIterator
 *
 * @author raver119@gmail.com
 */
public class SentenceIteratorConverter implements LabelAwareIterator {
    private SentenceIterator backendIterator;
    private LabelsSource generator;
    protected static final Logger log = LoggerFactory.getLogger(SentenceIteratorConverter.class);

    public SentenceIteratorConverter(@NonNull SentenceIterator iterator) {
        this.backendIterator = iterator;
        this.generator = new LabelsSource();
    }

    public SentenceIteratorConverter(@NonNull SentenceIterator iterator, @NonNull LabelsSource generator) {
        this.backendIterator = iterator;
        this.generator = generator;
    }

    @Override
    public boolean hasNextDocument() {
        return backendIterator.hasNext();
    }

    @Override
    public LabelledDocument nextDocument() {
        LabelledDocument document = new LabelledDocument();

        document.setContent(backendIterator.nextSentence());
        if (backendIterator instanceof LabelAwareSentenceIterator) {
            String currentLabel = ((LabelAwareSentenceIterator) backendIterator).currentLabel();
            document.setLabel(currentLabel);
            generator.storeLabel(currentLabel);
        } else if (generator != null) document.setLabel(generator.nextLabel());

        return document;
    }

    @Override
    public void reset() {
        generator.reset();
        backendIterator.reset();
    }

    @Override
    public LabelsSource getLabelsSource() {
        return generator;
    }
}
