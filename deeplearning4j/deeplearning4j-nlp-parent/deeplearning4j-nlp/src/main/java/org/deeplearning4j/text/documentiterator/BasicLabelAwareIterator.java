package org.deeplearning4j.text.documentiterator;

import lombok.NonNull;
import org.deeplearning4j.text.documentiterator.interoperability.DocumentIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;

import java.util.concurrent.atomic.AtomicLong;

/**
 * This is simple class, for building Sentence-Label pairs for ParagraphVectors/Doc2Vec.
 * Idea is simple - you provide SentenceIterator or DocumentIterator, and it builds nice structure for future model reuse
 *
 * @author raver119@gmail.com
 */
public class BasicLabelAwareIterator implements LabelAwareIterator {
    // this counter is used for dumb labels generation
    protected AtomicLong documentPosition = new AtomicLong(0);

    protected LabelsSource generator;

    protected transient LabelAwareIterator backendIterator;

    private BasicLabelAwareIterator() {

    }

    /**
     * This method checks, if there's more LabelledDocuments
     * @return
     */
    public boolean hasNextDocument() {
        return backendIterator.hasNextDocument();
    }

    /**
     * This method returns next LabelledDocument
     * @return
     */
    public LabelledDocument nextDocument() {
        return backendIterator.nextDocument();
    }

    /**
    * This methods resets LabelAwareIterator
    */
    public void reset() {
        backendIterator.reset();
    }

    /**
     * This method returns LabelsSource instance, containing all labels derived from this iterator
     * @return
     */
    @Override
    public LabelsSource getLabelsSource() {
        return generator;
    }

    @Override
    public boolean hasNext() {
        return hasNextDocument();
    }

    @Override
    public LabelledDocument next() {
        return nextDocument();
    }

    @Override
    public void shutdown() {
        // no-op
    }

    @Override
    public void remove() {
        // no-op
    }

    public static class Builder {
        private String labelTemplate = "DOC_";

        private LabelAwareIterator labelAwareIterator;
        private LabelsSource generator = new LabelsSource(labelTemplate);

        /**
         * This method should stay protected, since it's only viable for testing purposes
         */
        protected Builder() {

        }

        /**
         * We assume that each sentence in this iterator is separate document/paragraph
         *
         * @param iterator
         */
        public Builder(@NonNull SentenceIterator iterator) {
            this.labelAwareIterator = new SentenceIteratorConverter(iterator, generator);
        }

        /**
         * We assume that each inputStream in this iterator is separate document/paragraph
         * @param iterator
         */
        public Builder(@NonNull DocumentIterator iterator) {
            this.labelAwareIterator = new DocumentIteratorConverter(iterator, generator);
        }

        /**
         * We assume that each sentence in this iterator is separate document/paragraph.
         * Labels will be converted into LabelledDocument format
         *
         * @param iterator
         */
        public Builder(@NonNull LabelAwareSentenceIterator iterator) {
            this.labelAwareIterator = new SentenceIteratorConverter(iterator, generator);
        }

        /**
         * We assume that each inputStream in this iterator is separate document/paragraph
         * Labels will be converted into LabelledDocument format
         *
         * @param iterator
         */
        public Builder(@NonNull LabelAwareDocumentIterator iterator) {
            this.labelAwareIterator = new DocumentIteratorConverter(iterator, generator);
        }


        public Builder(@NonNull LabelAwareIterator iterator) {
            this.labelAwareIterator = iterator;
            this.generator = iterator.getLabelsSource();
        }

        /**
         * Label template will be used for sentence labels generation. I.e. if provided template is "DOCUMENT_", all documents/paragraphs will have their labels starting from "DOCUMENT_0" to "DOCUMENT_X", where X is the total number of documents - 1
         *
         * @param template
         * @return
         */
        public Builder setLabelTemplate(@NonNull String template) {
            this.labelTemplate = template;
            this.generator.setTemplate(template);
            return this;
        }

        /**
         * TODO: To be implemented
         *
         * @param source
         * @return
         */
        public Builder setLabelsSource(@NonNull LabelsSource source) {
            this.generator = source;
            return this;
        }

        public BasicLabelAwareIterator build() {
            BasicLabelAwareIterator iterator = new BasicLabelAwareIterator();
            iterator.generator = this.generator;
            iterator.backendIterator = this.labelAwareIterator;

            return iterator;
        }
    }
}
