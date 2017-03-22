package org.deeplearning4j.text.documentiterator;

import lombok.NonNull;

import java.util.Iterator;

/**
 * This class provide option to build LabelAwareIterator from Iterable<LabelledDocument> or Iterator<LabelledDocument> objects
 *
 * PLEASE NOTE: This iterator is meant to be used with externally-originated data via Java Iterable/Iterator interface.
 * It IS possible to use Collection/List object here, but it's NOT recommended, since huge List with data might cause significant
 * performance penalty due to JVM Garbage Collection mechanics.
 *
 * @author raver119@gmail.com
 */
public class SimpleLabelAwareIterator implements LabelAwareIterator {
    protected transient Iterable<LabelledDocument> underlyingIterable;
    protected transient Iterator<LabelledDocument> currentIterator;
    protected LabelsSource labels = new LabelsSource();

    /**
     * Builds LabelAwareIterator instance using Iterable object
     * @param iterable
     */
    public SimpleLabelAwareIterator(@NonNull Iterable<LabelledDocument> iterable) {
        this.underlyingIterable = iterable;
        this.currentIterator = underlyingIterable.iterator();
    }

    /**
     * Builds LabelAwareIterator instance using Iterator object
     * PLEASE NOTE: If instance is built using Iterator object, reset() method becomes unavailable
     *
     * @param iterator
     */
    public SimpleLabelAwareIterator(@NonNull Iterator<LabelledDocument> iterator) {
        this.currentIterator = iterator;
    }

    /**
     * This method checks, if there's more LabelledDocuments in underlying iterator
     * @return
     */
    @Override
    public boolean hasNextDocument() {
        return currentIterator.hasNext();
    }

    /**
     * This method returns next LabelledDocument from underlying iterator
     * @return
     */
    @Override
    public LabelledDocument nextDocument() {
        LabelledDocument document = currentIterator.next();
        for (String label : document.getLabels()) {
            labels.storeLabel(label);
        }

        return document;
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
    public void remove() {
        // no-op
    }

    @Override
    public void shutdown() {
        // no-op
    }

    /**
     * This methods resets LabelAwareIterator by creating new Iterator from Iterable internally
     */
    @Override
    public void reset() {
        if (underlyingIterable != null)
            this.currentIterator = this.underlyingIterable.iterator();
        else
            throw new UnsupportedOperationException(
                            "You can't use reset() method for Iterator<> based instance, please provide Iterable<> instead, or avoid reset()");
    }

    /**
     * This method returns LabelsSource instance, containing all labels derived from this iterator
     * @return
     */
    @Override
    public LabelsSource getLabelsSource() {
        return labels;
    }
}
