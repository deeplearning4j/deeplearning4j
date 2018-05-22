package org.deeplearning4j.iterator.provider;

import lombok.NonNull;
import org.deeplearning4j.iterator.LabeledSentenceProvider;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.nd4j.linalg.primitives.Pair;

import java.util.List;

/**
 * Simple class for conversion between LabelAwareIterator -> LabeledSentenceProvider for neural nets.
 * Since we already have converters for all other classes - this single converter allows us to accept all possible iterators
 *
 * @author raver119@gmail.com
 */
public class LabelAwareConverter implements LabeledSentenceProvider {
    private LabelAwareIterator backingIterator;
    private List<String> labels;

    public LabelAwareConverter(@NonNull LabelAwareIterator iterator, @NonNull List<String> labels) {
        this.backingIterator = iterator;
        this.labels = labels;
    }

    @Override
    public boolean hasNext() {
        return backingIterator.hasNext();
    }

    @Override
    public Pair<String, String> nextSentence() {
        LabelledDocument document = backingIterator.nextDocument();

        // TODO: probably worth to allow more then one label? i.e. pass same document twice, sequentially
        return Pair.makePair(document.getContent(), document.getLabels().get(0));
    }

    @Override
    public void reset() {
        backingIterator.reset();
    }

    @Override
    public int totalNumSentences() {
        return -1;
    }

    @Override
    public List<String> allLabels() {
        return labels;
    }

    @Override
    public int numLabelClasses() {
        return labels.size();
    }
}
