package org.deeplearning4j.models.abstractvectors.transformers.impl;

import lombok.NonNull;
import org.deeplearning4j.models.abstractvectors.sequence.Sequence;
import org.deeplearning4j.models.abstractvectors.transformers.SequenceTransformer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.documentiterator.BasicLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

/**
 * This class is responsible for conversion lines of text to Sequences of SequenceElements
 *
 * @author raver119@gmail.com
 */
public class SentenceTransformer implements SequenceTransformer<VocabWord, String> {
    /*
            So, we must accept any SentenceIterator implementations, and build vocab out of it, and use it for further transforms between text and Sequences
     */
    protected TokenizerFactory tokenizerFactory;
    protected LabelAwareIterator iterator;

    private SentenceTransformer(@NonNull LabelAwareIterator iterator) {

    }

    @Override
    public VocabCache<VocabWord> derivedVocabulary() {
        return null;
    }

    @Override
    public Sequence<VocabWord> transformToSequence(String object) {
        return null;
    }

    public static class Builder {
        protected TokenizerFactory tokenizerFactory;
        protected LabelAwareIterator iterator;

        public Builder() {

        }

        public Builder tokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder iterator(@NonNull SentenceIterator iterator) {
            this.iterator = new BasicLabelAwareIterator.Builder(iterator).build();
            return this;
        }

        public Builder iterator(@NonNull DocumentIterator iterator) {
            this.iterator = new BasicLabelAwareIterator.Builder(iterator).build();
            return this;
        }

        public SentenceTransformer build() {
            SentenceTransformer transformer = new SentenceTransformer(this.iterator);
            transformer.iterator = this.iterator;

            return transformer;
        }
    }
}
