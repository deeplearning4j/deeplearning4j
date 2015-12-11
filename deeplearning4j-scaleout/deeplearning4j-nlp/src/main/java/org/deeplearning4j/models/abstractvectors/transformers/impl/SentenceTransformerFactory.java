package org.deeplearning4j.models.abstractvectors.transformers.impl;

import lombok.NonNull;
import org.deeplearning4j.models.abstractvectors.transformers.SequenceTransformer;
import org.deeplearning4j.models.abstractvectors.transformers.TransformerFactory;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;

/**
 * Created by fartovii on 11.12.15.
 */
public class SentenceTransformerFactory implements TransformerFactory<VocabWord, String> {

    protected TokenizerFactory tokenizerFactory;

    protected SentenceTransformerFactory() {

    }

    @Override
    public SequenceTransformer<VocabWord, String> getLearningTransformer(VocabCache<VocabWord> vocabCache) {
        return new SentenceTransformer.Builder()
                .readOnly(false)
                .tokenizerFactory(tokenizerFactory)
                .build();
    }

    @Override
    public SequenceTransformer<VocabWord, String> getUnmodifiableTransformer(VocabCache<VocabWord> vocabCache) {
        return new SentenceTransformer.Builder()
                .readOnly(true)
                .tokenizerFactory(tokenizerFactory)
                .build();
    }

    public static class Builder {
        protected TokenizerFactory tokenizerFactory;

        public Builder() {

        }

        public Builder tokenizerFactory(@NonNull TokenizerFactory factory) {
            this.tokenizerFactory = factory;
            return this;
        }

        public SentenceTransformerFactory build() {
            SentenceTransformerFactory factory = new SentenceTransformerFactory();
            factory.tokenizerFactory = this.tokenizerFactory;

            return factory;
        }
    }
}
