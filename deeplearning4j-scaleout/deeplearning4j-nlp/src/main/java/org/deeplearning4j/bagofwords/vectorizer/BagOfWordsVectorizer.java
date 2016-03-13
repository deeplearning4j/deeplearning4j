package org.deeplearning4j.bagofwords.vectorizer;

import lombok.Data;
import lombok.NonNull;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.StreamLineIterator;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class BagOfWordsVectorizer extends  BaseTextVectorizer {


    @Override
    public DataSet vectorize(String text, String label) {
        INDArray input = transform(text);
        INDArray labelMatrix = FeatureUtil.toOutcomeVector(labelsSource.indexOf(label), labelsSource.size());

        return new DataSet(input, labelMatrix);
    }

    @Override
    public INDArray transform(String text) {
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();
        INDArray input = Nd4j.create(1, vocabCache.numWords());
        for (String token: tokens) {
            int idx = vocabCache.indexOf(token);
            if (vocabCache.indexOf(token) >= 0)
                input.putScalar(idx, vocabCache.wordFrequency(token));
        }
        return input;
    }

    protected BagOfWordsVectorizer() {

    }

    public static class Builder {
        protected TokenizerFactory tokenizerFactory;
        protected SentenceIterator iterator;
        protected int minWordFrequency;
        protected VocabCache<VocabWord> vocabCache;

        public Builder() {
            ;
        }

        public Builder setTokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder setIterator(@NonNull DocumentIterator iterator) {
            this.iterator = new StreamLineIterator.Builder(iterator)
                    .setFetchSize(100)
                    .build();
            return this;
        }

        public Builder setIterator(@NonNull SentenceIterator iterator) {
            this.iterator = iterator;
            return this;
        }

        public Builder setVocab(@NonNull VocabCache<VocabWord> vocab) {
            this.vocabCache = vocab;
            return this;
        }

        public BagOfWordsVectorizer build() {
            BagOfWordsVectorizer vectorizer = new BagOfWordsVectorizer();

            vectorizer.tokenizerFactory = this.tokenizerFactory;
            vectorizer.iterator = this.iterator;
            vectorizer.minWordFrequency = this.minWordFrequency;

            if (this.vocabCache == null) {
                this.vocabCache = new AbstractCache.Builder<VocabWord>().build();
            }

            vectorizer.vocabCache = this.vocabCache;

            return vectorizer;
        }

    }
}
