package org.deeplearning4j.bagofwords.vectorizer;

import lombok.NonNull;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.documentiterator.DocumentIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.documentiterator.interoperability.DocumentIteratorConverter;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.FeatureUtil;

import java.io.BufferedReader;
import java.io.File;
import java.io.InputStream;
import java.io.InputStreamReader;
import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class TfidfVectorizer extends BaseTextVectorizer {
    /**
     * Text coming from an input stream considered as one document
     *
     * @param is    the input stream to read from
     * @param label the label to assign
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(InputStream is, String label) {
        try {
            BufferedReader reader = new BufferedReader(new InputStreamReader(is, "UTF-8"));
            String line = "";
            StringBuilder builder = new StringBuilder();
            while ((line = reader.readLine()) != null) {
                builder.append(line);
            }
            return vectorize(builder.toString(), label);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Vectorizes the passed in text treating it as one document
     *
     * @param text  the text to vectorize
     * @param label the label of the text
     * @return a dataset with a transform of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(String text, String label) {
        INDArray input = transform(text);
        INDArray labelMatrix = FeatureUtil.toOutcomeVector(labelsSource.indexOf(label), labelsSource.size());

        return new DataSet(input, labelMatrix);
    }

    /**
     * @param input the text to vectorize
     * @param label the label of the text
     * @return {@link DataSet} with a applyTransformToDestination of
     * weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(File input, String label) {
        try {
            String string = FileUtils.readFileToString(input);
            return vectorize(string, label);
        } catch (Exception e) {
            throw new RuntimeException(e);
        }
    }

    /**
     * Transforms the matrix
     *
     * @param text text to transform
     * @return {@link INDArray}
     */
    @Override
    public INDArray transform(String text) {
        INDArray ret = Nd4j.create(1, vocabCache.numWords());
        Tokenizer tokenizer = tokenizerFactory.create(text);
        List<String> tokens = tokenizer.getTokens();

        for(int i = 0;i < tokens.size(); i++) {
            int idx = vocabCache.indexOf(tokens.get(i));
            if(idx >= 0) {
                //System.out.println("TF-IDF for word: " + tokens.get(i));
                ret.putScalar(idx, tfidfWord(tokens.get(i), tokens.size()));
            }
        }
        return ret;
    }

    private double tfidfWord(String word, int documentLength) {
        return MathUtils.tfidf(tfForWord(word, documentLength),idfForWord(word));
    }

    private double tfForWord(String word, int documentLength) {
        return MathUtils.tf(vocabCache.wordFrequency(word), documentLength);
    }

    private double idfForWord(String word) {
        return MathUtils.idf(vocabCache.totalNumberOfDocs(),vocabCache.docAppearedIn(word));
    }


    /**
     * Vectorizes the input source in to a dataset
     *
     * @return Adam Gibson
     */
    @Override
    public DataSet vectorize() {
        return null;
    }

    public static class Builder {
        protected TokenizerFactory tokenizerFactory;
        protected LabelAwareIterator iterator;
        protected int minWordFrequency;
        protected VocabCache<VocabWord> vocabCache;
        protected LabelsSource labelsSource = new LabelsSource();
        protected List<String> stopWords = new ArrayList<>();

        public Builder() {
            ;
        }

        public Builder setTokenizerFactory(@NonNull TokenizerFactory tokenizerFactory) {
            this.tokenizerFactory = tokenizerFactory;
            return this;
        }

        public Builder setIterator(@NonNull LabelAwareIterator iterator) {
            this.iterator = iterator;
            return this;
        }

        public Builder setIterator(@NonNull DocumentIterator iterator) {
            this.iterator = new DocumentIteratorConverter(iterator, labelsSource);
            return this;
        }

        public Builder setIterator(@NonNull SentenceIterator iterator) {
            this.iterator = new SentenceIteratorConverter(iterator, labelsSource);
            return this;
        }

        public Builder setVocab(@NonNull VocabCache<VocabWord> vocab) {
            this.vocabCache = vocab;
            return this;
        }

        public Builder setMinWordFrequency(int minWordFrequency) {
            this.minWordFrequency = minWordFrequency;
            return this;
        }

        public Builder setStopWords(Collection<String> stopWords) {

            return this;
        }

        public TfidfVectorizer build() {
            TfidfVectorizer vectorizer = new TfidfVectorizer();

            vectorizer.tokenizerFactory = this.tokenizerFactory;
            vectorizer.iterator = this.iterator;
            vectorizer.minWordFrequency = this.minWordFrequency;
            vectorizer.labelsSource = this.labelsSource;

            if (this.vocabCache == null) {
                this.vocabCache = new AbstractCache.Builder<VocabWord>().build();
            }

            vectorizer.vocabCache = this.vocabCache;

            return vectorizer;
        }

    }
}
