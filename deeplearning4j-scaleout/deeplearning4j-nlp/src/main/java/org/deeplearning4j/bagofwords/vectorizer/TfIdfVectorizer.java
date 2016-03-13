package org.deeplearning4j.bagofwords.vectorizer;

import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.util.MathUtils;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.io.InputStream;
import java.util.List;

/**
 * @author raver119@gmail.com
 */
public class TfIdfVectorizer extends BaseTextVectorizer {
    /**
     * Text coming from an input stream considered as one document
     *
     * @param is    the input stream to read from
     * @param label the label to assign
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(InputStream is, String label) {
        return null;
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
        return null;
    }

    /**
     * @param input the text to vectorize
     * @param label the label of the text
     * @return {@link DataSet} with a applyTransformToDestination of
     * weights(relative to impl; could be word counts or tfidf scores)
     */
    @Override
    public DataSet vectorize(File input, String label) {
        return null;
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
            if(idx >= 0)
                ret.putScalar(idx, tfidfWord(tokens.get(i)));
        }
        return ret;
    }

    private double tfidfWord(String word) {
        return MathUtils.tfidf(tfForWord(word),idfForWord(word));
    }

    private double tfForWord(String word) {
        return MathUtils.tf(vocabCache.wordFrequency(word));
    }

    private double idfForWord(String word) {
        return MathUtils.idf(vocabCache.totalNumberOfDocs(),vocabCache.docAppearedIn(word));
    }

    /**
     * Returns the number of words encountered so far
     *
     * @return the number of words encountered so far
     */
    @Override
    public long numWordsEncountered() {
        return vocabCache.numWords();
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
}
