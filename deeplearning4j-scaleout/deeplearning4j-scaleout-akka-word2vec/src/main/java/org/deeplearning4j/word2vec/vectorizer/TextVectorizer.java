package org.deeplearning4j.word2vec.vectorizer;

import java.io.InputStream;
import java.io.File;

import org.deeplearning4j.datasets.DataSet;
import org.deeplearning4j.datasets.vectorizer.Vectorizer;
import org.deeplearning4j.util.Index;
import org.jblas.DoubleMatrix;

/**
 * Vectorizes text
 * @author Adam Gibson
 */
public interface TextVectorizer extends Vectorizer {

    /**
     * The vocab sorted in descending order
     * @return the vocab sorted in descending order
     */
    public Index vocab();


    /**
     * Text coming from an input stream considered as one document
     * @param is the input stream to read from
     * @param label the label to assign
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    DataSet vectorize(InputStream is,String label);

    /**
     * Vectorizes the passed in text treating it as one document
     * @param text the text to vectorize
     * @param label the label of the text
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    DataSet vectorize(String text,String label);

    /**
     *
     * @param input the text to vectorize
     * @param label the label of the text
     * @return a dataset with a applyTransformToDestination of weights(relative to impl; could be word counts or tfidf scores)
     */
    DataSet vectorize(File input,String label);


    /**
     * Transforms the matrix
     * @param text
     * @return
     */
    DoubleMatrix transform(String text);

}
