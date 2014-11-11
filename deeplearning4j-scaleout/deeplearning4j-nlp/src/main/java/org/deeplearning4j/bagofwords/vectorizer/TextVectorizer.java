package org.deeplearning4j.bagofwords.vectorizer;

import java.io.InputStream;
import java.io.File;
import java.util.Iterator;
import java.util.List;

import org.deeplearning4j.datasets.vectorizer.Vectorizer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.deeplearning4j.util.Index;

/**
 * Vectorizes text
 * @author Adam Gibson
 */
public interface TextVectorizer extends Vectorizer {


    /**
     * Sampling for building mini batches
     * @return the sampling
     */
    double sample();

    /**
     * For word vectors, this is the batch size for how to partition documents
     * in to workloads
     * @return the batchsize for partitioning documents in to workloads
     */
    int batchSize();

    /**
     * The vocab sorted in descending order
     * @return the vocab sorted in descending order
     */
    public VocabCache vocab();


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
     * Train the model
     */
    void fit();

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
    INDArray transform(String text);

    /**
     * Returns the number of words encountered so far
     * @return the number of words encountered so far
     */
    long numWordsEncountered();

    /**
     * Inverted index
     * @return the inverted index for this vectorizer
     */
    InvertedIndex index();
}
