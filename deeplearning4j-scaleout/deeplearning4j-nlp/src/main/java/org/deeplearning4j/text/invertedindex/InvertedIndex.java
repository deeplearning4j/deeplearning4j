package org.deeplearning4j.text.invertedindex;

import com.google.common.base.Function;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;

import java.io.Serializable;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.ExecutorService;

/**
 * An inverted index for mapping words to documents
 * and documents to words
 */
public interface InvertedIndex extends Serializable {


    /**
     * Iterate over batches
     * @return the batch size
     */
    Iterator<List<List<VocabWord>>> batchIter(int batchSize);

    /**
     * Iterate over documents
     * @return
     */
    Iterator<List<VocabWord>> docs();

    /**
     * Unlock the index
     */
    void unlock();

    /**
     * Cleanup any resources used
     */
    void cleanup();

    /**
     * Sampling for creating mini batches
     * @return the sampling for mini batches
     */
    double sample();

    /**
     * Iterates over mini batches
     * @return the mini batches created by this vectorizer
     */
    Iterator<List<VocabWord>> miniBatches();

    /**
     * Returns a list of words for a document
     * @param index
     * @return
     */
    List<VocabWord> document(int index);

    /**
     * Returns a list of words for a document
     * @param index
     * @return
     */
    Pair<List<VocabWord>,String> documentWithLabel(int index);


    /**
     * Returns the list of documents a vocab word is in
     * @param vocabWord the vocab word to get documents for
     * @return the documents for a vocab word
     */
    int[] documents(VocabWord vocabWord);

    /**
     * Returns the number of documents
     * @return
     */
    int numDocuments();

    /**
     * Returns a list of all documents
     * @return the list of all documents
     */
    int[] allDocs();



    /**
     * Add word to a document
     * @param doc the document to add to
     * @param word the word to add
     */
    void addWordToDoc(int doc,VocabWord word);


    /**
     * Adds words to the given document
     * @param doc the document to add to
     * @param words the words to add
     */
    void addWordsToDoc(int doc,List<VocabWord> words);



    /**
     * Add word to a document
     * @param doc the document to add to
     * @param word the word to add
     */
    void addLabelForDoc(int doc,VocabWord word);


    /**
     * Adds words to the given document
     * @param doc the document to add to
     *
     */
    void addLabelForDoc(int doc,String label);




    /**
     * Adds words to the given document
     * @param doc the document to add to
     * @param words the words to add
     * @param label the label for the document
     */
    void addWordsToDoc(int doc,List<VocabWord> words,String label);


    /**
     * Adds words to the given document
     * @param doc the document to add to
     * @param words the words to add
     * @param label the label for the document
     */
    void addWordsToDoc(int doc,List<VocabWord> words,VocabWord label);





    /**
     * Finishes saving data
     */
    void finish();

    /**
     * Total number of words in the index
     * @return the total number of words in the index
     */
    int totalWords();

    /**
     * For word vectors, this is the batch size for which to train on
     * @return the batch size for which to train on
     */
    int batchSize();

    /**
     * Iterate over each document
     * @param func the function to apply
     * @param exec executor service for execution
     */
    void eachDoc(Function<List<VocabWord>, Void> func, ExecutorService exec);
}
