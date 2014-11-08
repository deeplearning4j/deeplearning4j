package org.deeplearning4j.text.invertedindex;

import org.deeplearning4j.models.word2vec.VocabWord;

import java.io.Serializable;
import java.util.List;

/**
 * An inverted index for mapping words to documents
 * and documents to words
 */
public interface InvertedIndex extends Serializable {

    /**
     * Returns a list of words for a document
     * @param index
     * @return
     */
    List<VocabWord> document(int index);

    /**
     * Returns the list of documents a vocab word is in
     * @param vocabWord the vocab word to get documents for
     * @return the documents for a vocab word
     */
    List<Integer> documents(VocabWord vocabWord);

    /**
     * Returns the number of documents
     * @return
     */
    int numDocuments();

    /**
     * Returns a list of all documents
     * @return the list of all documents
     */
    java.util.Collection<Integer> allDocs();



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
     * Finishes saving data
     */
    void finish();

    /**
     * Total number of words in the index
     * @return the total number of words in the index
     */
    int totalWords();



}
