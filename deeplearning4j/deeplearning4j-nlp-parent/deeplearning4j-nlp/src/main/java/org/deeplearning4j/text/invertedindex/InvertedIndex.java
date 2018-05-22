/*-
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.text.invertedindex;

import com.google.common.base.Function;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.nd4j.linalg.primitives.Pair;

import java.io.Serializable;
import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.Executor;

/**
 * An inverted index for mapping words to documents
 * and documents to words
 */
public interface InvertedIndex<T extends SequenceElement> extends Serializable {


    /**
     * Iterate over batches
     * @return the batch size
     */
    Iterator<List<List<T>>> batchIter(int batchSize);

    /**
     * Iterate over documents
     * @return
     */
    Iterator<List<T>> docs();

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
    Iterator<List<T>> miniBatches();

    /**
     * Returns a list of words for a document
     * @param index
     * @return
     */
    List<T> document(int index);

    /**
     * Returns a list of words for a document
     * and the associated label
     * @param index
     * @return
     */
    Pair<List<T>, String> documentWithLabel(int index);

    /**
     * Returns a list of words associated with the document
     * and the associated labels
     * @param index
     * @return
     */
    Pair<List<T>, Collection<String>> documentWithLabels(int index);

    /**
     * Returns the list of documents a vocab word is in
     * @param vocabWord the vocab word to get documents for
     * @return the documents for a vocab word
     */
    int[] documents(T vocabWord);

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
    void addWordToDoc(int doc, T word);


    /**
     * Adds words to the given document
     * @param doc the document to add to
     * @param words the words to add
     */
    void addWordsToDoc(int doc, List<T> words);



    /**
     * Add word to a document
     * @param doc the document to add to
     * @param word the word to add
     */
    void addLabelForDoc(int doc, T word);


    /**
     * Adds words to the given document
     * @param doc the document to add to
     *
     */
    void addLabelForDoc(int doc, String label);



    /**
     * Adds words to the given document
     * @param doc the document to add to
     * @param words the words to add
     * @param label the label for the document
     */
    void addWordsToDoc(int doc, List<T> words, String label);


    /**
     * Adds words to the given document
     * @param doc the document to add to
     * @param words the words to add
     * @param label the label for the document
     */
    void addWordsToDoc(int doc, List<T> words, T label);



    /**
     * Add word to a document
     * @param doc the document to add to
     * @param word the word to add
     */
    void addLabelsForDoc(int doc, List<T> word);


    /**
     * Adds words to the given document
     * @param doc the document to add to
     * @param label the labels to add
     *
     */
    void addLabelsForDoc(int doc, Collection<String> label);



    /**
     * Adds words to the given document
     * @param doc the document to add to
     * @param words the words to add
     * @param label the label for the document
     */
    void addWordsToDoc(int doc, List<T> words, Collection<String> label);


    /**
     * Adds words to the given document
     * @param doc the document to add to
     * @param words the words to add
     * @param label the label for the document
     */
    void addWordsToDocVocabWord(int doc, List<T> words, Collection<T> label);



    /**
     * Finishes saving data
     */
    void finish();

    /**
     * Total number of words in the index
     * @return the total number of words in the index
     */
    long totalWords();

    /**
     * For word vectors, this is the batch size for which to train on
     * @return the batch size for which to train on
     */
    int batchSize();

    /**
     * Iterate over each document with a label
     * @param func the function to apply
     * @param exec executor service for execution
     */
    void eachDocWithLabels(Function<Pair<List<T>, Collection<String>>, Void> func, Executor exec);


    /**
     * Iterate over each document with a label
     * @param func the function to apply
     * @param exec executor service for execution
     */
    void eachDocWithLabel(Function<Pair<List<T>, String>, Void> func, Executor exec);

    /**
     * Iterate over each document
     * @param func the function to apply
     * @param exec executor service for execution
     */
    void eachDoc(Function<List<T>, Void> func, Executor exec);
}
