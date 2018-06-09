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

package org.deeplearning4j.models.word2vec.wordstore;


import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;

import java.io.Serializable;
import java.util.Collection;


/**
 * A VocabCache handles the storage of information needed for the word2vec look up table.
 *
 * @author Adam Gibson
 */
public interface VocabCache<T extends SequenceElement> extends Serializable {


    /**
     * Load vocab
     */
    void loadVocab();


    /**
     * Vocab exists already
     * @return
     */
    boolean vocabExists();

    /**
     * Saves the vocab: this allow for reuse of word frequencies	
     */
    void saveVocab();


    /**
     * Returns all of the words in the vocab
     * @returns all the words in the vocab
     */
    Collection<String> words();


    /**
     * Increment the count for the given word
     * @param word the word to increment the count for
     */
    void incrementWordCount(String word);


    /**
     * Increment the count for the given word by
     * the amount increment
     * @param word the word to increment the count for
     * @param increment the amount to increment by
     */
    void incrementWordCount(String word, int increment);

    /**
     * Returns the number of times the word has occurred
     * @param word the word to retrieve the occurrence frequency for
     * @return 0 if hasn't occurred or the number of times
     * the word occurs
     */
    int wordFrequency(String word);

    /**
     * Returns true if the cache contains the given word
     * @param word the word to check for
     * @return
     */
    boolean containsWord(String word);

    /**
     * Returns the word contained at the given index or null
     * @param index the index of the word to get
     * @return the word at the given index
     */
    String wordAtIndex(int index);

    /**
     * Returns SequenceElement at the given index or null
     *
     * @param index
     * @return
     */
    T elementAtIndex(int index);

    /**
     * Returns the index of a given word
     * @param word the index of a given word
     * @return the index of a given word or -1
     * if not found
     */
    int indexOf(String word);


    /**
     * Returns all of the vocab word nodes
     * @return
     */
    Collection<T> vocabWords();


    /**
     * The total number of word occurrences
     * @return the total number of word occurrences
     */
    long totalWordOccurrences();


    /**
     *
     * @param word
     * @return
     */
    T wordFor(String word);


    T wordFor(long id);

    /**
     *
     * @param index
     * @param word
     */
    void addWordToIndex(int index, String word);


    void addWordToIndex(int index, long elementId);

    /**
     * Inserts the word as a vocab word
     * (it gets the vocab word from the internal token store).
     * Note that the index must be set on the token.
     * @param word the word to add to the vocab
     */
    @Deprecated
    void putVocabWord(String word);

    /**
     * Returns the number of words in the cache
     * @return the number of words in the cache
     */
    int numWords();


    /**
     * Count of documents a word appeared in
     * @param word the number of documents the word appeared in
     * @return
     */
    int docAppearedIn(String word);

    /**
     * Increment the document count
     * @param word the word to increment by
     * @param howMuch
     */
    void incrementDocCount(String word, long howMuch);


    /**
     * Set the count for the number of documents the word appears in
     * @param word the word to set the count for
     * @param count the count of the word
     */
    void setCountForDoc(String word, long count);

    /**
     * Returns the total of number of documents encountered in the corpus
     * @return the total number of docs in the corpus
     */
    long totalNumberOfDocs();


    /**
     * Increment the doc count
     */
    void incrementTotalDocCount();

    /**
     * Increment the doc count
     * @param  by the number to increment by
     */
    void incrementTotalDocCount(long by);

    /**
     * All of the tokens in the cache, (not necessarily apart of the vocab)
     * @return the tokens for this cache
     */
    Collection<T> tokens();


    /**
     * Adds a token
     * to the cache
     * @param element the word to add
     */
    void addToken(T element);

    /**
     * Returns the token (again not necessarily in the vocab)
     * for this word
     * @param word the word to get the token for
     * @return the vocab word for this token
     */
    T tokenFor(String word);

    T tokenFor(long id);

    /**
     * Returns whether the cache
     * contains this token or not
     * @param token the token to tes
     * @return whether the token exists in
     * the cache or not
     *
     */
    boolean hasToken(String token);


    /**
     * imports vocabulary
     *
     * @param vocabCache
     */
    void importVocabulary(VocabCache<T> vocabCache);

    /**
     * Updates counters
     */
    void updateWordsOccurrences();

    /**
     * Removes element with specified label from vocabulary
     * Please note: Huffman index should be updated after element removal
     *
     * @param label label of the element to be removed
     */
    void removeElement(String label);


    /**
     * Removes specified element from vocabulary
     * Please note: Huffman index should be updated after element removal
     *
     * @param element SequenceElement to be removed
     */
    void removeElement(T element);
}
