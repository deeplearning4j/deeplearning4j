/*
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

package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.movingwindow.Util;
import org.deeplearning4j.util.Index;
import org.deeplearning4j.util.SerializationUtils;


import java.io.*;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * In memory lookup cache for smaller datasets
 *
 * @author Adam Gibson
 */
public class InMemoryLookupCache implements VocabCache<VocabWord>,Serializable {

    private Index wordIndex = new Index();
    public Counter<String> wordFrequencies = Util.parallelCounter();
    public Counter<String> docFrequencies = Util.parallelCounter();
    public Map<String, VocabWord> vocabs = new ConcurrentHashMap<>();
    public Map<String, VocabWord> tokens = new ConcurrentHashMap<>();
    private AtomicLong totalWordOccurrences = new AtomicLong(0);
    private int numDocs = 0;

    public synchronized void setWordFrequencies(Counter <String> cnt) {
        this.wordFrequencies = cnt;
    }
    public synchronized Map<String, VocabWord> getVocabs() {
        return this.vocabs;
    }

    public synchronized void setVocabs(Map<String, VocabWord> vocabs) {
        this.vocabs = vocabs;
    }
    public synchronized Counter<String> getWordFrequencies() {
        return this.wordFrequencies;
    }

    public synchronized void setTokens(Map<String, VocabWord> tokens) {
        this.tokens = tokens;
    }
    public synchronized Map<String, VocabWord> getTokens() {
        return this.tokens;
    }

    public InMemoryLookupCache() {
      //  this(false);
    }

    @Deprecated
    public InMemoryLookupCache(boolean addUnk) {
        /*if(addUnk) {
            T word = (T) new SequenceElement(); //VocabWord(1.0, Word2Vec.UNK);
            word.setIndex(0);
            addToken(word);
            addWordToIndex(0, Word2Vec.UNK);
            putVocabWord(Word2Vec.UNK);
        }
        */
    }

    /**
     * Returns all of the words in the vocab
     *
     * @returns all the words in the vocab
     */
    @Override
    public synchronized Collection<String> words() {
        return vocabs.keySet();
    }


    /**
     * Increment the count for the given word
     *
     * @param word the word to increment the count for
     */
    @Override
    public synchronized void incrementWordCount(String word) {
        incrementWordCount(word, 1);
    }

    /**
     * Increment the count for the given word by
     * the amount increment
     *
     * @param word      the word to increment the count for
     * @param increment the amount to increment by
     */
    @Override
    public synchronized void incrementWordCount(String word, int increment) {
        if(word == null || word.isEmpty())
            throw new IllegalArgumentException("Word can't be empty or null");
        wordFrequencies.incrementCount(word, increment);

        if(hasToken(word)) {
            VocabWord token = tokenFor(word);
            token.increaseElementFrequency(increment);
        }
        totalWordOccurrences.set(totalWordOccurrences.get() + increment);
    }

    /**
     * Returns the number of times the word has occurred
     *
     * @param word the word to retrieve the occurrence frequency for
     * @return 0 if hasn't occurred or the number of times
     * the word occurs
     */
    @Override
    public synchronized int wordFrequency(String word) {
        return (int) wordFrequencies.getCount(word);
    }

    /**
     * Returns true if the cache contains the given word
     *
     * @param word the word to check for
     * @return
     */
    @Override
    public synchronized boolean containsWord(String word) {
        return vocabs.containsKey(word);
    }

    /**
     * Returns the word contained at the given index or null
     *
     * @param index the index of the word to get
     * @return the word at the given index
     */
    @Override
    public synchronized String wordAtIndex(int index) {
        return (String) wordIndex.get(index);
    }

    @Override
    public VocabWord elementAtIndex(int index) {
        return wordFor(wordAtIndex(index));
    }

    /**
     * Returns the index of a given word
     *
     * @param word the index of a given word
     * @return the index of a given word or -1
     * if not found
     */
    @Override
    public synchronized int indexOf(String word) {
        return wordFor(word).getIndex();
    }


    /**
     * Returns all of the vocab word nodes
     *
     * @return
     */
    @Override
    public synchronized Collection<VocabWord> vocabWords() {
        return vocabs.values();
    }

    /**
     * The total number of word occurrences
     *
     * @return the total number of word occurrences
     */
    @Override
    public synchronized long totalWordOccurrences() {
        return  totalWordOccurrences.get();
    }



    /**
     * @param word
     * @return
     */
    @Override
    public  synchronized  VocabWord wordFor(String word) {
        if(word == null)
            return null;
        VocabWord ret =  vocabs.get(word);
        return ret;
    }

    /**
     * @param index
     * @param word
     */
    @Override
    public synchronized void addWordToIndex(int index, String word) {
        if(word == null || word.isEmpty())
            throw new IllegalArgumentException("Word can't be empty or null");



        if(!tokens.containsKey(word)) {
            VocabWord token = new VocabWord(1.0, word);
            tokens.put(word, token);
            wordFrequencies.incrementCount(word, 1.0);
        }

        /*
            If we're speaking about adding any word to index directly, it means it's going to be vocab word, not token
         */
        if (!vocabs.containsKey(word)) {
            VocabWord vw = tokenFor(word);
            vw.setIndex(index);
            vocabs.put(word, vw);
            vw.setIndex(index);
        }

        if (!wordFrequencies.containsKey(word))
                wordFrequencies.incrementCount(word, 1);

            wordIndex.add(word, index);

    }

    /**
     * @param word
     */
    @Override
    @Deprecated
    public synchronized void putVocabWord(String word) {
        if(word == null || word.isEmpty())
            throw new IllegalArgumentException("Word can't be empty or null");
        // STOP and UNK are not added as tokens
        if(word.equals("STOP") || word.equals("UNK"))
            return;
        VocabWord token = tokenFor(word);
        if(token == null)
            throw new IllegalStateException("Word " + word + " not found as token in vocab");
        int ind = token.getIndex();
        addWordToIndex(ind, word);
        if(!hasToken(word))
            throw new IllegalStateException("Unable to add token " + word + " when not already a token");
        vocabs.put(word,token);
        wordIndex.add(word,token.getIndex());
    }


    /**
     * Returns the number of words in the cache
     *
     * @return the number of words in the cache
     */
    @Override
    public synchronized int numWords() {
        return vocabs.size();
    }

    @Override
    public synchronized int docAppearedIn(String word) {
        return (int) docFrequencies.getCount(word);
    }

    @Override
    public synchronized void incrementDocCount(String word, int howMuch) {
        docFrequencies.incrementCount(word, howMuch);
    }

    @Override
    public synchronized void setCountForDoc(String word, int count) {
        docFrequencies.setCount(word, count);
    }

    @Override
    public synchronized int totalNumberOfDocs() {
        return numDocs;
    }

    @Override
    public synchronized void incrementTotalDocCount() {
        numDocs++;
    }

    @Override
    public synchronized void incrementTotalDocCount(int by) {
        numDocs += by;
    }

    @Override
    public synchronized Collection<VocabWord> tokens() {
        return tokens.values();
    }

    @Override
    public synchronized void addToken(VocabWord word) {
        tokens.put(word.getLabel(),word);
    }

    @Override
    public synchronized VocabWord tokenFor(String word) {
        return tokens.get(word);
    }

    @Override
    public synchronized boolean hasToken(String token) {
        return tokenFor(token) != null;
    }

    @Override
    public void importVocabulary(VocabCache<VocabWord> vocabCache) {
        for (VocabWord word: vocabCache.vocabWords()) {
            if (vocabs.containsKey(word.getLabel())) {
                wordFrequencies.incrementCount(word.getLabel(), word.getElementFrequency());
            } else {
                tokens.put(word.getLabel(), word);
                vocabs.put(word.getLabel(), word);
                wordFrequencies.incrementCount(word.getLabel(), word.getElementFrequency());
            }
            totalWordOccurrences.addAndGet((long) word.getElementFrequency());
        }
    }

    @Override
    public void updateWordsOccurencies() {
        totalWordOccurrences.set(0);
        for (VocabWord word: vocabWords()) {
            totalWordOccurrences.addAndGet((long) word.getElementFrequency());
        }
    }

    @Override
    public void removeElement(String label) {
        if (vocabs.containsKey(label)) {
            vocabs.remove(label);
        }

        if (tokens.containsKey(label)) {
            tokens.remove(label);
        }
    }

    @Override
    public void removeElement(VocabWord element) {
        removeElement(element.getLabel());
    }


    @Override
    public synchronized void saveVocab() {
        SerializationUtils.saveObject(this, new File("ser"));
    }

    @Override
    public synchronized boolean vocabExists() {
        return new File("ser").exists();
    }


    /**
     * Load a look up cache from an input stream
     * delimited by \n
     * @param from the input stream to read from
     * @return the in memory lookup cache
     */
    public static InMemoryLookupCache load(InputStream from) {
        /*
        Reader inputStream = new InputStreamReader(from);
        LineIterator iter = IOUtils.lineIterator(inputStream);
        String line;
        InMemoryLookupCache ret = new InMemoryLookupCache();
        int count = 0;
        while((iter.hasNext())) {
            line = iter.nextLine();
            if(line.isEmpty())
                continue;
            ret.incrementWordCount(line);
            VocabWord word = new VocabWord(1.0,line);
            word.setIndex(count);
            ret.addToken((SequenceElement) word);
            ret.addWordToIndex(count,line);
            ret.putVocabWord(line);
            count++;

        }

        return ret; */
        return null;
    }

    @Override
    public synchronized void loadVocab() {
        InMemoryLookupCache cache = SerializationUtils.readObject(new File("ser"));
        this.vocabs = cache.vocabs;
        this.wordFrequencies = cache.wordFrequencies;
        this.wordIndex = cache.wordIndex;
        this.tokens = cache.tokens;


    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        InMemoryLookupCache that = (InMemoryLookupCache) o;

        if (numDocs != that.numDocs) return false;
        if (wordIndex != null ? !wordIndex.equals(that.wordIndex) : that.wordIndex != null) return false;
        if (wordFrequencies != null ? !wordFrequencies.equals(that.wordFrequencies) : that.wordFrequencies != null)
            return false;
        if (docFrequencies != null ? !docFrequencies.equals(that.docFrequencies) : that.docFrequencies != null)
            return false;
        if(vocabWords().equals(that.vocabWords()))
            return true;

        return true;

    }

    @Override
    public int hashCode() {
        int result = wordIndex != null ? wordIndex.hashCode() : 0;
        result = 31 * result + (wordFrequencies != null ? wordFrequencies.hashCode() : 0);
        result = 31 * result + (docFrequencies != null ? docFrequencies.hashCode() : 0);
        result = 31 * result + (vocabs != null ? vocabs.hashCode() : 0);
        result = 31 * result + (tokens != null ? tokens.hashCode() : 0);
        result = 31 * result + (totalWordOccurrences != null ? totalWordOccurrences.hashCode() : 0);
        result = 31 * result + numDocs;
        return result;
    }

    @Override
    public String toString() {
        return "InMemoryLookupCache{" +
                "wordIndex=" + wordIndex +
                ", wordFrequencies=" + wordFrequencies +
                ", docFrequencies=" + docFrequencies +
                ", vocabs=" + vocabs +
                ", tokens=" + tokens +
                ", totalWordOccurrences=" + totalWordOccurrences +
                ", numDocs=" + numDocs +
                '}';
    }
}
