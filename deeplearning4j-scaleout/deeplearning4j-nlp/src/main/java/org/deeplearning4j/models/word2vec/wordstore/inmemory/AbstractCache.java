package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import org.deeplearning4j.models.abstractvectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * This is generic VocabCache implementation designed as abstract SequenceElements vocabulary
 *
 * @author raver119@gmail.com
 */
public class AbstractCache<T extends SequenceElement> implements VocabCache<T> {

    // map for label->object dictionary
    private Map<String, T> vocabulary = new ConcurrentHashMap<>();

    private Map<Integer, T> idxMap = new ConcurrentHashMap<>();

    private int minWordFrequency = 0;
    private boolean hugeModelExpected = false;


    // we're using <String>for compatibility & failproof reasons: it's easier to store unique labels then abstract objects of unknown size
    private List<String> stopWords = new ArrayList<>();

    // this variable defines how often scavenger will be activated
    private int scavengerThreshold  = 3000000;
    private int retentionDelay = 3;

    // for scavenger mechanics we need to know the actual number of words being added
    private transient AtomicLong hiddenWordsCounter = new AtomicLong(0);

    private AtomicLong totalWordCount = new AtomicLong(0);

    private Logger logger = LoggerFactory.getLogger(AbstractCache.class);

    private static final int MAX_CODE_LENGTH = 40;

    @Override
    public void loadVocab() {
        // TODO: this method should be static and accept path
    }

    @Override
    public boolean vocabExists() {
        return false;
    }

    @Override
    public void saveVocab() {
        // TODO: this method should be static and accept path
    }

    @Override
    public Collection<String> words() {
        return null;
    }

    @Override
    public void incrementWordCount(String word) {

    }

    @Override
    public void incrementWordCount(String word, int increment) {

    }

    @Override
    public int wordFrequency(String word) {
        return 0;
    }

    @Override
    public boolean containsWord(String word) {
        return false;
    }

    @Override
    public String wordAtIndex(int index) {
        return null;
    }

    @Override
    public int indexOf(String word) {
        return 0;
    }

    @Override
    public Collection<T> vocabWords() {
        return null;
    }

    @Override
    public long totalWordOccurrences() {
        return 0;
    }

    @Override
    public <T1 extends SequenceElement> T1 wordFor(String word) {
        return null;
    }

    @Override
    public void addWordToIndex(int index, String word) {

    }

    @Override
    public void putVocabWord(String word) {

    }

    @Override
    public int numWords() {
        return 0;
    }

    @Override
    public int docAppearedIn(String word) {
        return 0;
    }

    @Override
    public void incrementDocCount(String word, int howMuch) {

    }

    @Override
    public void setCountForDoc(String word, int count) {

    }

    @Override
    public int totalNumberOfDocs() {
        return 0;
    }

    @Override
    public void incrementTotalDocCount() {

    }

    @Override
    public void incrementTotalDocCount(int by) {

    }

    @Override
    public Collection<T> tokens() {
        return null;
    }

    @Override
    public void addToken(T word) {

    }

    @Override
    public T tokenFor(String word) {
        return null;
    }

    @Override
    public boolean hasToken(String token) {
        return false;
    }

    public static class Builder<T extends SequenceElement> {

        public Builder<T> hugeModelExpected(boolean reallyExpected) {

            return this;
        }

        public Builder<T> scavengerThreshold(int threshold) {

            return this;
        }

        public Builder<T> scavengerRetentionDelay(int delay) {

            return this;
        }

        public Builder<T> minElementFrequency(int minFrequency) {

            return this;
        }
    }
}
