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
    // TODO: wtf this one is doing here?
    private List<String> stopWords = new ArrayList<>();

    // this variable defines how often scavenger will be activated
    private int scavengerThreshold  = 3000000;
    private int retentionDelay = 3;

    // for scavenger mechanics we need to know the actual number of words being added
    private transient AtomicLong hiddenWordsCounter = new AtomicLong(0);

    private AtomicLong totalWordCount = new AtomicLong(0);

    private Logger logger = LoggerFactory.getLogger(AbstractCache.class);

    private static final int MAX_CODE_LENGTH = 40;

    /**
     * Deserialize vocabulary from specified path
     */
    @Override
    public void loadVocab() {
        // TODO: this method should be static and accept path
    }

    /**
     * Returns true, if number of elements in vocabulary > 0, false otherwise
     *
     * @return
     */
    @Override
    public boolean vocabExists() {
        return vocabulary.size() > 0;
    }

    /**
     * Serialize vocabulary to specified path
     *
     */
    @Override
    public void saveVocab() {
        // TODO: this method should be static and accept path
    }

    /**
     * Returns collection of labels available in this vocabulary
     *
     * @return
     */
    @Override
    public Collection<String> words() {
        return null;
    }

    /**
     * Increment frequency for specified label by 1
     *
     * @param word the word to increment the count for
     */
    @Override
    public void incrementWordCount(String word) {

    }

    /**
     * Increment frequency for specified label by specified value
     *
     * @param word the word to increment the count for
     * @param increment the amount to increment by
     */
    @Override
    public void incrementWordCount(String word, int increment) {

    }

    /**
     * Returns the SequenceElement's frequency over training corpus
     *
     * @param word the word to retrieve the occurrence frequency for
     * @return
     */
    @Override
    public int wordFrequency(String word) {
        return 0;
    }

    /**
     * Checks, if specified label exists in vocabulary
     *
     * @param word the word to check for
     * @return
     */
    @Override
    public boolean containsWord(String word) {
        return false;
    }

    /**
     * Checks, if specified element exists in vocabulary
     *
     * @param element
     * @return
     */
    public boolean containsElement(T element) {
        return false;
    }

    /**
     * Returns the label of the element at specified Huffman index
     *
     * @param index the index of the word to get
     * @return
     */
    @Override
    public String wordAtIndex(int index) {
        return null;
    }

    /**
     * Returns Huffman index for specified label
     *
     * @param word the index of a given word
     * @return
     */
    @Override
    public int indexOf(String word) {
        return 0;
    }

    /**
     * Returns collection of SequenceElements stored in this vocabulary
     *
     * @return
     */
    @Override
    public Collection<T> vocabWords() {
        return null;
    }

    /**
     * Returns total number of elements observed
     *
     * @return
     */
    @Override
    public long totalWordOccurrences() {
        return 0;
    }

    /**
     * Returns SequenceElement for specified label
     *
     * @param word
     * @return
     */
    @Override
    public T wordFor(String word) {
        return null;
    }

    /**
     * This method allows to insert specified label to specified Huffman tree position.
     * CAUTION: Never use this, unless you 100% sure what are you doing.
     *
     * @param index
     * @param word
     */
    @Override
    public void addWordToIndex(int index, String word) {

    }

    /**
     *
     * @param word the word to add to the vocab
     */
    @Override
    @Deprecated
    public void putVocabWord(String word) {
        // TODO: to be deprecated and removed
    }

    /**
     * Returns number of elements in this vocabulary
     *
     * @return
     */
    @Override
    public int numWords() {
        return 0;
    }

    /**
     * Returns number of documents (if applicable) the label was observed in.
     *
     * @param word the number of documents the word appeared in
     * @return
     */
    @Override
    public int docAppearedIn(String word) {
        return 0;
    }

    /**
     * Increment number of documents the label was observed in
     *
     * @param word the word to increment by
     * @param howMuch
     */
    @Override
    public void incrementDocCount(String word, int howMuch) {

    }

    /**
     * Set exact number number of observed documents that contain specified word
     *
     * @param word the word to set the count for
     * @param count the count of the word
     */
    @Override
    public void setCountForDoc(String word, int count) {

    }

    /**
     * Returns total number of documents observed (if applicable)
     *
     * @return
     */
    @Override
    public int totalNumberOfDocs() {
        return 0;
    }

    /**
     * Increment total number of documents observed by 1
     */
    @Override
    public void incrementTotalDocCount() {

    }

    /**
     * Increment total number of documents observed by specified value
     */
    @Override
    public void incrementTotalDocCount(int by) {

    }

    /**
     * Returns collection of SequenceElements from this vocabulary. The same as vocabWords() method
     *
     * @return collection of SequenceElements
     */
    @Override
    public Collection<T> tokens() {
        return vocabWords();
    }

    /**
     * This method adds specified SequenceElement to vocabulary
     *
     * @param word the word to add
     */
    @Override
    public void addToken(T word) {

    }

    /**
     * Returns SequenceElement for specified label. The same as wordFor() method.
     *
     * @param label the label to get the token for
     * @return
     */
    @Override
    public T tokenFor(String label) {
        return wordFor(label);
    }

    /**
     * Checks, if specified label already exists in vocabulary. The same as containsWord() method.
     *
     * @param label the token to test
     * @return
     */
    @Override
    public boolean hasToken(String label) {
        return containsWord(label);
    }



    public static class Builder<T extends SequenceElement> {
        protected int scavengerThreshold  = 3000000;
        protected int retentionDelay = 3;
        protected int minElementFrequency;


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
            this.minElementFrequency = minFrequency;
            return this;
        }

        public AbstractCache<T> build() {
            AbstractCache<T> cache = new AbstractCache<>();
            cache.minWordFrequency = this.minElementFrequency;
            cache.scavengerThreshold = this.scavengerThreshold;
            cache.retentionDelay = this.retentionDelay;

            return cache;
        }
    }
}
