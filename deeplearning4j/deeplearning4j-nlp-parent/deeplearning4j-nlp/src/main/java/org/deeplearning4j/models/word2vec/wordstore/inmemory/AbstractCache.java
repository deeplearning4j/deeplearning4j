package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.ConcurrentMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 *
 * This is generic VocabCache implementation designed as abstract SequenceElements vocabulary
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class AbstractCache<T extends SequenceElement> implements VocabCache<T> {

    // map for label->object dictionary
    private final ConcurrentMap<Long, T> vocabulary = new ConcurrentHashMap<>();

    private final Map<String, T> extendedVocabulary = new ConcurrentHashMap<>();

    private final Map<Integer, T> idxMap = new ConcurrentHashMap<>();

    private final AtomicLong documentsCounter = new AtomicLong(0);

    private int minWordFrequency = 0;
    private boolean hugeModelExpected = false;


    // we're using <String>for compatibility & failproof reasons: it's easier to store unique labels then abstract objects of unknown size
    // TODO: wtf this one is doing here?
    private List<String> stopWords = new ArrayList<>();

    // this variable defines how often scavenger will be activated
    private int scavengerThreshold = 3000000;
    private int retentionDelay = 3;

    // for scavenger mechanics we need to know the actual number of words being added
    private transient AtomicLong hiddenWordsCounter = new AtomicLong(0);

    private final AtomicLong totalWordCount = new AtomicLong(0);



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
        return !vocabulary.isEmpty();
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
        return Collections.unmodifiableCollection(extendedVocabulary.keySet());
    }

    /**
     * Increment frequency for specified label by 1
     *
     * @param word the word to increment the count for
     */
    @Override
    public void incrementWordCount(String word) {
        incrementWordCount(word, 1);
    }


    /**
     * Increment frequency for specified label by specified value
     *
     * @param word the word to increment the count for
     * @param increment the amount to increment by
     */
    @Override
    public void incrementWordCount(String word, int increment) {
        T element = extendedVocabulary.get(word);
        if (element != null) {
            element.increaseElementFrequency(increment);
            totalWordCount.addAndGet(increment);
        }
    }

    /**
     * Returns the SequenceElement's frequency over training corpus
     *
     * @param word the word to retrieve the occurrence frequency for
     * @return
     */
    @Override
    public int wordFrequency(@NonNull String word) {
        // TODO: proper wordFrequency impl should return long, instead of int
        T element = extendedVocabulary.get(word);
        if (element != null)
            return (int) element.getElementFrequency();
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
        return extendedVocabulary.containsKey(word);
    }

    /**
     * Checks, if specified element exists in vocabulary
     *
     * @param element
     * @return
     */
    public boolean containsElement(T element) {
        // FIXME: lolwtf
        return vocabulary.values().contains(element);
    }

    /**
     * Returns the label of the element at specified Huffman index
     *
     * @param index the index of the word to get
     * @return
     */
    @Override
    public String wordAtIndex(int index) {
        T element = idxMap.get(index);
        if (element != null) {
            return element.getLabel();
        }
        return null;
    }

    /**
     * Returns SequenceElement at specified index
     *
     * @param index
     * @return
     */
    @Override
    public T elementAtIndex(int index) {
        return idxMap.get(index);
    }

    /**
     * Returns Huffman index for specified label
     *
     * @param label the label to get index for
     * @return >=0 if label exists, -1 if Huffman tree wasn't built yet, -2 if specified label wasn't found
     */
    @Override
    public int indexOf(String label) {
        T token = tokenFor(label);
        if (token != null) {
            return token.getIndex();
        } else
            return -2;
    }

    /**
     * Returns collection of SequenceElements stored in this vocabulary
     *
     * @return
     */
    @Override
    public Collection<T> vocabWords() {
        return vocabulary.values();
    }

    /**
     * Returns total number of elements observed
     *
     * @return
     */
    @Override
    public long totalWordOccurrences() {
        return totalWordCount.get();
    }

    /**
     * Returns SequenceElement for specified label
     *
     * @param label to fetch element for
     * @return
     */
    @Override
    public T wordFor(@NonNull String label) {
        return extendedVocabulary.get(label);
    }

    @Override
    public T wordFor(long id) {
        return vocabulary.get(id);
    }

    /**
     * This method allows to insert specified label to specified Huffman tree position.
     * CAUTION: Never use this, unless you 100% sure what are you doing.
     *
     * @param index
     * @param label
     */
    @Override
    public void addWordToIndex(int index, String label) {
        if (index >= 0) {
            T token = tokenFor(label);
            if (token != null) {
                idxMap.put(index, token);
                token.setIndex(index);
            }
        }
    }

    @Override
    public void addWordToIndex(int index, long elementId) {
        if (index >= 0)
            idxMap.put(index, tokenFor(elementId));
    }

    @Override
    @Deprecated
    public void putVocabWord(String word) {
        if (!containsWord(word))
            throw new IllegalStateException("Specified label is not present in vocabulary");
    }

    /**
     * Returns number of elements in this vocabulary
     *
     * @return
     */
    @Override
    public int numWords() {
        return vocabulary.size();
    }

    /**
     * Returns number of documents (if applicable) the label was observed in.
     *
     * @param word the number of documents the word appeared in
     * @return
     */
    @Override
    public int docAppearedIn(String word) {
        T element = extendedVocabulary.get(word);
        if (element != null) {
            return (int) element.getSequencesCount();
        } else
            return -1;
    }

    /**
     * Increment number of documents the label was observed in
     *
     * Please note: this method is NOT thread-safe
     *
     * @param word the word to increment by
     * @param howMuch
     */
    @Override
    public void incrementDocCount(String word, long howMuch) {
        T element = extendedVocabulary.get(word);
        if (element != null) {
            element.incrementSequencesCount();
        }
    }

    /**
     * Set exact number of observed documents that contain specified word
     *
     * Please note: this method is NOT thread-safe
     *
     * @param word the word to set the count for
     * @param count the count of the word
     */
    @Override
    public void setCountForDoc(String word, long count) {
        T element = extendedVocabulary.get(word);
        if (element != null) {
            element.setSequencesCount(count);
        }
    }

    /**
     * Returns total number of documents observed (if applicable)
     *
     * @return
     */
    @Override
    public long totalNumberOfDocs() {
        return documentsCounter.intValue();
    }

    /**
     * Increment total number of documents observed by 1
     */
    @Override
    public void incrementTotalDocCount() {
        documentsCounter.incrementAndGet();
    }

    /**
     * Increment total number of documents observed by specified value
     */
    @Override
    public void incrementTotalDocCount(long by) {
        documentsCounter.addAndGet(by);
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
     * @param element the word to add
     */
    @Override
    public void addToken(T element) {
        T oldElement = vocabulary.putIfAbsent(element.getStorageId(), element);
        if (oldElement == null) {
            //putIfAbsent added our element
            if (element.getLabel() != null) {
                extendedVocabulary.put(element.getLabel(), element);
            }
            oldElement = element;
        } else {
            oldElement.incrementSequencesCount(element.getSequencesCount());
            oldElement.increaseElementFrequency((int) element.getElementFrequency());
        }
        totalWordCount.addAndGet((long) oldElement.getElementFrequency());
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

    @Override
    public T tokenFor(long id) {
        return vocabulary.get(id);
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


    /**
     * This method imports all elements from VocabCache passed as argument
     * If element already exists,
     *
     * @param vocabCache
     */
    public void importVocabulary(@NonNull VocabCache<T> vocabCache) {
        for (T element : vocabCache.vocabWords()) {
            this.addToken(element);
        }
        //logger.info("Current state: {}; Adding value: {}", this.documentsCounter.get(), vocabCache.totalNumberOfDocs());
        this.documentsCounter.addAndGet(vocabCache.totalNumberOfDocs());
    }

    @Override
    public void updateWordsOccurrences() {
        totalWordCount.set(0);
        for (T element : vocabulary.values()) {
            long value = (long) element.getElementFrequency();

            if (value > 0) {
                totalWordCount.addAndGet(value);
            }
        }
        log.info("Updated counter: [" + totalWordCount.get() + "]");
    }

    @Override
    public void removeElement(String label) {
        SequenceElement element = extendedVocabulary.get(label);
        if (element != null) {
            totalWordCount.getAndAdd((long) element.getElementFrequency() * -1);
            idxMap.remove(element.getIndex());
            extendedVocabulary.remove(label);
            vocabulary.remove(element.getStorageId());
        } else
            throw new IllegalStateException("Can't get label: '" + label + "'");
    }

    @Override
    public void removeElement(T element) {
        removeElement(element.getLabel());
    }

    public static class Builder<T extends SequenceElement> {
        protected int scavengerThreshold = 3000000;
        protected int retentionDelay = 3;
        protected int minElementFrequency;
        protected boolean hugeModelExpected = false;


        public Builder<T> hugeModelExpected(boolean reallyExpected) {
            this.hugeModelExpected = reallyExpected;
            return this;
        }

        public Builder<T> scavengerThreshold(int threshold) {
            this.scavengerThreshold = threshold;
            return this;
        }

        public Builder<T> scavengerRetentionDelay(int delay) {
            this.retentionDelay = delay;
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
