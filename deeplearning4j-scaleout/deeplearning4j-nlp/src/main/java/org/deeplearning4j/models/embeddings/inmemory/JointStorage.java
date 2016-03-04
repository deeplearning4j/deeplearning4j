package org.deeplearning4j.models.embeddings.inmemory;

import lombok.Getter;
import lombok.NonNull;
import lombok.Setter;
import org.apache.commons.lang.math.RandomUtils;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.ui.UiConnectionInfo;
import org.nd4j.linalg.api.ndarray.INDArray;

import java.io.File;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicLong;

/**
 * This is going to be primitive implementation of joint WeightLookupTable, used for ParagraphVectors and Word2Vec joint training.
 *
 * Main idea of this implementation nr.1: in some cases you have to train corpus for 2 vocabs instead of 1. Or you need to extend vocab,
 * and you can use few separate instead of rebuilding one big WeightLookupTable which can double used memory.
 *
 *
 *  WORK IS IN PROGRESS, PLEASE DO NOT USE
 *
 * @author raver119@gmail.com
 */
@Deprecated
public class JointStorage<T extends SequenceElement> implements WeightLookupTable<T>, VocabCache<T> {
    private Map<Long, WeightLookupTable<T>> mapTables = new ConcurrentHashMap<>();
    private Map<Long, VocabCache<T>> mapVocabs = new ConcurrentHashMap<>();
    private int layerSize;

    @Getter @Setter protected Long tableId;

    @Override
    public void loadVocab() {

    }

    @Override
    public boolean vocabExists() {
        return false;
    }

    @Override
    public void saveVocab() {

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
    public T elementAtIndex(int index) {
        return null;
    }

    @Override
    public int indexOf(String word) {
        return 0;
    }

    /**
     * Returns all SequenceElements in this JointStorage instance.
     * Please note, if used in distributed environment this can cause OOM exceptions. Use with caution, or use iterator instead
     *
     * @return Collection of all VocabWords in this JointStorage
     */
    @Override
    public Collection<T> vocabWords() {
        ArrayList<T> words = new ArrayList<>();
        for (VocabCache cache: mapVocabs.values()) {
            words.addAll(cache.vocabWords());
        }
        return Collections.unmodifiableCollection(words);
    }

    @Override
    public long totalWordOccurrences() {
        return 0;
    }

    @Override
    public T wordFor(String word) {
        return null;
    }

    @Override
    public void addWordToIndex(int index, String word) {

    }


    @Override
    public void putVocabWord(String word) {

    }

    /**
     * Returns number of words in all underlying vocabularies
     *
     * @return
     */
    @Override
    public int numWords() {
        // TODO: this should return Long in future, since joint storage mechanics should allow really huge vocabularies and maps

        AtomicLong counter = new AtomicLong(0);
        for (VocabCache cache: mapVocabs.values()) {
            counter.addAndGet(cache.numWords());
        }
        return counter.intValue();
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
    public void addToken(SequenceElement word) {

    }

    @Override
    public T tokenFor(String word) {
        return null;
    }

    @Override
    public boolean hasToken(String token) {
        return false;
    }

    @Override
    public void importVocabulary(VocabCache<T> vocabCache) {

    }

    @Override
    public void updateWordsOccurencies() {

    }

    @Override
    public void removeElement(String label) {

    }

    @Override
    public void removeElement(T element) {

    }

    @Override
    public int layerSize() {
        return this.layerSize;
    }

    /**
     * Returns gradient for specified word
     *
     * @param column
     * @param gradient
     * @return
     */
    @Override
    public double getGradient(int column, double gradient) {
        return 0;
    }

    @Override
    public void resetWeights(boolean reset) {

    }

    /**
     * Render the words via TSNE
     *
     * @param tsne           the tsne to use
     * @param numWords
     * @param connectionInfo
     */
    @Override
    public void plotVocab(Tsne tsne, int numWords, UiConnectionInfo connectionInfo) {

    }

    /**
     * Render the words via TSNE
     *
     * @param tsne     the tsne to use
     * @param numWords
     * @param file
     */
    @Override
    public void plotVocab(Tsne tsne, int numWords, File file) {

    }

    /**
     * Render the words via tsne
     *
     * @param numWords
     * @param connectionInfo
     */
    @Override
    public void plotVocab(int numWords, UiConnectionInfo connectionInfo) {

    }

    /**
     * Render the words via tsne
     *
     * @param numWords
     * @param file
     */
    @Override
    public void plotVocab(int numWords, File file) {

    }


    @Override
    public void putCode(int codeIndex, INDArray code) {

    }

    @Override
    public INDArray loadCodes(int[] codes) {
        return null;
    }

    @Override
    public void iterate(T w1, T w2) {

    }

    @Override
    public void iterateSample(T w1, T w2, AtomicLong nextRandom, double alpha) {

    }

    @Override
    public void putVector(String word, INDArray vector) {

    }

    @Override
    public INDArray vector(String word) {
        return null;
    }

    @Override
    public void resetWeights() {

    }

    @Override
    public void setLearningRate(double lr) {

    }

    @Override
    public Iterator<INDArray> vectors() {
        return null;
    }

    @Override
    public INDArray getWeights() {
        return null;
    }

    @Override
    public VocabCache<T> getVocabCache() {
        return this;
    }

    public static class Builder<T extends SequenceElement> {
        private Map<Long, WeightLookupTable<T>> mapTables = new ConcurrentHashMap<>();
        private Map<Long, VocabCache<T>> mapVocabs = new ConcurrentHashMap<>();
        private int layerSize;

        public Builder() {

        }

        /**
         * Adds InMemoryLookupTable into JointStorage, VocabCache will be fetched from table
         *
         * @param lookupTable InMemoryLookupTable that's going to be part of Joint Lookup Table
         * @return
         */
        public Builder addLookupPair(@NonNull InMemoryLookupTable<T> lookupTable) {
            return addLookupPair(lookupTable, lookupTable.getVocab());
        }

        /**
         * Adds WeightLookupTable into JointStorage
         *
         * @param lookupTable WeightLookupTable that's going to be part of Joint Lookup Table
         * @param cache VocabCache that contains vocabulary for lookupTable
         * @return
         */
        public Builder addLookupPair(@NonNull WeightLookupTable<T> lookupTable, @NonNull VocabCache<T> cache) {
            /*
                we should assume, that each word in VocabCache is tagged with pair Vocab/Table ID
            */
            if (lookupTable.getTableId() == null || lookupTable.getTableId().longValue() == 0)
                lookupTable.setTableId(RandomUtils.nextLong());

            for (T word: cache.vocabWords()) {
                // each word should be tagged here
                word.setStorageId(lookupTable.getTableId());
            }

            mapTables.put(lookupTable.getTableId(), lookupTable);
            mapVocabs.put(lookupTable.getTableId(), cache);
            return this;
        }

        public Builder layerSize(int layerSize) {
            this.layerSize = layerSize;
            return this;
        }

        public JointStorage build() {
            JointStorage<T> lookupTable = new JointStorage();
            lookupTable.mapTables = this.mapTables;
            lookupTable.mapVocabs = this.mapVocabs;
            lookupTable.layerSize = this.layerSize;

            return lookupTable;
        }
    }
}
