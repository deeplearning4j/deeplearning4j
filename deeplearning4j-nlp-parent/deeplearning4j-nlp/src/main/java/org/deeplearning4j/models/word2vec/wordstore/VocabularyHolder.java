package org.deeplearning4j.models.word2vec.wordstore;

import lombok.NonNull;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.Serializable;
import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 *This class is used as simplifed VocabCache for vocabulary building routines.
 * As soon as vocab is built, all words will be transferred into VocabCache.
 *
 * @author raver119@gmail.com
 */
public class VocabularyHolder implements Serializable {
    private Map<String, VocabularyWord> vocabulary = new ConcurrentHashMap<>();

    // idxMap marked as transient, since there's no real reason to save this data on serialization
    private transient Map<Integer, VocabularyWord> idxMap = new ConcurrentHashMap<>();
    private int minWordFrequency = 0;
    private boolean hugeModelExpected = false;
    private int retentionDelay = 3;

    private VocabCache vocabCache;

    // this variable defines how often scavenger will be activated
    private int scavengerThreshold = 2000000;

    private long totalWordOccurencies = 0;

    // for scavenger mechanics we need to know the actual number of words being added
    private transient AtomicLong hiddenWordsCounter = new AtomicLong(0);

    private AtomicInteger totalWordCount = new AtomicInteger(0);

    private Logger logger = LoggerFactory.getLogger(VocabularyHolder.class);

    private static final int MAX_CODE_LENGTH = 40;

    /**
     * Default constructor
     */
    protected VocabularyHolder() {

    }

    /**
     * Builds VocabularyHolder from VocabCache.
     *
     * Basically we just ignore tokens, and transfer VocabularyWords, supposing that it's already truncated by minWordFrequency.
     *
     * Huffman tree data is ignored and recalculated, due to suspectable flaw in dl4j huffman impl, and it's exsessive memory usage.
     *
     * This code is required for compatibility between dl4j w2v implementation, and standalone w2v
     * @param cache
     */
    protected VocabularyHolder(@NonNull VocabCache<? extends SequenceElement> cache, boolean markAsSpecial) {
        this.vocabCache = cache;
        for (SequenceElement word : cache.tokens()) {
            VocabularyWord vw = new VocabularyWord(word.getLabel());
            vw.setCount((int) word.getElementFrequency());

            // since we're importing this word from external VocabCache, we'll assume that this word is SPECIAL, and should NOT be affected by minWordFrequency
            vw.setSpecial(markAsSpecial);

            // please note: we don't transfer huffman data, since proper way is  to recalculate it after new words being added
            if (word.getPoints() != null && !word.getPoints().isEmpty()) {
                vw.setHuffmanNode(buildNode(word.getCodes(), word.getPoints(), word.getCodeLength(), word.getIndex()));
            }

            vocabulary.put(vw.getWord(), vw);
        }

        // there's no sense building huffman tree just for UNK word
        if (numWords() > 1)
            updateHuffmanCodes();
        logger.info("Init from VocabCache is complete. " + numWords() + " word(s) were transferred.");
    }

    public static HuffmanNode buildNode(List<Byte> codes, List<Integer> points, int codeLen, int index) {
        return new HuffmanNode(listToArray(codes), listToArray(points, MAX_CODE_LENGTH), index, (byte) codeLen);
    }


    public void transferBackToVocabCache() {
        transferBackToVocabCache(this.vocabCache, true);
    }

    public void transferBackToVocabCache(VocabCache cache) {
        transferBackToVocabCache(cache, true);
    }

    /**
     * This method is required for compatibility purposes.
     *  It just transfers vocabulary from VocabHolder into VocabCache
     *
     * @param cache
     */
    public void transferBackToVocabCache(VocabCache cache, boolean emptyHolder) {
        if (!(cache instanceof InMemoryLookupCache))
            throw new IllegalStateException("Sorry, only InMemoryLookupCache use implemented.");

        // make sure that huffman codes are updated before transfer
        List<VocabularyWord> words = words(); //updateHuffmanCodes();

        for (VocabularyWord word : words) {
            if (word.getWord().isEmpty())
                continue;
            VocabWord vocabWord = new VocabWord(1, word.getWord());

            // if we're transferring full model, it CAN contain HistoricalGradient for AdaptiveGradient feature
            if (word.getHistoricalGradient() != null) {
                INDArray gradient = Nd4j.create(word.getHistoricalGradient());
                vocabWord.setHistoricalGradient(gradient);
            }

            // put VocabWord into both Tokens and Vocabs maps
            ((InMemoryLookupCache) cache).getVocabs().put(word.getWord(), vocabWord);
            ((InMemoryLookupCache) cache).getTokens().put(word.getWord(), vocabWord);


            // update Huffman tree information
            if (word.getHuffmanNode() != null) {
                vocabWord.setIndex(word.getHuffmanNode().getIdx());
                vocabWord.setCodeLength(word.getHuffmanNode().getLength());
                vocabWord.setPoints(arrayToList(word.getHuffmanNode().getPoint(), word.getHuffmanNode().getLength()));
                vocabWord.setCodes(arrayToList(word.getHuffmanNode().getCode(), word.getHuffmanNode().getLength()));

                // put word into index
                cache.addWordToIndex(word.getHuffmanNode().getIdx(), word.getWord());
            }

            //update vocabWord counter. substract 1, since its the base value for any token
            // >1 hack is required since VocabCache impl imples 1 as base word count, not 0
            if (word.getCount() > 1)
                cache.incrementWordCount(word.getWord(), word.getCount() - 1);
        }

        // at this moment its pretty safe to nullify all vocabs.
        if (emptyHolder) {
            idxMap.clear();
            vocabulary.clear();
        }
    }

    /**
     * This method is needed ONLY for unit tests and should NOT be available in public scope.
     *
     * It sets the vocab size ratio, at wich dynamic scavenger will be activated
     * @param threshold
     */
    protected void setScavengerActivationThreshold(int threshold) {
        this.scavengerThreshold = threshold;
    }


    /**
     *  This method is used only for VocabCache compatibility purposes
     * @param array
     * @param codeLen
     * @return
     */
    public static List<Byte> arrayToList(byte[] array, int codeLen) {
        List<Byte> result = new ArrayList<>();
        for (int x = 0; x < codeLen; x++) {
            result.add(array[x]);
        }
        return result;
    }

    public static byte[] listToArray(List<Byte> code) {
        byte[] array = new byte[MAX_CODE_LENGTH];
        for (int x = 0; x < code.size(); x++) {
            array[x] = code.get(x);
        }
        return array;
    }

    public static int[] listToArray(List<Integer> points, int codeLen) {
        int[] array = new int[points.size()];
        for (int x = 0; x < points.size(); x++) {
            array[x] = points.get(x).intValue();
        }
        return array;
    }

    /**
     *  This method is used only for VocabCache compatibility purposes
     * @param array
     * @param codeLen
     * @return
     */
    public static List<Integer> arrayToList(int[] array, int codeLen) {
        List<Integer> result = new ArrayList<>();
        for (int x = 0; x < codeLen; x++) {
            result.add(array[x]);
        }
        return result;
    }

    public Collection<VocabularyWord> getVocabulary() {
        return vocabulary.values();
    }

    public VocabularyWord getVocabularyWordByString(String word) {
        return vocabulary.get(word);
    }

    public VocabularyWord getVocabularyWordByIdx(Integer id) {
        return idxMap.get(id);
    }

    /**
     * Checks vocabulary for the word existance
     *
     * @param word to be looked for
     * @return TRUE of contains, FALSE otherwise
     */
    public boolean containsWord(String word) {
        return vocabulary.containsKey(word);
    }

    /**
     * Increments by one number of occurencies of the word in corpus
     *
     * @param word whose counter is to be incremented
     */
    public void incrementWordCounter(String word) {
        if (vocabulary.containsKey(word)) {
            vocabulary.get(word).incrementCount();
        }
        // there's no need to throw such exception here. just do nothing if word is not found
        //else throw new IllegalStateException("No such word found");
    }

    /**
     * Adds new word to vocabulary
     *
     * @param word to be added
     */
    // TODO: investigate, if it's worth to make this internally synchronized and virtually thread-safe
    public void addWord(String word) {
        if (!vocabulary.containsKey(word)) {
            VocabularyWord vw = new VocabularyWord(word);

            /*
                TODO: this should be done in different way, since this implementation causes minWordFrequency ultimate ignoral if markAsSpecial set to TRUE
            
                Probably the best way to solve it, is remove markAsSpecial option here, and let this issue be regulated with minWordFrequency
              */
            // vw.setSpecial(markAsSpecial);

            // initialize frequencyShift only if hugeModelExpected. It's useless otherwise :)
            if (hugeModelExpected)
                vw.setFrequencyShift(new byte[retentionDelay]);

            vocabulary.put(word, vw);



            if (hugeModelExpected && minWordFrequency > 1
                            && hiddenWordsCounter.incrementAndGet() % scavengerThreshold == 0)
                activateScavenger();

            return;
        }
    }

    public void addWord(VocabularyWord word) {
        vocabulary.put(word.getWord(), word);
    }

    public void consumeVocabulary(VocabularyHolder holder) {
        for (VocabularyWord word : holder.getVocabulary()) {
            if (!this.containsWord(word.getWord())) {
                this.addWord(word);
            } else {
                holder.incrementWordCounter(word.getWord());
            }
        }
    }

    /**
     * This method removes low-frequency words based on their frequency change between activations.
     * I.e. if word has appeared only once, and it's retained the same frequency over consequence activations, we can assume it can be removed freely
     */
    protected synchronized void activateScavenger() {
        int initialSize = vocabulary.size();
        List<VocabularyWord> words = new ArrayList<>(vocabulary.values());
        for (VocabularyWord word : words) {
            // scavenging could be applied only to non-special tokens that are below minWordFrequency
            if (word.isSpecial() || word.getCount() >= minWordFrequency || word.getFrequencyShift() == null) {
                word.setFrequencyShift(null);
                continue;
            }

            // save current word counter to byte array at specified position
            word.getFrequencyShift()[word.getRetentionStep()] = (byte) word.getCount();

            /*
                    we suppose that we're hunting only low-freq words that already passed few activations
                    so, we assume word personal threshold as 20% of minWordFrequency, but not less then 1.
            
                    so, if after few scavenging cycles wordCount is still <= activation - just remove word.
                    otherwise nullify word.frequencyShift to avoid further checks
              */
            int activation = Math.max(minWordFrequency / 5, 2);
            logger.debug("Current state> Activation: [" + activation + "], retention info: "
                            + Arrays.toString(word.getFrequencyShift()));
            if (word.getCount() <= activation && word.getFrequencyShift()[this.retentionDelay - 1] > 0) {

                // if final word count at latest retention point is the same as at the beginning - just remove word
                if (word.getFrequencyShift()[this.retentionDelay - 1] <= activation
                                && word.getFrequencyShift()[this.retentionDelay - 1] == word.getFrequencyShift()[0]) {
                    vocabulary.remove(word.getWord());
                }
            }

            // shift retention history to the left
            if (word.getRetentionStep() < retentionDelay - 1) {
                word.incrementRetentionStep();
            } else {
                for (int x = 1; x < retentionDelay; x++) {
                    word.getFrequencyShift()[x - 1] = word.getFrequencyShift()[x];
                }
            }
        }
        logger.info("Scavenger was activated. Vocab size before: [" + initialSize + "],  after: [" + vocabulary.size()
                        + "]");
    }

    /**
     * This methods reset counters for all words in vocabulary
     */
    public void resetWordCounters() {
        for (VocabularyWord word : getVocabulary()) {
            word.setHuffmanNode(null);
            word.setFrequencyShift(null);
            word.setCount(0);
        }
    }

    /**
     *
     * @return number of words in vocabulary
     */
    public int numWords() {
        return vocabulary.size();
    }

    /**
     * The same as truncateVocabulary(this.minWordFrequency)
     */
    public void truncateVocabulary() {
        truncateVocabulary(minWordFrequency);
    }

    /**
     * All words with frequency below threshold wii be removed
     *
     * @param threshold exclusive threshold for removal
     */
    public void truncateVocabulary(int threshold) {
        logger.debug("Truncating vocabulary to minWordFrequency: [" + threshold + "]");
        Set<String> keyset = vocabulary.keySet();
        for (String word : keyset) {
            VocabularyWord vw = vocabulary.get(word);

            // please note: we're not applying threshold to SPECIAL words
            if (!vw.isSpecial() && vw.getCount() < threshold) {
                vocabulary.remove(word);
                if (vw.getHuffmanNode() != null)
                    idxMap.remove(vw.getHuffmanNode().getIdx());
            }
        }
    }

    /**
     * build binary tree ordered by counter.
     *
     * Based on original w2v by google
     */
    public List<VocabularyWord> updateHuffmanCodes() {
        int min1i;
        int min2i;
        int b;
        int i;
        // get vocabulary as sorted list
        List<VocabularyWord> vocab = this.words();
        int count[] = new int[vocab.size() * 2 + 1];
        int parent_node[] = new int[vocab.size() * 2 + 1];
        byte binary[] = new byte[vocab.size() * 2 + 1];

        // at this point vocab is sorted, with descending order
        for (int a = 0; a < vocab.size(); a++)
            count[a] = vocab.get(a).getCount();
        for (int a = vocab.size(); a < vocab.size() * 2; a++)
            count[a] = Integer.MAX_VALUE;
        int pos1 = vocab.size() - 1;
        int pos2 = vocab.size();
        for (int a = 0; a < vocab.size(); a++) {
            // First, find two smallest nodes 'min1, min2'
            if (pos1 >= 0) {
                if (count[pos1] < count[pos2]) {
                    min1i = pos1;
                    pos1--;
                } else {
                    min1i = pos2;
                    pos2++;
                }
            } else {
                min1i = pos2;
                pos2++;
            }
            if (pos1 >= 0) {
                if (count[pos1] < count[pos2]) {
                    min2i = pos1;
                    pos1--;
                } else {
                    min2i = pos2;
                    pos2++;
                }
            } else {
                min2i = pos2;
                pos2++;
            }
            count[vocab.size() + a] = count[min1i] + count[min2i];
            parent_node[min1i] = vocab.size() + a;
            parent_node[min2i] = vocab.size() + a;
            binary[min2i] = 1;
        }

        // Now assign binary code to each vocabulary word
        byte[] code = new byte[MAX_CODE_LENGTH];
        int[] point = new int[MAX_CODE_LENGTH];

        for (int a = 0; a < vocab.size(); a++) {
            b = a;
            i = 0;
            byte[] lcode = new byte[MAX_CODE_LENGTH];
            int[] lpoint = new int[MAX_CODE_LENGTH];
            while (true) {
                code[i] = binary[b];
                point[i] = b;
                i++;
                b = parent_node[b];
                if (b == vocab.size() * 2 - 2)
                    break;
            }

            lpoint[0] = vocab.size() - 2;
            for (b = 0; b < i; b++) {
                lcode[i - b - 1] = code[b];
                lpoint[i - b] = point[b] - vocab.size();
            }

            vocab.get(a).setHuffmanNode(new HuffmanNode(lcode, lpoint, a, (byte) i));
        }

        idxMap.clear();
        for (VocabularyWord word : vocab) {
            idxMap.put(word.getHuffmanNode().getIdx(), word);
        }

        return vocab;
    }

    /**
     * This method returns index of word in sorted list.
     *
     * @param word
     * @return
     */
    public int indexOf(String word) {
        if (vocabulary.containsKey(word)) {
            return vocabulary.get(word).getHuffmanNode().getIdx();
        } else
            return -1;
    }


    /**
     * Returns sorted list of words in vocabulary.
     * Sort is DESCENDING.
     *
     * @return list of VocabularyWord
     */
    public List<VocabularyWord> words() {
        List<VocabularyWord> vocab = new ArrayList<>(vocabulary.values());
        Collections.sort(vocab, new Comparator<VocabularyWord>() {
            @Override
            public int compare(VocabularyWord o1, VocabularyWord o2) {
                return Integer.compare(o2.getCount(), o1.getCount());
            }
        });

        return vocab;
    }

    public long totalWordsBeyondLimit() {
        if (totalWordOccurencies == 0) {
            for (VocabularyWord word : vocabulary.values()) {
                totalWordOccurencies += word.getCount();
            }
            return totalWordOccurencies;
        } else
            return totalWordOccurencies;
    }

    public static class Builder {
        private VocabCache cache = null;
        private int minWordFrequency = 0;
        private boolean hugeModelExpected = false;
        private int scavengerThreshold = 2000000;
        private int retentionDelay = 3;

        public Builder() {

        }

        public Builder externalCache(@NonNull VocabCache cache) {
            this.cache = cache;
            return this;
        }

        public Builder minWordFrequency(int threshold) {
            this.minWordFrequency = threshold;
            return this;
        }

        /**
         * With this argument set to true, you'll have your vocab scanned for low-freq words periodically.
         *
         * Please note: this is incompatible with SPECIAL mechanics.
         *
         * @param reallyExpected
         * @return
         */
        public Builder hugeModelExpected(boolean reallyExpected) {
            this.hugeModelExpected = reallyExpected;
            return this;
        }

        /**
         *  Activation threshold defines, how ofter scavenger will be executed, to throw away low-frequency keywords.
         *  Good values to start mostly depends on your workstation. Something like 1000000 looks pretty nice to start with.
         *  Too low values can lead to undesired removal of words from vocab.
         *
         *  Please note: this is incompatible with SPECIAL mechanics.
         *
         * @param threshold
         * @return
         */
        public Builder scavengerActivationThreshold(int threshold) {
            this.scavengerThreshold = threshold;
            return this;
        }

        /**
         * Retention delay defines, how long low-freq word will be kept in vocab, during building.
         * Good values to start with: 3,4,5. Not too high, and not too low.
         *
         * Please note: this is incompatible with SPECIAL mechanics.
         *
         * @param delay
         * @return
         */
        public Builder scavengerRetentionDelay(int delay) {
            if (delay < 2)
                throw new IllegalStateException("Delay < 2 doesn't really makes sense");
            this.retentionDelay = delay;
            return this;
        }

        public VocabularyHolder build() {
            VocabularyHolder holder = null;
            if (cache != null) {
                holder = new VocabularyHolder(cache, true);
            } else {
                holder = new VocabularyHolder();
            }
            holder.minWordFrequency = this.minWordFrequency;
            holder.hugeModelExpected = this.hugeModelExpected;
            holder.scavengerThreshold = this.scavengerThreshold;
            holder.retentionDelay = this.retentionDelay;

            return holder;
        }
    }
}
