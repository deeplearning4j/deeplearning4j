package org.deeplearning4j.models.word2vec.wordstore;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.*;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;

/**
 *This class is used as simplifed VocabCache for vocabulary building routines.
 * As soon as vocab is built, all words will be transferred into VocabCache.
 *
 * @author raver119@gmail.com
 */
public class VocabularyHolder  {
    private Map<String, VocabularyWord> vocabulary = new ConcurrentHashMap<>();
    private Map<Integer, VocabularyWord> idxMap = new ConcurrentHashMap<>();

    private  long totalWordOccurencies = 0;

    // TODO: this list is probably NOT needed at all, and can be easily replaced by vocabulary.values(), with sort enabled
//    private List<VocabularyWord> vocab = new ArrayList<>();

    private AtomicInteger totalWordCount = new AtomicInteger(0);

    private Logger logger = LoggerFactory.getLogger(VocabularyHolder.class);

    private static final int MAX_CODE_LENGTH = 40;

    /**
     * Default constructor
     */
    public VocabularyHolder() {

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
    public VocabularyHolder(VocabCache cache) {
        for (VocabWord word: cache.tokens()) {
            VocabularyWord vw = new VocabularyWord(word.getWord());
            vw.setCount((int) word.getWordFrequency());

            // TODO: i'm not sure, if it's really worth to transfer Huffman tree data. Maybe it's easier to recalculate everything.
            if (word.getCodeLength() != 0 && !word.getCodes().isEmpty()) {
                // do nothing. see comment above ^^^
            }
            vocabulary.putIfAbsent(vw.getWord(), vw);
        }

        // there's no sense building huffman tree just for UNK word
        if (numWords() > 1) updateHuffmanCodes();
        logger.debug("Init from VocabCache is complete. " + numWords() + " word(s) were transferred.");
    }

    /**
     * This method is required for compatibility purposes.
     *  It just transfers vocabulary from VocabHolder into VocabCache
     *
     * @param cache
     */
    public void transferBackToVocabCache(VocabCache cache) {
        if (!(cache instanceof InMemoryLookupCache)) throw new IllegalStateException("Sorry, only InMemoryLookupCache use implemented.");

        // make sure that huffman codes are updated before transfer
        List<VocabularyWord> words = updateHuffmanCodes();

        for (VocabularyWord word: words) {
            if (word.getWord().isEmpty()) continue;
            VocabWord vocabWord = new VocabWord(word.getCount(), word.getWord());

            // update Huffman tree information
            vocabWord.setIndex(word.getHuffmanNode().getIdx());
            vocabWord.setCodeLength(word.getHuffmanNode().getLength());
            vocabWord.setPoints(arrayToList(word.getHuffmanNode().getPoint(), word.getHuffmanNode().getLength()));
            vocabWord.setCodes(arrayToList(word.getHuffmanNode().getCode(), word.getHuffmanNode().getLength()));

            // put VocabWord into both Tokens and Vocabs maps
            ((InMemoryLookupCache) cache).getVocabs().putIfAbsent(word.getWord(), vocabWord);
            ((InMemoryLookupCache) cache).getTokens().putIfAbsent(word.getWord(), vocabWord);

            // put word into index
            cache.addWordToIndex(word.getHuffmanNode().getIdx(), word.getWord());

            //update vocabWord counter. substract 1, since its the base value for any token
            cache.incrementWordCount(word.getWord(), word.getCount() - 1);
        }

        // at this moment its pretty safe to nullify all vocabs.
        idxMap.clear();
        vocabulary.clear();
    }

    private List<Integer> arrayToList(byte[] array, int codeLen) {
        List<Integer> result = new ArrayList<>();
        for (int x = 0; x < codeLen; x++) {
            result.add((int) array[x]);
        }
        return result;
    }

    private List<Integer> arrayToList(int[] array, int codeLen) {
        List<Integer> result = new ArrayList<>();
        for (int x = 0; x < codeLen; x++) {
            result.add(array[x]);
        }
        return result;
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
        } else throw new IllegalStateException("No such word found");
    }

    /**
     * Adds new word to vocabulary
     *
     * @param word to be added
     */
    // TODO: investigate, if it's worth to make this synchronized and virtually thread-safe
    public void addWord(String word) {
        if (!vocabulary.containsKey(word)) {
            VocabularyWord vw = new VocabularyWord(word);
            vocabulary.putIfAbsent(word, vw);
//            vocab.add(vw);
//            idxMap.putIfAbsent(vw.getId(), vw);
            return;
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
     * All words with frequency below threshold wii be removed
     *
     * @param threshold exclusive threshold for removal
     */
    public void truncateVocabulary(int threshold) {
        Set<String> keyset = vocabulary.keySet();
        for (String word: keyset) {
            if (vocabulary.get(word).getCount() < threshold) {
                VocabularyWord vw = vocabulary.get(word);

                vocabulary.remove(word);
//                vocab.remove(vw);
                if (vw.getHuffmanNode() != null) idxMap.remove(vw.getHuffmanNode().getIdx());
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
        for (int a = 0; a < vocab.size(); a++) count[a] = vocab.get(a).getCount();
        for (int a = vocab.size(); a < vocab.size() * 2; a++) count[a] = Integer.MAX_VALUE;
        int pos1 = vocab.size() - 1;
        int pos2 = vocab.size();
        for (int a = 0; a< vocab.size(); a++) {
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
                if (b == vocab.size() * 2 - 2) break;
            }

            lpoint[0] = vocab.size() - 2;
            for (b = 0; b < i; b++) {
                lcode[i - b - 1] = code[b];
                lpoint[i - b] = point[b] - vocab.size();
            }

            vocab.get(a).setHuffmanNode(new HuffmanNode(lcode, lpoint,a,  (byte) i));
        }

        idxMap.clear();
        int a = 0;
        for (VocabularyWord word: vocab) {
            idxMap.putIfAbsent(word.getHuffmanNode().getIdx(), word);
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
        } else return -1;
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
            for (VocabularyWord word: vocabulary.values()) {
                totalWordOccurencies += word.getCount();
            }
            return totalWordOccurencies;
        } else return totalWordOccurencies;
    }
}
