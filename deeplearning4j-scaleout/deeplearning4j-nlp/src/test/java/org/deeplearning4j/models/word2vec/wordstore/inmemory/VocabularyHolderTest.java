package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by fartovii on 08.11.15.
 */
public class VocabularyHolderTest {

    @Before
    public void setUp() throws Exception {

    }

    /**
     * Testing vocab transfer from VocabCache to VocabularyHolder
     * @throws Exception
     */
    @Test
    public void testNumWords1() throws Exception {
        InMemoryLookupCache cache = new InMemoryLookupCache(true);
        VocabularyHolder holder = new VocabularyHolder(cache);

        assertEquals(1, holder.numWords());
    }

    /**
     * Testing vocab transfer from VocabCache to VocabularyHolder
     * @throws Exception
     */
    @Test
    public void testNumWords2() throws Exception {
        InMemoryLookupCache cache = new InMemoryLookupCache(true);
        cache.addToken(new VocabWord(1.0d, "word"));
        cache.addToken(new VocabWord(2.0d, "wordzzz"));
        VocabularyHolder holder = new VocabularyHolder(cache);

        assertEquals(3, holder.numWords());
    }

    /**
     * Testing Huffman tree indexes and vocab transfer from VocabularyHolder to VocabCache
     *
     * @throws Exception
     */
    @Test
    public void testVocabTransfer() throws Exception {
        InMemoryLookupCache cache = new InMemoryLookupCache(false);
        VocabularyHolder holder = new VocabularyHolder();
        holder.addWord("word");
        holder.addWord("words");
        holder.addWord("wordz");
        holder.incrementWordCounter("word");
        holder.incrementWordCounter("words");
        holder.incrementWordCounter("words");
        holder.incrementWordCounter("wordz");
        holder.incrementWordCounter("wordz");
        holder.incrementWordCounter("wordz");

        holder.transferBackToVocabCache(cache);

        assertEquals("wordz", cache.wordAtIndex(0));
        assertEquals("words", cache.wordAtIndex(1));
        assertEquals("word", cache.wordAtIndex(2));
    }
}