package org.deeplearning4j.models.word2vec.wordstore;

import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by fartovii on 08.11.15.
 */
public class VocabularyHolderTest {

    @Test
    public void testTransferBackToVocabCache() throws Exception {
        VocabularyHolder holder = new VocabularyHolder();
        holder.addWord("test");
        holder.addWord("tests");
        holder.addWord("testz");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("testz");

        InMemoryLookupCache cache = new InMemoryLookupCache(false);
        holder.transferBackToVocabCache(cache);

        // checking word frequency transfer
        assertEquals(3,cache.numWords());
        assertEquals(1, cache.wordFrequency("test"));
        assertEquals(2, cache.wordFrequency("testz"));
        assertEquals(3, cache.wordFrequency("tests"));


        // checking Huffman tree transfer
        assertEquals("tests", cache.wordAtIndex(0));
        assertEquals("testz", cache.wordAtIndex(1));
        assertEquals("test", cache.wordAtIndex(2));
    }

    @Test
    public void testConstructor() throws Exception {
        InMemoryLookupCache cache = new InMemoryLookupCache(true);
        VocabularyHolder holder = new VocabularyHolder(cache);

        assertEquals(1, holder.numWords());
    }
}