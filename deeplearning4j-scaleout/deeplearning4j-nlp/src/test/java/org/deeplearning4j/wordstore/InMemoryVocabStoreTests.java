package org.deeplearning4j.wordstore;

import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 8/31/14.
 */
public class InMemoryVocabStoreTests {
    private static Logger log = LoggerFactory.getLogger(InMemoryVocabStoreTests.class);

    @Test
    public void testStorePut() {
        VocabCache cache = new InMemoryLookupCache(50);
        assertFalse(cache.containsWord("hello"));
        cache.addWordToIndex(0,"hello");
        assertTrue(cache.containsWord("hello"));
        assertEquals(1,cache.numWords());
        assertEquals("hello",cache.wordAtIndex(0));
    }



}
