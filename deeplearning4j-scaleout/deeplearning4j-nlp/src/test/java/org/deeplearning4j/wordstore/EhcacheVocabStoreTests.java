package org.deeplearning4j.wordstore;

import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.ehcache.EhCacheVocabCache;
import org.junit.Test;
import static org.junit.Assert.*;

import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

/**
 * Created by agibsonccc on 8/31/14.
 */
public class EhcacheVocabStoreTests {
    private static Logger log = LoggerFactory.getLogger(EhcacheVocabStoreTests.class);

    @Test
    public void testStorePut() {
        VocabCache cache = new EhCacheVocabCache();
        assertFalse(cache.containsWord("hello"));
        cache.addWordToIndex(0,"hello");
        assertTrue(cache.containsWord("hello"));
        assertEquals(1,cache.numWords());
        assertEquals("hello",cache.wordAtIndex(0));
    }



}
