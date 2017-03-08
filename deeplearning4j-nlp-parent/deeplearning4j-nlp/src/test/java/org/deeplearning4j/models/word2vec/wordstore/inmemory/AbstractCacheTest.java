package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import org.deeplearning4j.models.word2vec.Huffman;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;

import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by fartovii on 10.12.15.
 */
public class AbstractCacheTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testNumWords() throws Exception {
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        cache.addToken(new VocabWord(1.0, "word"));
        cache.addToken(new VocabWord(1.0, "test"));

        assertEquals(2, cache.numWords());
    }

    @Test
    public void testHuffman() throws Exception {
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        cache.addToken(new VocabWord(1.0, "word"));
        cache.addToken(new VocabWord(2.0, "test"));
        cache.addToken(new VocabWord(3.0, "tester"));

        assertEquals(3, cache.numWords());

        Huffman huffman = new Huffman(cache.tokens());
        huffman.build();
        huffman.applyIndexes(cache);

        assertEquals("tester", cache.wordAtIndex(0));
        assertEquals("test", cache.wordAtIndex(1));
        assertEquals("word", cache.wordAtIndex(2));

        VocabWord word = cache.tokenFor("tester");
        assertEquals(0, word.getIndex());
    }

    @Test
    public void testWordsOccurencies() throws Exception {
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        cache.addToken(new VocabWord(1.0, "word"));
        cache.addToken(new VocabWord(2.0, "test"));
        cache.addToken(new VocabWord(3.0, "tester"));

        assertEquals(3, cache.numWords());
        assertEquals(6, cache.totalWordOccurrences());
    }

    @Test
    public void testRemoval() throws Exception {
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        cache.addToken(new VocabWord(1.0, "word"));
        cache.addToken(new VocabWord(2.0, "test"));
        cache.addToken(new VocabWord(3.0, "tester"));

        assertEquals(3, cache.numWords());
        assertEquals(6, cache.totalWordOccurrences());

        cache.removeElement("tester");
        assertEquals(2, cache.numWords());
        assertEquals(3, cache.totalWordOccurrences());
    }

    @Test
    public void testLabels() throws Exception {
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        cache.addToken(new VocabWord(1.0, "word"));
        cache.addToken(new VocabWord(2.0, "test"));
        cache.addToken(new VocabWord(3.0, "tester"));

        Collection<String> collection = cache.words();
        assertEquals(3, collection.size());

        assertTrue(collection.contains("word"));
        assertTrue(collection.contains("test"));
        assertTrue(collection.contains("tester"));
    }
}
