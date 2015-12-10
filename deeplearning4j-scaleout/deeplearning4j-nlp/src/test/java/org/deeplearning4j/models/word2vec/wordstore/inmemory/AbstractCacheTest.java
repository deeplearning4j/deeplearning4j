package org.deeplearning4j.models.word2vec.wordstore.inmemory;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by fartovii on 10.12.15.
 */
public class AbstractCacheTest {

    @Before
    public void setUp() throws Exception {

    }

    @Test
    public void testNumWords() throws Exception {
        AbstractCache<VocabWord>  cache = new AbstractCache.Builder<VocabWord>()
                .build();

        cache.addToken(new VocabWord(1.0, "word"));
        cache.addToken(new VocabWord(1.0, "test"));

        assertEquals(2, cache.numWords());
    }
}