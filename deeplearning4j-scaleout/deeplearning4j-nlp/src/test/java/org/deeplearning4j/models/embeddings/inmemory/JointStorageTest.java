package org.deeplearning4j.models.embeddings.inmemory;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.junit.Before;
import org.junit.Test;

import static org.junit.Assert.*;

/**
 * Created by fartovii on 05.12.15.
 */
public class JointStorageTest {

    InMemoryLookupCache sampleCache1;
    InMemoryLookupCache sampleCache2;
    InMemoryLookupTable sampleTable1;
    InMemoryLookupTable sampleTable2;

    @Before
    public void setUp() throws Exception {
        sampleCache1 = new InMemoryLookupCache(false);
        sampleCache2 = new InMemoryLookupCache(false);

        sampleTable1 = (InMemoryLookupTable) new InMemoryLookupTable.Builder()
                .cache(sampleCache1)
                .build();

        sampleTable2 = (InMemoryLookupTable) new InMemoryLookupTable.Builder()
                .cache(sampleCache2)
                .build();

        sampleCache1.vocabs.put("test1_0", new VocabWord(1.0,"test1_0"));

        sampleCache2.vocabs.put("test2_0", new VocabWord(2.0,"test2_0"));
        sampleCache2.vocabs.put("test2_1", new VocabWord(3.0,"test2_1"));
    }

    @Test
    public void testNumWords1() throws Exception {
        JointStorage jointTable = new JointStorage.Builder()
                .addLookupPair(sampleTable1)
                .addLookupPair(sampleTable2)
                .layerSize(100)
                .build();

        assertEquals(3, jointTable.numWords());
    }

    @Test
    public void testVocabsAsList() throws Exception {
        JointStorage jointTable = new JointStorage.Builder()
                .addLookupPair(sampleTable1)
                .addLookupPair(sampleTable2)
                .layerSize(100)
                .build();

        assertEquals(3, jointTable.vocabWords().size());
    }
}