package org.deeplearning4j.models.word2vec.wordstore;

import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.junit.Test;

import static org.junit.Assert.assertEquals;

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
        holder.updateHuffmanCodes();
        holder.transferBackToVocabCache(cache);

        // checking word frequency transfer
        assertEquals(3, cache.numWords());
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
        VocabularyHolder holder = new VocabularyHolder(cache, false);

        // no more UNK token here
        assertEquals(0, holder.numWords());
    }

    /**
     * In this test we make sure SPECIAL words are not affected by truncation in extending vocab
     * @throws Exception
     */
    @Test
    public void testSpecial1() throws Exception {
        VocabularyHolder holder = new VocabularyHolder.Builder().minWordFrequency(1).build();

        holder.addWord("test");
        holder.addWord("tests");

        holder.truncateVocabulary();

        assertEquals(2, holder.numWords());

        VocabCache cache = new InMemoryLookupCache();
        holder.transferBackToVocabCache(cache);

        VocabularyHolder holder2 = new VocabularyHolder.Builder().externalCache(cache).minWordFrequency(10)
                        //                .markAsSpecial(true)
                        .build();

        holder2.addWord("testz");
        assertEquals(3, holder2.numWords());

        holder2.truncateVocabulary();
        assertEquals(2, holder2.numWords());
    }

    @Test
    public void testScavenger1() throws Exception {
        VocabularyHolder holder = new VocabularyHolder.Builder().minWordFrequency(5).hugeModelExpected(true)
                        .scavengerActivationThreshold(1000000) // this value doesn't really matters, since we'll call for scavenger manually
                        .scavengerRetentionDelay(3).build();

        holder.addWord("test");
        holder.addWord("tests");

        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");

        holder.activateScavenger();
        assertEquals(2, holder.numWords());
        holder.activateScavenger();
        assertEquals(2, holder.numWords());

        // after third activation, word "test" should be removed
        holder.activateScavenger();
        assertEquals(1, holder.numWords());
    }

    @Test
    public void testScavenger2() throws Exception {
        VocabularyHolder holder = new VocabularyHolder.Builder().minWordFrequency(5).hugeModelExpected(true)
                        .scavengerActivationThreshold(1000000) // this value doesn't really matters, since we'll call for scavenger manually
                        .scavengerRetentionDelay(3).build();

        holder.addWord("test");
        holder.incrementWordCounter("test");

        holder.addWord("tests");

        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");

        holder.activateScavenger();
        assertEquals(2, holder.numWords());
        holder.activateScavenger();
        assertEquals(2, holder.numWords());

        // after third activation, word "test" should be removed
        holder.activateScavenger();
        assertEquals(1, holder.numWords());
    }

    @Test
    public void testScavenger3() throws Exception {
        VocabularyHolder holder = new VocabularyHolder.Builder().minWordFrequency(5).hugeModelExpected(true)
                        .scavengerActivationThreshold(1000000) // this value doesn't really matters, since we'll call for scavenger manually
                        .scavengerRetentionDelay(3).build();

        holder.addWord("test");

        holder.activateScavenger();
        assertEquals(1, holder.numWords());

        holder.incrementWordCounter("test");
        holder.addWord("tests");

        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");


        holder.activateScavenger();
        assertEquals(2, holder.numWords());

        // after third activation, word "test" should NOT be removed, since at point 0 we have freq == 1, and 2 in the following tests
        holder.activateScavenger();
        assertEquals(2, holder.numWords());

        // here we should have all retention points shifted, and word "test" should be removed
        holder.activateScavenger();
        assertEquals(1, holder.numWords());
    }

    @Test
    public void testScavenger4() throws Exception {
        VocabularyHolder holder = new VocabularyHolder.Builder().minWordFrequency(5).hugeModelExpected(true)
                        .scavengerActivationThreshold(1000000) // this value doesn't really matters, since we'll call for scavenger manually
                        .scavengerRetentionDelay(3).build();

        holder.addWord("test");

        holder.activateScavenger();
        assertEquals(1, holder.numWords());

        holder.incrementWordCounter("test");

        holder.addWord("tests");

        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");
        holder.incrementWordCounter("tests");


        holder.activateScavenger();
        assertEquals(2, holder.numWords());

        // after third activation, word "test" should NOT be removed, since at point 0 we have freq == 1, and 2 in the following tests
        holder.activateScavenger();
        assertEquals(2, holder.numWords());

        holder.incrementWordCounter("test");

        // here we should have all retention points shifted, and word "test" should NOT be removed, since now it's above the scavenger threshold
        holder.activateScavenger();
        assertEquals(2, holder.numWords());
    }
}
