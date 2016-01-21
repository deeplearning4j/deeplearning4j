package org.deeplearning4j.ui.storage;

import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * This set of tests addresses real-world use of UiServer: some views have use of history, some (like t-SNE) are not.
 *
 * @author raver119@gmail.com
 */
public class HistoryStorageTest {
    HistoryStorage storage;
    @Before
    public void setUp() {
        storage = HistoryStorage.getInstance();

        // we explicitly wipe storage before each test
        storage.wipeStorage();
    }

    /**
     * This test covers singular element stored/accessed, that's used for TSNE
     * @throws Exception
     */
    @Test
    public void testGetSingleLatest() throws Exception {
        List<String> list = new ArrayList<>();
        list.add("some string 1");
        list.add("some string 2");
        list.add("some string 3");

        // store list as TSNE coords object
        storage.put("TSNE", Pair.makePair(1,0), list);

        // get it back and check equality
        List<String> list2 = (List<String>) storage.getLatest("TSNE");
        assertEquals(list, list2);
        assertEquals(list.size(), list2.size());
    }

    /**
     * This test covers multiple elements stored in unknown order, but accessed in ascending order
     * @throws Exception
     */
    @Test
    public void testGetSortedAscending() throws Exception {
        VocabWord word1 = new VocabWord(1.0, "word1");
        VocabWord word2 = new VocabWord(2.0, "word2");
        VocabWord word3 = new VocabWord(3.0, "word3");
        VocabWord word4 = new VocabWord(4.0, "word4");

        storage.put("ABSTRACT", Pair.makePair(1, 0), word1);
        storage.put("ABSTRACT", Pair.makePair(1, 2), word3);
        storage.put("ABSTRACT", Pair.makePair(1, 3), word4);
        storage.put("ABSTRACT", Pair.makePair(1, 1), word2);

        List<Object> list = storage.getSorted("ABSTRACT", HistoryStorage.SortOutput.ASCENDING);
        assertEquals(4, list.size());

        assertEquals(word1, (VocabWord) list.get(0));
        assertEquals(word2, (VocabWord) list.get(1));
        assertEquals(word3, (VocabWord) list.get(2));
        assertEquals(word4, (VocabWord) list.get(3));
    }

    /**
     * This test covers multiple elements stored in unknown order, but accessed in descending order
     * @throws Exception
     */
    @Test
    public void testGetSortedDescending() throws Exception {
        VocabWord word1 = new VocabWord(1.0, "word1");
        VocabWord word2 = new VocabWord(2.0, "word2");
        VocabWord word3 = new VocabWord(3.0, "word3");
        VocabWord word4 = new VocabWord(4.0, "word4");

        storage.put("ABSTRACT", Pair.makePair(1, 0), word1);
        storage.put("ABSTRACT", Pair.makePair(1, 2), word3);
        storage.put("ABSTRACT", Pair.makePair(1, 3), word4);
        storage.put("ABSTRACT", Pair.makePair(1, 1), word2);

        List<Object> list = storage.getSorted("ABSTRACT", HistoryStorage.SortOutput.DESCENDING);
        assertEquals(4, list.size());

        assertEquals(word4, (VocabWord) list.get(0));
        assertEquals(word3, (VocabWord) list.get(1));
        assertEquals(word2, (VocabWord) list.get(2));
        assertEquals(word1, (VocabWord) list.get(3));
    }

    /**
     * This test covers multiple elements stored in unknown order, but only latest retrieved
     * @throws Exception
     */
    @Test
    public void testGetLatest2() throws Exception {
        VocabWord word1 = new VocabWord(1.0, "word1");
        VocabWord word2 = new VocabWord(2.0, "word2");
        VocabWord word3 = new VocabWord(3.0, "word3");
        VocabWord word4 = new VocabWord(4.0, "word4");

        storage.put("ABSTRACT", Pair.makePair(1, 0), word1);
        storage.put("ABSTRACT", Pair.makePair(1, 2), word3);
        storage.put("ABSTRACT", Pair.makePair(1, 3), word4);
        storage.put("ABSTRACT", Pair.makePair(1, 1), word2);


        VocabWord xWord = (VocabWord) storage.getLatest("ABSTRACT");

        assertEquals(word4, xWord);
    }

    /**
     * This test covers multiple elements stored in unknown order, but only latest retrieved
     * @throws Exception
     */
    @Test
    public void testGetOldest() throws Exception {
        VocabWord word1 = new VocabWord(1.0, "word1");
        VocabWord word2 = new VocabWord(2.0, "word2");
        VocabWord word3 = new VocabWord(3.0, "word3");
        VocabWord word4 = new VocabWord(4.0, "word4");

        storage.put("ABSTRACT", Pair.makePair(1, 0), word1);
        storage.put("ABSTRACT", Pair.makePair(1, 2), word3);
        storage.put("ABSTRACT", Pair.makePair(1, 3), word4);
        storage.put("ABSTRACT", Pair.makePair(1, 1), word2);


        VocabWord xWord = (VocabWord) storage.getOldest("ABSTRACT");

        assertEquals(word1, xWord);
    }
}