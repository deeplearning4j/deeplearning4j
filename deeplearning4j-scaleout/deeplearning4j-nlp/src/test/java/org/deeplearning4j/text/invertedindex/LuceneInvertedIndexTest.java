package org.deeplearning4j.text.invertedindex;

import static org.junit.Assert.*;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCacheBuilder;
import org.junit.Test;

import java.util.Arrays;

/**
 * Created by agibsonccc on 10/21/14.
 */
public class LuceneInvertedIndexTest {

    @Test
    public void testLuceneInvertedIndex() {
        VocabCache cache = new InMemoryLookupCacheBuilder().vectorLength(100).createInMemoryLookupCache();
        cache.addToken(new VocabWord(1,"hello"));
        cache.addToken(new VocabWord(2,"hello2"));
        cache.addWordToIndex(0,"hello");
        cache.addWordToIndex(1,"hello2");
        cache.putVocabWord("hello");
        cache.putVocabWord("hello2");
        InvertedIndex index = new LuceneInvertedIndex(cache,true);
        index.addWordsToDoc(0, Arrays.asList(cache.wordFor("hello"),cache.wordFor("hello2")));
        index.addWordsToDoc(1, Arrays.asList(cache.wordFor("hello"),cache.wordFor("hello2")));
        index.finish();
        assertEquals(2,index.numDocuments());
        assertEquals(2,index.document(0).size());
        assertEquals(2,index.document(1).size());

        assertEquals(2,index.documents(cache.wordFor("hello")).size());
        assertEquals(2,index.documents(cache.wordFor("hello2")).size());


    }


}
