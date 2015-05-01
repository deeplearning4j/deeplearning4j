/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.text.invertedindex;

import static org.junit.Assert.*;

import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.junit.Before;
import org.junit.Test;

import java.io.File;
import java.util.Arrays;

/**
 * Created by agibsonccc on 10/21/14.
 */
public class LuceneInvertedIndexTest {

    @Before
    public void before() {
        new File(LuceneInvertedIndex.DEFAULT_INDEX_DIR).delete();
    }


    @Test
    public void testLuceneInvertedIndex() {
        VocabCache cache = new InMemoryLookupCache();
        cache.addToken(new VocabWord(1,"hello"));
        cache.addToken(new VocabWord(2,"hello2"));
        cache.addWordToIndex(0,"hello");
        cache.addWordToIndex(1,"hello2");
        cache.putVocabWord("hello");
        cache.putVocabWord("hello2");
        InvertedIndex index = new LuceneInvertedIndex(cache,false);
        index.cleanup();
        index.addWordsToDoc(0, Arrays.asList(cache.wordFor("hello"),cache.wordFor("hello2")));
        index.addWordsToDoc(1, Arrays.asList(cache.wordFor("hello"),cache.wordFor("hello2")));
        index.finish();
        assertEquals(2,index.numDocuments());
        assertEquals(2,index.document(0).size());
        assertEquals(2,index.document(1).size());

        assertEquals(2,index.documents(cache.wordFor("hello")).length);
        assertEquals(2,index.documents(cache.wordFor("hello2")).length);


    }


}
