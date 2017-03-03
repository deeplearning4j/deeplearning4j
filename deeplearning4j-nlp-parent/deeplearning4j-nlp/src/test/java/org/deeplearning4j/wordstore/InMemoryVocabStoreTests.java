/*-
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

package org.deeplearning4j.wordstore;

import org.deeplearning4j.models.word2vec.VocabWord;
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
    private static final Logger log = LoggerFactory.getLogger(InMemoryVocabStoreTests.class);

    @Test
    public void testStorePut() {
        VocabCache<VocabWord> cache = new InMemoryLookupCache();
        assertFalse(cache.containsWord("hello"));
        cache.addWordToIndex(0, "hello");
        assertTrue(cache.containsWord("hello"));
        assertEquals(1, cache.numWords());
        assertEquals("hello", cache.wordAtIndex(0));
    }



}
