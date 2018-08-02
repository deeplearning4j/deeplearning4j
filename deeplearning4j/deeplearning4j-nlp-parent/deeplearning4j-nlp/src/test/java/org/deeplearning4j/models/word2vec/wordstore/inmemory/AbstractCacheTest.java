/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
