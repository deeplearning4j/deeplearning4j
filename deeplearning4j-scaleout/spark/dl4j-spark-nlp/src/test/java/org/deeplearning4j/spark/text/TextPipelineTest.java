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

package org.deeplearning4j.spark.text;

import static org.junit.Assert.*;

import org.apache.commons.lang3.Range;
import org.apache.spark.api.java.JavaRDD;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.spark.models.embeddings.word2vec.Word2Vec;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import java.io.FileWriter;
import java.util.*;

/**
 * @author Jeffrey Tang
 */
public class TextPipelineTest extends BaseSparkTest {

    /**
     *
     * Testing the TextPipeline on a toy example to make sure words and vocabs are counted correctly
     *
     * @throws Exception
     */
    @Test
    public void testTextPipelineSimple() throws Exception {
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("basic/word2vec.txt").getFile().getAbsolutePath(), 3);
        TextPipeline pipeline = new TextPipeline(corpus, 1); //Min word freq
        Pair<VocabCache,Long> pair = pipeline.process();
        InMemoryLookupCache lookupCache = (InMemoryLookupCache)pair.getFirst();

        // vocabWord count
        assertEquals(lookupCache.vocabs.size(), 6);
        // Check vocabWord Index
        ArrayList<Integer> vocabWordInds = new ArrayList<>();
        for (VocabWord vw : lookupCache.vocabs.values()) {
            vocabWordInds.add(vw.getIndex());
        }
        assertTrue(Collections.min(vocabWordInds) == 0);
        assertTrue(Collections.max(vocabWordInds) == 5);
        // Check vocabWord themselves
        // STOP and UNK are not added to
        assertTrue(lookupCache.vocabs.containsKey("She"));
        assertTrue(lookupCache.vocabs.containsKey("found"));
        assertTrue(lookupCache.vocabs.containsKey("one"));
        assertTrue(lookupCache.vocabs.containsKey("two"));
        assertTrue(lookupCache.vocabs.containsKey("ba"));
        assertTrue(lookupCache.vocabs.containsKey("abab"));
        // Check total word count
        assertTrue(pair.getSecond() == 8);
    }

    /**
     *
     * Testing the TextPipeline on a bigger corpus to make sure words and vocabs are counted correctly
     *
     * @throws Exception
     */
    @Test
    public void testTextPipelineFull() throws Exception {
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath(), 3);
        TextPipeline pipeline = new TextPipeline(corpus, 1); //Min word freq
        Pair<VocabCache, Long> pair = pipeline.process();
        InMemoryLookupCache lookupCache = (InMemoryLookupCache) pair.getFirst();

        assertEquals(lookupCache.vocabs.size(), 542);
        assertEquals(lookupCache.vocabs.get("SHOULD").getIndex(), 5);
        assertEquals((int)lookupCache.wordFrequencies.getCount("SHOULD"), 4);
    }

    /**
     * Test Word2Vec on a toy dataset to make sure things at least run and check if words and vocab are
     * counted properly after Word2Vec is trained
     *
     * @throws Exception
     */
    @Test
    public void testWord2VecSimple() throws Exception {
        // Train Word2Vec
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("basic/word2vec.txt").getFile().getAbsolutePath(), 3);
        Word2Vec model = new Word2Vec();
        model.train(corpus);

        // Test lookupcache
        InMemoryLookupCache lookupCache = (InMemoryLookupCache)model.getVocabCacheBroadcast().value();
        // vocabWord count
        assertEquals(lookupCache.vocabs.size(), 7);
        // Check vocabWord Index
        ArrayList<Integer> vocabWordInds = new ArrayList<>();
        for (VocabWord vw : lookupCache.vocabs.values()) {
            vocabWordInds.add(vw.getIndex());
        }
        assertTrue(Collections.min(vocabWordInds) == 1);
        assertTrue(Collections.max(vocabWordInds) == 7);
        // Check vocabWord themselves
        assertTrue(lookupCache.vocabs.containsKey("She"));
        assertTrue(lookupCache.vocabs.containsKey("found"));
        assertTrue(lookupCache.vocabs.containsKey("STOP"));
        assertTrue(lookupCache.vocabs.containsKey("one"));
        assertTrue(lookupCache.vocabs.containsKey("two"));
        assertTrue(lookupCache.vocabs.containsKey("ba"));
        assertTrue(lookupCache.vocabs.containsKey("abab"));
        // Check word frequencies
        assertTrue(lookupCache.wordFrequencies.getCount("She") == 1.0);
        assertTrue(lookupCache.wordFrequencies.getCount("found") == 1.0);
        assertTrue(lookupCache.wordFrequencies.getCount("STOP") == 2.0);
        assertTrue(lookupCache.wordFrequencies.getCount("one") == 1.0);
        assertTrue(lookupCache.wordFrequencies.getCount("two") == 1.0);
        assertTrue(lookupCache.wordFrequencies.getCount("ba") == 1.0);
        assertTrue(lookupCache.wordFrequencies.getCount("abab") == 1.0);
    }

    /**
     *
     * Run Word2Vec to a bigger corpus to make sure the results are sensible
     *
     * @throws Exception
     */
    @Test
    public void testWord2VecFull() throws Exception {
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath());
        Word2Vec model = new Word2Vec();
        model.train(corpus);
        InMemoryLookupCache lookupCache = (InMemoryLookupCache)model.getVocabCacheBroadcast().value();
        Collection<String> lst = model.wordsNearest("day", 10);

    }

}

