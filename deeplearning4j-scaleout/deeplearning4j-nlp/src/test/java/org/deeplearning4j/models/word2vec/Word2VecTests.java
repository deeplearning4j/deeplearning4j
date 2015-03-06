/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.models.word2vec;

import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.plot.Tsne;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;

import static org.junit.Assert.assertEquals;



/**
 * Created by agibsonccc on 8/31/14.
 */
public class Word2VecTests {

    private static final Logger log = LoggerFactory.getLogger(Word2VecTests.class);


    @Before
    public void before() {
        new File("word2vec-index").delete();
    }

    @Test
    public void testWord2VecRunThroughVectors() throws Exception {
        ClassPathResource resource = new ClassPathResource("/basic2/line2.txt");
        File file = resource.getFile().getParentFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(file.getAbsolutePath());
        new File("cache.ser").delete();
        InMemoryLookupCache cache = new InMemoryLookupCache();


        TokenizerFactory t = new UimaTokenizerFactory();

        WeightLookupTable table = new InMemoryLookupTable
                .Builder()
                .vectorLength(100).useAdaGrad(false).cache(cache)
                .lr(0.025f).build();

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1).iterations(5)
                .layerSize(100).lookupTable(table)
                .stopWords(new ArrayList<String>())
                .vocabCache(cache)
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());


        vec.fit();


        double sim = vec.similarity("Adam","deeplearning4j");
        new File("cache.ser").delete();


    }



    @Test
    public void testWord2VecRunThroughVectorsTsne() throws Exception {
        ClassPathResource resource = new ClassPathResource("/basic2/line2.txt");
        File file = resource.getFile().getParentFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(file.getAbsolutePath());
        new File("cache.ser").delete();
        InMemoryLookupCache cache = new InMemoryLookupCache();


        TokenizerFactory t = new UimaTokenizerFactory();

        WeightLookupTable table = new InMemoryLookupTable
                .Builder()
                .vectorLength(100).useAdaGrad(false).cache(cache)
                .lr(0.025f).build();

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(1).iterations(5)
                .layerSize(100).lookupTable(table)
                .stopWords(new ArrayList<String>())
                .vocabCache(cache)
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());


        vec.fit();
        Tsne calculation = new Tsne.Builder().setMaxIter(1).usePca(false).setSwitchMomentumIteration(20)
                .normalize(true).useAdaGrad(true).learningRate(500f).perplexity(20f).minGain(1e-1f)
                .build();

        vec.lookupTable().plotVocab(calculation);
        WordVectorSerializer.writeTsneFormat(vec,calculation.getY(),new File("test.csv"));
        double sim = vec.similarity("Adam","deeplearning4j");
        new File("cache.ser").delete();



    }

}
