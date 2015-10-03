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

package org.deeplearning4j.models.word2vec;

import com.google.common.primitives.Doubles;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;


/**
 * @author jeffreytang
 */
public class Word2VecTests {

    private static final Logger log = LoggerFactory.getLogger(Word2VecTests.class);

    private File inputFile;
    private String pathToWriteto;
    private WordVectors googleModel;

    @Before
    public void before() throws Exception {
        File googleModelTextFile = new ClassPathResource("word2vecserialization/google_news_30.txt").getFile();
        googleModel = WordVectorSerializer.loadGoogleModel(googleModelTextFile, false);
        inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        pathToWriteto = "testing_word2vec_serialization.txt";
        FileUtils.deleteDirectory(new File("word2vec-index"));
    }

    @Test
    public void testGoogleModelLoaded() throws Exception {
        assertEquals(googleModel.vocab().numWords(), 30);
        assertTrue(googleModel.hasWord("Morgan_Freeman"));
        double[] wordVector = googleModel.getWordVector("Morgan_Freeman");
        assertTrue(wordVector.length == 300);
        assertEquals(Doubles.asList(wordVector).get(0), 0.044423, 1e-3);
    }

    @Test
    public void testSimilarity() throws Exception {
        testGoogleModelLoaded();
        assertEquals(googleModel.similarity("Benkovic", "Boeremag_trialists"), 0.1204, 1e-2);
        assertEquals(googleModel.similarity("Benkovic", "Gopie"), 0.3350, 1e-2);
        assertEquals(googleModel.similarity("Benkovic", "Youku.com"), 0.0116, 1e-2);
    }

    @Test
    public void testWordsNearest() throws Exception {
        testGoogleModelLoaded();
        List<Object> lst = Arrays.asList(googleModel.wordsNearest("Benkovic", 10).toArray());
        assertEquals(lst.get(0), "Gopie");
        assertEquals(lst.get(1), "JIM_HOOK_Senior");
    }

    @Test
    public void testUIMAIterator() throws Exception {
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        assertEquals(iter.nextSentence(), "No ,  he says now .");
    }


    @Test
    public void testRunWord2Vec() throws Exception {
        // Strip white space before and after for each line
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        InMemoryLookupCache cache = new InMemoryLookupCache();
        WeightLookupTable table = new InMemoryLookupTable.Builder()
                .vectorLength(100)
                .useAdaGrad(false)
                .cache(cache)
                .lr(0.025f).build();

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5).iterations(1)
                .layerSize(100).lookupTable(table)
                .stopWords(new ArrayList<String>())
                .vocabCache(cache).seed(42)
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());
        vec.fit();
        WordVectorSerializer.writeWordVectors(vec, pathToWriteto);
        Collection<String> lst = vec.wordsNearest("day", 10);
        System.out.println(Arrays.toString(lst.toArray()));
        System.out.println(vec.similarity("day", "night"));
        new File("cache.ser").delete();
    }

    @Test
    public void testLoadingWordVectors() throws Exception {
        File modelFile = new File(pathToWriteto);
        if (!modelFile.exists()) {
            testRunWord2Vec();
        }
        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(modelFile);
        Collection<String> lst = wordVectors.wordsNearest("day", 10);
        System.out.println(Arrays.toString(lst.toArray()));
    }

//    @Test
//    public void testWord2VecRunThroughVectorsTsne() throws Exception {
//        ClassPathResource resource = new ClassPathResource("/basic2/line2.txt");
//        File file = resource.getFile().getParentFile();
//        SentenceIterator iter = UimaSentenceIterator.createWithPath(file.getAbsolutePath());
//        new File("cache.ser").delete();
//
//        TokenizerFactory t = new UimaTokenizerFactory();
//
//        Word2Vec vec = new Word2Vec.Builder()
//                .minWordFrequency(1).iterations(5)
//                .layerSize(100)
//                .stopWords(new ArrayList<String>())
//                .windowSize(5).iterate(iter).tokenizerFactory(t).build();
//
//        assertEquals(new ArrayList<String>(), vec.getStopWords());
//
//        vec.fit();
//        Tsne calculation = new Tsne.Builder().setMaxIter(1).usePca(false).setSwitchMomentumIteration(20)
//                .normalize(true).useAdaGrad(true).learningRate(500f).perplexity(20f).minGain(1e-1f)
//                .build();
//
//        vec.lookupTable().plotVocab(calculation);
//        WordVectorSerializer.writeTsneFormat(vec,calculation.getY(),new File("test.csv"));
//        double sim = vec.similarity("Adam","deeplearning4j");
//        new File("cache.ser").delete();
//
//
//
//    }

}
