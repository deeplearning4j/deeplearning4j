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
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
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

        assertTrue(lst.contains("Gopie"));
        assertTrue(lst.contains("JIM_HOOK_Senior"));
        /*
        assertEquals(lst.get(0), "Gopie");
        assertEquals(lst.get(1), "JIM_HOOK_Senior");
        */
    }

    @Test
    public void testUIMAIterator() throws Exception {
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        assertEquals(iter.nextSentence(), "No ,  he says now .");
    }

    @Test
    public void testWord2VecAdaGrad() throws Exception {
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder()
                .minWordFrequency(5)
                .iterations(3)
                .learningRate(0.025)
                .layerSize(100)
                .seed(42)
                .sampling(0)
                .negativeSample(0)
                .windowSize(5)
                .modelUtils(new BasicModelUtils<VocabWord>())
                .useAdaGrad(true)
                .iterate(iter)
                .workers(10)
                .tokenizerFactory(t)
                .build();

        vec.fit();

        Collection<String> lst = vec.wordsNearest("day", 10);
        log.info(Arrays.toString(lst.toArray()));

     //   assertEquals(10, lst.size());

        double sim = vec.similarity("day", "night");
        log.info("Day/night similarity: " + sim);
    }


    @Test
    public void testRunWord2Vec() throws Exception {
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
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
                .minWordFrequency(5)
                .iterations(1)
                .layerSize(100)
                .lookupTable(table)
                .stopWords(new ArrayList<String>())
                .vocabCache(cache)
                .seed(42)
                .sampling(0)
                .windowSize(5)
                .modelUtils(new BasicModelUtils<VocabWord>())
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());
        vec.fit();
      //  WordVectorSerializer.writeWordVectors(vec, pathToWriteto);
        WordVectorSerializer.writeFullModel(vec,"/ext/Temp/Models/raw_sentences.dat");
        Collection<String> lst = vec.wordsNearest("day", 10);
        //log.info(Arrays.toString(lst.toArray()));
        printWords("day", lst, vec);

        assertEquals(10, lst.size());

        double sim = vec.similarity("day", "night");
        log.info("Day/night similarity: " + sim);

        assertTrue(sim < 1.0);
        assertTrue(sim > 0.6);


        assertTrue(lst.contains("week"));
        assertTrue(lst.contains("night"));
        assertTrue(lst.contains("year"));

        assertFalse(lst.contains(null));


        lst = vec.wordsNearest("day", 10);
        //log.info(Arrays.toString(lst.toArray()));
        printWords("day", lst, vec);

        assertTrue(lst.contains("week"));
        assertTrue(lst.contains("night"));
        assertTrue(lst.contains("year"));

        new File("cache.ser").delete();
    }

    /**
     * Adding test for cosine similarity, to track changes in Transforms.cosineSim()
     */
    @Test
    public void testCosineSim() {
        double[] array1 = new double[]{1.01, 0.91, 0.81, 0.71};
        double[] array2 = new double[]{1.01, 0.91, 0.81, 0.71};
        double[] array3 = new double[]{1.0, 0.9, 0.8, 0.7};

        double sim12 = Transforms.cosineSim(Nd4j.create(array1), Nd4j.create(array2));
        double sim23 = Transforms.cosineSim(Nd4j.create(array2), Nd4j.create(array3));
        log.info("Arrays 1/2 cosineSim: " + sim12);
        log.info("Arrays 2/3 cosineSim: " + sim23);
        log.info("Arrays 1/2 dot: " + Nd4j.getBlasWrapper().dot(Nd4j.create(array1), Nd4j.create(array2)));
        log.info("Arrays 2/3 dot: " + Nd4j.getBlasWrapper().dot(Nd4j.create(array2), Nd4j.create(array3)));

        assertEquals(1.0d, sim12, 0.01d);
        assertEquals(0.99d, sim23, 0.01d);
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

    private static void printWords(String target, Collection<String> list, Word2Vec vec) {
        System.out.println("Words close to ["+target+"]:");
        for (String word: list) {
            double sim = vec.similarity(target, word);
            System.out.print("'"+ word+"': ["+ sim+"]");
        }
        System.out.print("\n");
    }
//
}

