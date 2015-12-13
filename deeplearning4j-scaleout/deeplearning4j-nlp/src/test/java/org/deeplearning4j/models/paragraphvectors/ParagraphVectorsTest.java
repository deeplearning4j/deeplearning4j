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

package org.deeplearning4j.models.paragraphvectors;


import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareUimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.*;

/**
 * Created by agibsonccc on 12/3/14.
 */
public class ParagraphVectorsTest {
    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsTest.class);

    @Before
    public void before() {
        new File("word2vec-index").delete();
    }



    @Test
    public void testWord2VecRunThroughVectors() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile().getParentFile();
        LabelAwareSentenceIterator iter = LabelAwareUimaSentenceIterator.createWithPath(file.getAbsolutePath());


        TokenizerFactory t = new UimaTokenizerFactory();


        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1).iterations(5).labels(Arrays.asList("label1", "deeple"))
                .layerSize(100)
                .stopWords(new ArrayList<String>())
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());


        vec.fit();
        double sim = vec.similarity("day","night");
        log.info("day/night similarity: " + sim);
        new File("cache.ser").delete();

    }

    /**
     * This test checks, how vocab is built using SentenceIterator provided, without labels.
     *
     * @throws Exception
     */
    @Test
    public void testParagraphVectorsVocabBuilding1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();//.getParentFile();
        SentenceIterator iter = new BasicLineIterator(file); //UimaSentenceIterator.createWithPath(file.getAbsolutePath());

        int numberOfLines = 0;
        while (iter.hasNext()) {
            iter.nextSentence();
            numberOfLines++;
        }

        iter.reset();

        InMemoryLookupCache cache = new InMemoryLookupCache(false);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

       // LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1).iterations(5)
                .layerSize(100)
          //      .labelsGenerator(source)
                .windowSize(5)
                .iterate(iter)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .build();

        vec.buildVocab();

        LabelsSource source = vec.getLabelsSource();


        //VocabCache cache = vec.getVocab();
        log.info("Number of lines in corpus: " + numberOfLines);
        assertEquals(numberOfLines, source.getLabels().size());
        assertEquals(97162, source.getLabels().size());

        assertNotEquals(null, cache);
        assertEquals(97406, cache.numWords());

        // proper number of words for minWordsFrequency = 1 is 244
        assertEquals(244, cache.numWords() - source.getLabels().size());
    }

    @Test
    public void testParagraphVectorsModelling1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

        InMemoryLookupCache cache = new InMemoryLookupCache(false);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(3)
                .epochs(1)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(false)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build();

        vec.fit();


        int cnt1 = cache.wordFrequency("day");
        int cnt2 = cache.wordFrequency("me");

        assertNotEquals(1, cnt1);
        assertNotEquals(1, cnt2);
        assertNotEquals(cnt1, cnt2);

        /*
            We have few lines that contain pretty close words invloved.
            These sentences should be pretty close to each other in vector space
         */
        // line 3721: This is my way .
        // line 6348: This is my case .
        // line 9836: This is my house .
        // line 12493: This is my world .
        // line 16393: This is my work .

        // line 9853: We now have one .

        double similarityD = vec.similarity("day", "night");
        log.info("day/night similarity: " + similarityD);

        double similarityW = vec.similarity("way", "work");
        log.info("way/work similarity: " + similarityW);

        double similarityH = vec.similarity("house", "world");
        log.info("house/world similarity: " + similarityH);

        double similarityC = vec.similarity("case", "way");
        log.info("case/way similarity: " + similarityC);

        double similarity1 = vec.similarity("DOC_9835", "DOC_12492");
        log.info("9835/12492 similarity: " + similarity1);
//        assertTrue(similarity1 > 0.7d);

        double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
        log.info("3720/16392 similarity: " + similarity2);
//        assertTrue(similarity2 > 0.7d);

        double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
        log.info("6347/3720 similarity: " + similarity3);
//        assertTrue(similarity2 > 0.7d);

        // likelihood in this case should be significantly lower
        double similarityX = vec.similarity("DOC_3720", "DOC_9852");
        log.info("3720/9852 similarity: " + similarityX);
        assertTrue(similarityX < 0.5d);
    }

    @Test
    public void testParagraphVectorsWithWordVectorsModelling1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

        InMemoryLookupCache cache = new InMemoryLookupCache(false);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(3)
                .epochs(1)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(true)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build();

        vec.fit();


        int cnt1 = cache.wordFrequency("day");
        int cnt2 = cache.wordFrequency("me");

        assertNotEquals(1, cnt1);
        assertNotEquals(1, cnt2);
        assertNotEquals(cnt1, cnt2);

        /*
            We have few lines that contain pretty close words invloved.
            These sentences should be pretty close to each other in vector space
         */
        // line 3721: This is my way .
        // line 6348: This is my case .
        // line 9836: This is my house .
        // line 12493: This is my world .
        // line 16393: This is my work .

        // line 9853: We now have one .

        double similarityD = vec.similarity("day", "night");
        log.info("day/night similarity: " + similarityD);

        double similarityW = vec.similarity("way", "work");
        log.info("way/work similarity: " + similarityW);

        double similarityH = vec.similarity("house", "world");
        log.info("house/world similarity: " + similarityH);

        double similarityC = vec.similarity("case", "way");
        log.info("case/way similarity: " + similarityC);

        double similarity1 = vec.similarity("DOC_9835", "DOC_12492");
        log.info("9835/12492 similarity: " + similarity1);
//        assertTrue(similarity1 > 0.7d);

        double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
        log.info("3720/16392 similarity: " + similarity2);
//        assertTrue(similarity2 > 0.7d);

        double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
        log.info("6347/3720 similarity: " + similarity3);
//        assertTrue(similarity2 > 0.7d);

        // likelihood in this case should be significantly lower
        // however, since corpus is small, and weight initialization is random-based, sometimes this test CAN fail
        double similarityX = vec.similarity("DOC_3720", "DOC_9852");
        log.info("3720/9852 similarity: " + similarityX);
        assertTrue(similarityX < 0.5d);
    }

    /*
        In this test we'll build w2v model, and will use it's vocab and weights for ParagraphVectors.
        IS NOT READY YET
    */
    @Test
    public void testParagraphVectorsOverExistingWordVectorsModel() throws Exception {
        /*
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

        InMemoryLookupCache cache = new InMemoryLookupCache(false);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec wordVectors = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(1)
                .epochs(1)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        wordVectors.fit();

        INDArray vector_day1 = wordVectors.getWordVectorMatrix("day");
        double similarityD = wordVectors.similarity("day", "night");
        log.info("day/night similarity: " + similarityD);
        assertTrue(similarityD > 0.65d);

        // At this moment we have ready w2v model. It's time to use it for ParagraphVectors

        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .iterate(iter)
                .iterations(1)
                .epochs(1)
                .tokenizerFactory(t)
                .trainWordVectors(false)
                .wordVectorsModel(wordVectors)
                .labelsTemplate("SNT_%d")
                .build();

        paragraphVectors.fit();
        INDArray vector_day2 = paragraphVectors.getWordVectorMatrix("day");
        double crossDay = arraysSimilarity(vector_day1, vector_day2);

        log.info("Day1: " + vector_day1);
        log.info("Day2: " + vector_day2);
        log.info("Cross-Day similarity: " + crossDay);

        assertTrue(crossDay > 0.9d);
        */
    }

    private double arraysSimilarity(INDArray array1, INDArray array2) {
        if (array1.equals(array2)) return 1.0;

        INDArray vector = Transforms.unitVec(array1);
        INDArray vector2 = Transforms.unitVec(array2);
        if(vector == null || vector2 == null)
            return -1;
        return  Nd4j.getBlasWrapper().dot(vector, vector2);

    }
}
