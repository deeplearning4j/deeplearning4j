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

package org.deeplearning4j.models.glove;

import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentencePreProcessor;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by agibsonccc on 12/3/14.
 */
public class GloveTest {
    private static final Logger log = LoggerFactory.getLogger(GloveTest.class);
    private Glove glove;
    private SentenceIterator iter;

    @Before
    public void before() throws Exception {

        ClassPathResource resource = new ClassPathResource("/raw_sentences.txt");
        File file = resource.getFile();
        iter = new LineSentenceIterator(file);
        iter.setPreProcessor(new SentencePreProcessor() {
            @Override
            public String preProcess(String sentence) {
                return sentence.toLowerCase();
            }
        });

    }


    @Ignore
    @Test
    public void testGlove() throws Exception {
        /*
        glove = new Glove.Builder().iterate(iter).symmetric(true).shuffle(true)
                .minWordFrequency(1).iterations(10).learningRate(0.1)
                .layerSize(300)
                .build();
        
        glove.fit();
        Collection<String> words = glove.wordsNearest("day", 20);
        log.info("Nearest words to 'day': " + words);
        assertTrue(words.contains("week"));
        
        */

    }

    @Ignore
    @Test
    public void testGloVe1() throws Exception {
        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();

        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Glove glove = new Glove.Builder().iterate(iter).tokenizerFactory(t).alpha(0.75).learningRate(0.1).epochs(45)
                        .xMax(100).shuffle(true).symmetric(true).build();

        glove.fit();

        double simD = glove.similarity("day", "night");
        double simP = glove.similarity("best", "police");



        log.info("Day/night similarity: " + simD);
        log.info("Best/police similarity: " + simP);

        Collection<String> words = glove.wordsNearest("day", 10);
        log.info("Nearest words to 'day': " + words);


        assertTrue(simD > 0.7);

        // actually simP should be somewhere at 0
        assertTrue(simP < 0.5);

        assertTrue(words.contains("night"));
        assertTrue(words.contains("year"));
        assertTrue(words.contains("week"));

        File tempFile = File.createTempFile("glove", "temp");
        tempFile.deleteOnExit();

        INDArray day1 = glove.getWordVectorMatrix("day").dup();

        WordVectorSerializer.writeWordVectors(glove, tempFile);

        WordVectors vectors = WordVectorSerializer.loadTxtVectors(tempFile);

        INDArray day2 = vectors.getWordVectorMatrix("day").dup();

        assertEquals(day1, day2);

        tempFile.delete();
    }
}
