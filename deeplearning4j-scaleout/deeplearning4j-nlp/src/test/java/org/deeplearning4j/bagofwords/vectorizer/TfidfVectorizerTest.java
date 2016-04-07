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

package org.deeplearning4j.bagofwords.vectorizer;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.fail;
import static org.junit.Assume.assumeNotNull;

/**
 * @author Adam Gibson
 */
public class TfidfVectorizerTest {

    private static final Logger log = LoggerFactory.getLogger(TfidfVectorizerTest.class);


    @Test
    public void testTfIdfVectorizer() throws Exception {
        File rootDir = new ClassPathResource("tripledir").getFile();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        TfidfVectorizer vectorizer = new TfidfVectorizer.Builder()
                .setMinWordFrequency(1)
                .setStopWords(new ArrayList<String>())
                .setTokenizerFactory(tokenizerFactory)
                .setIterator(iter)
//                .labels(labels)
//                .cleanup(true)
                .build();

        vectorizer.fit();
        VocabWord word =vectorizer.getVocabCache().wordFor("file.");
        assumeNotNull(word);
        assertEquals(word,vectorizer.getVocabCache().tokenFor("file."));
        assertEquals(3,vectorizer.getVocabCache().totalNumberOfDocs());

        assertEquals(3, word.getSequencesCount());
        assertEquals(3, word.getElementFrequency(), 0.1);

        VocabWord word1 =vectorizer.getVocabCache().wordFor("1");

        assertEquals(1, word1.getSequencesCount());
        assertEquals(1, word1.getElementFrequency(), 0.1);

        log.info("Labels used: " + vectorizer.getLabelsSource().getLabels());
        assertEquals(3, vectorizer.getLabelsSource().getNumberOfLabelsUsed());

        assertEquals(3, vectorizer.getVocabCache().totalNumberOfDocs());

        assertEquals(11, vectorizer.numWordsEncountered());

        INDArray vector = vectorizer.transform("This is 3 file.");
        log.info("TF-IDF vector: " + Arrays.toString(vector.data().asDouble()));

        assertEquals(0, vector.getDouble(0), 0.001);
        assertEquals(0.088, vector.getDouble(1), 0.001);
        assertEquals(0.088, vector.getDouble(2), 0.001);
        assertEquals(0, vector.getDouble(3), 0.001);
        assertEquals(0.119, vector.getDouble(4), 0.001);
        assertEquals(0, vector.getDouble(5), 0.001);
        assertEquals(0, vector.getDouble(6), 0.001);


        DataSet dataSet = vectorizer.vectorize("This is 3 file.", "label3");
        assertEquals(0.0, dataSet.getLabels().getDouble(0), 0.1);
        assertEquals(0.0, dataSet.getLabels().getDouble(1), 0.1);
        assertEquals(1.0, dataSet.getLabels().getDouble(2), 0.1);


        File tempFile = File.createTempFile("somefile","Dsdas");
        tempFile.deleteOnExit();

        SerializationUtils.saveObject(vectorizer, tempFile);

        TfidfVectorizer vectorizer2 = SerializationUtils.readObject(tempFile);
        vectorizer2.setTokenizerFactory(tokenizerFactory);

        dataSet = vectorizer2.vectorize("This is 3 file.", "label2");
        assertEquals(vector, dataSet.getFeatureMatrix());
    }

    @Test
    @Ignore
    public void testLegacyTFIDF() throws Exception{
        File rootDir = new ClassPathResource("rootdir").getFile();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
        List<String> docStrings = new ArrayList<>();

        while(iter.hasNext())
            docStrings.add(iter.nextSentence());

        iter.reset();

        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        List<String> labels = Arrays.asList("label1","label2");
        TokenizerFactory tokenizerFactory = new UimaTokenizerFactory();
        TextVectorizer vectorizer = new LegacyTfidfVectorizer.Builder()
                .minWords(1)
//                .index(index)
                .cache(cache)
                .stopWords(new ArrayList<String>())
                .tokenize(tokenizerFactory)
                .labels(labels)
                .iterate(iter)
                .build();

        vectorizer.fit();
        try {
            vectorizer.vectorize("",null);
            fail("Vectorizer should receive non-null label.");
        } catch (IllegalArgumentException e) {
            ;
        }

        INDArray vector = vectorizer.transform("This is 1 file.");
        log.info("Vector: " + vector);


    }
}
