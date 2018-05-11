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

package org.deeplearning4j.bagofwords.vectorizer;

import lombok.val;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.sentenceiterator.CollectionSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.DefaultTokenizer;
import org.deeplearning4j.text.tokenization.tokenizer.Tokenizer;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.concurrent.atomic.AtomicLong;

import static org.junit.Assert.assertArrayEquals;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assume.assumeNotNull;

/**
 * @author Adam Gibson
 */
public class TfidfVectorizerTest {

    private static final Logger log = LoggerFactory.getLogger(TfidfVectorizerTest.class);


    @Test(timeout = 60000L)
    public void testTfIdfVectorizer() throws Exception {
        File rootDir = new ClassPathResource("tripledir").getFile();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        TfidfVectorizer vectorizer = new TfidfVectorizer.Builder().setMinWordFrequency(1)
                        .setStopWords(new ArrayList<String>()).setTokenizerFactory(tokenizerFactory).setIterator(iter)
                        .allowParallelTokenization(false)
                        //                .labels(labels)
                        //                .cleanup(true)
                        .build();

        vectorizer.fit();
        VocabWord word = vectorizer.getVocabCache().wordFor("file.");
        assumeNotNull(word);
        assertEquals(word, vectorizer.getVocabCache().tokenFor("file."));
        assertEquals(3, vectorizer.getVocabCache().totalNumberOfDocs());

        assertEquals(3, word.getSequencesCount());
        assertEquals(3, word.getElementFrequency(), 0.1);

        VocabWord word1 = vectorizer.getVocabCache().wordFor("1");

        assertEquals(1, word1.getSequencesCount());
        assertEquals(1, word1.getElementFrequency(), 0.1);

        log.info("Labels used: " + vectorizer.getLabelsSource().getLabels());
        assertEquals(3, vectorizer.getLabelsSource().getNumberOfLabelsUsed());

        assertEquals(3, vectorizer.getVocabCache().totalNumberOfDocs());

        assertEquals(11, vectorizer.numWordsEncountered());

        INDArray vector = vectorizer.transform("This is 3 file.");
        log.info("TF-IDF vector: " + Arrays.toString(vector.data().asDouble()));

        VocabCache<VocabWord> vocabCache = vectorizer.getVocabCache();

        assertEquals(.04402, vector.getDouble(vocabCache.tokenFor("This").getIndex()), 0.001);
        assertEquals(.04402, vector.getDouble(vocabCache.tokenFor("is").getIndex()), 0.001);
        assertEquals(0.119, vector.getDouble(vocabCache.tokenFor("3").getIndex()), 0.001);
        assertEquals(0, vector.getDouble(vocabCache.tokenFor("file.").getIndex()), 0.001);



        DataSet dataSet = vectorizer.vectorize("This is 3 file.", "label3");
        //assertEquals(0.0, dataSet.getLabels().getDouble(0), 0.1);
        //assertEquals(0.0, dataSet.getLabels().getDouble(1), 0.1);
        //assertEquals(1.0, dataSet.getLabels().getDouble(2), 0.1);
        int cnt = 0;
        for (int i = 0; i < 3; i++) {
            if (dataSet.getLabels().getDouble(i) > 0.1)
                cnt++;
        }

        assertEquals(1, cnt);


        File tempFile = File.createTempFile("somefile", "Dsdas");
        tempFile.deleteOnExit();

        SerializationUtils.saveObject(vectorizer, tempFile);

        TfidfVectorizer vectorizer2 = SerializationUtils.readObject(tempFile);
        vectorizer2.setTokenizerFactory(tokenizerFactory);

        dataSet = vectorizer2.vectorize("This is 3 file.", "label2");
        assertEquals(vector, dataSet.getFeatureMatrix());
    }

    @Test(timeout = 10000L)
    public void testParallelFlag1() throws Exception {
        val vectorizer = new TfidfVectorizer.Builder()
                .allowParallelTokenization(false)
                .build();

        assertFalse(vectorizer.isParallel);
    }


    @Test(expected = ND4JIllegalStateException.class, timeout = 20000L)
    public void testParallelFlag2() throws Exception {
        val collection = new ArrayList<String>();
        collection.add("First string");
        collection.add("Second string");
        collection.add("Third string");
        collection.add("");
        collection.add("Fifth string");
//        collection.add("caboom");

        val vectorizer = new TfidfVectorizer.Builder()
                .allowParallelTokenization(false)
                .setIterator(new CollectionSentenceIterator(collection))
                .setTokenizerFactory(new ExplodingTokenizerFactory(8, -1))
                .build();

        vectorizer.buildVocab();


        log.info("Fitting vectorizer...");

        vectorizer.fit();
    }

    @Test(expected = ND4JIllegalStateException.class, timeout = 20000L)
    public void testParallelFlag3() throws Exception {
        val collection = new ArrayList<String>();
        collection.add("First string");
        collection.add("Second string");
        collection.add("Third string");
        collection.add("");
        collection.add("Fifth string");
        collection.add("Long long long string");
        collection.add("Sixth string");

        val vectorizer = new TfidfVectorizer.Builder()
                .allowParallelTokenization(false)
                .setIterator(new CollectionSentenceIterator(collection))
                .setTokenizerFactory(new ExplodingTokenizerFactory(-1, 4))
                .build();

        vectorizer.buildVocab();


        log.info("Fitting vectorizer...");

        vectorizer.fit();
    }


    protected class ExplodingTokenizerFactory extends DefaultTokenizerFactory {
        protected int triggerSentence;
        protected int triggerWord;
        protected AtomicLong cnt = new AtomicLong(0);

        protected ExplodingTokenizerFactory(int triggerSentence, int triggerWord) {
            this.triggerSentence = triggerSentence;
            this.triggerWord = triggerWord;
        }

        @Override
        public Tokenizer create(String toTokenize) {

            if (triggerSentence >= 0 && cnt.incrementAndGet() >= triggerSentence)
                throw new ND4JIllegalStateException("TokenizerFactory exploded");


            val tkn = new ExplodingTokenizer(toTokenize, triggerWord);

            return tkn;
        }
    }

    protected class ExplodingTokenizer extends DefaultTokenizer {
        protected int triggerWord;

        public ExplodingTokenizer(String string, int triggerWord) {
            super(string);

            this.triggerWord = triggerWord;
            if (this.triggerWord >= 0)
                if (this.countTokens() >= triggerWord)
                    throw new ND4JIllegalStateException("Tokenizer exploded");
        }
    }
}
