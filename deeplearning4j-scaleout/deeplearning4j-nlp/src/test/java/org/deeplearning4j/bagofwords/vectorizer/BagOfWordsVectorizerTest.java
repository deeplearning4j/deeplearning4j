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


import static org.junit.Assume.*;

import org.apache.commons.io.FileUtils;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.invertedindex.InvertedIndex;
import org.deeplearning4j.text.invertedindex.LuceneInvertedIndex;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;


import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 *@author Adam Gibson
 */
public class BagOfWordsVectorizerTest {

    private static final Logger log = LoggerFactory.getLogger(BagOfWordsVectorizerTest.class);




    @Test
    public void testBagOfWordsVectorizer() throws Exception {
        File rootDir = new ClassPathResource("rootdir").getFile();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
        List<String> labels = Arrays.asList("label1", "label2");
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        BagOfWordsVectorizer vectorizer = new BagOfWordsVectorizer.Builder()
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
        assertEquals(2,vectorizer.getVocabCache().totalNumberOfDocs());

        assertEquals(2, word.getSequencesCount());
        assertEquals(2, word.getElementFrequency(), 0.1);

        VocabWord word1 =vectorizer.getVocabCache().wordFor("1");

        assertEquals(1, word1.getSequencesCount());
        assertEquals(1, word1.getElementFrequency(), 0.1);

        log.info("Labels used: " + vectorizer.getLabelsSource().getLabels());
        assertEquals(2, vectorizer.getLabelsSource().getNumberOfLabelsUsed());

        ///////////////////
        INDArray array = vectorizer.transform("This is 2 file.");
        log.info("Transformed array: " + array);
        assertEquals(5, array.columns());

        assertEquals(2, array.getDouble(0), 0.1);
        assertEquals(2, array.getDouble(1), 0.1);
        assertEquals(2, array.getDouble(2), 0.1);
        assertEquals(0, array.getDouble(3), 0.1);
        assertEquals(1, array.getDouble(4), 0.1);

        DataSet dataSet = vectorizer.vectorize("This is 2 file.", "label2");
        assertEquals(array, dataSet.getFeatureMatrix());

        INDArray labelz = dataSet.getLabels();
        log.info("Labels array: " + labelz);
        assertEquals(1.0, dataSet.getLabels().getDouble(0), 0.1);
        assertEquals(0.0, dataSet.getLabels().getDouble(1), 0.1);

        dataSet = vectorizer.vectorize("This is 1 file.", "label1");

        assertEquals(2, dataSet.getFeatureMatrix().getDouble(0), 0.1);
        assertEquals(2, dataSet.getFeatureMatrix().getDouble(1), 0.1);
        assertEquals(2, dataSet.getFeatureMatrix().getDouble(2), 0.1);
        assertEquals(1, dataSet.getFeatureMatrix().getDouble(3), 0.1);
        assertEquals(0, dataSet.getFeatureMatrix().getDouble(4), 0.1);

        assertEquals(0.0, dataSet.getLabels().getDouble(0), 0.1);
        assertEquals(1.0, dataSet.getLabels().getDouble(1), 0.1);

        // Serialization check
        File tempFile = File.createTempFile("fdsf", "fdfsdf");
        tempFile.deleteOnExit();

        SerializationUtils.saveObject(vectorizer, tempFile);

        BagOfWordsVectorizer vectorizer2 = SerializationUtils.readObject(tempFile);
        vectorizer2.setTokenizerFactory(tokenizerFactory);

        dataSet = vectorizer2.vectorize("This is 2 file.", "label2");
        assertEquals(array, dataSet.getFeatureMatrix());
    }


}
