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

package org.deeplearning4j.bagofwords.vectorizer;


import lombok.extern.slf4j.Slf4j;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.IndexAccumulation;
import org.nd4j.linalg.api.ops.impl.indexaccum.IMax;
import org.nd4j.linalg.dataset.DataSet;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.util.SerializationUtils;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assume.assumeNotNull;

/**
 *@author Adam Gibson
 */
@Slf4j
public class BagOfWordsVectorizerTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();



    @Test(timeout = 60000L)
    public void testBagOfWordsVectorizer() throws Exception {
        File rootDir = new ClassPathResource("rootdir").getFile();
        LabelAwareSentenceIterator iter = new LabelAwareFileSentenceIterator(rootDir);
        List<String> labels = Arrays.asList("label1", "label2");
        TokenizerFactory tokenizerFactory = new DefaultTokenizerFactory();

        BagOfWordsVectorizer vectorizer = new BagOfWordsVectorizer.Builder().setMinWordFrequency(1)
                        .setStopWords(new ArrayList<String>()).setTokenizerFactory(tokenizerFactory).setIterator(iter)
                        .allowParallelTokenization(false)
                        //                .labels(labels)
                        //                .cleanup(true)
                        .build();

        vectorizer.fit();
        VocabWord word = vectorizer.getVocabCache().wordFor("file.");
        assumeNotNull(word);
        assertEquals(word, vectorizer.getVocabCache().tokenFor("file."));
        assertEquals(2, vectorizer.getVocabCache().totalNumberOfDocs());

        assertEquals(2, word.getSequencesCount());
        assertEquals(2, word.getElementFrequency(), 0.1);

        VocabWord word1 = vectorizer.getVocabCache().wordFor("1");

        assertEquals(1, word1.getSequencesCount());
        assertEquals(1, word1.getElementFrequency(), 0.1);

        log.info("Labels used: " + vectorizer.getLabelsSource().getLabels());
        assertEquals(2, vectorizer.getLabelsSource().getNumberOfLabelsUsed());

        ///////////////////
        INDArray array = vectorizer.transform("This is 2 file.");
        log.info("Transformed array: " + array);
        assertEquals(5, array.columns());


        VocabCache<VocabWord> vocabCache = vectorizer.getVocabCache();

        assertEquals(2, array.getDouble(vocabCache.tokenFor("This").getIndex()), 0.1);
        assertEquals(2, array.getDouble(vocabCache.tokenFor("is").getIndex()), 0.1);
        assertEquals(2, array.getDouble(vocabCache.tokenFor("file.").getIndex()), 0.1);
        assertEquals(0, array.getDouble(vocabCache.tokenFor("1").getIndex()), 0.1);
        assertEquals(1, array.getDouble(vocabCache.tokenFor("2").getIndex()), 0.1);

        DataSet dataSet = vectorizer.vectorize("This is 2 file.", "label2");
        assertEquals(array, dataSet.getFeatures());

        INDArray labelz = dataSet.getLabels();
        log.info("Labels array: " + labelz);

        int idx2 = ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(labelz))).getFinalResult();

        //        assertEquals(1.0, dataSet.getLabels().getDouble(0), 0.1);
        //        assertEquals(0.0, dataSet.getLabels().getDouble(1), 0.1);

        dataSet = vectorizer.vectorize("This is 1 file.", "label1");

        assertEquals(2, dataSet.getFeatures().getDouble(vocabCache.tokenFor("This").getIndex()), 0.1);
        assertEquals(2, dataSet.getFeatures().getDouble(vocabCache.tokenFor("is").getIndex()), 0.1);
        assertEquals(2, dataSet.getFeatures().getDouble(vocabCache.tokenFor("file.").getIndex()), 0.1);
        assertEquals(1, dataSet.getFeatures().getDouble(vocabCache.tokenFor("1").getIndex()), 0.1);
        assertEquals(0, dataSet.getFeatures().getDouble(vocabCache.tokenFor("2").getIndex()), 0.1);

        int idx1 = ((IndexAccumulation) Nd4j.getExecutioner().exec(new IMax(dataSet.getLabels()))).getFinalResult();

        //assertEquals(0.0, dataSet.getLabels().getDouble(0), 0.1);
        //assertEquals(1.0, dataSet.getLabels().getDouble(1), 0.1);

        assertNotEquals(idx2, idx1);

        // Serialization check
        File tempFile = createTempFile("fdsf", "fdfsdf");
        tempFile.deleteOnExit();

        SerializationUtils.saveObject(vectorizer, tempFile);

        BagOfWordsVectorizer vectorizer2 = SerializationUtils.readObject(tempFile);
        vectorizer2.setTokenizerFactory(tokenizerFactory);

        dataSet = vectorizer2.vectorize("This is 2 file.", "label2");
        assertEquals(array, dataSet.getFeatures());
    }

    private File createTempFile(String prefix, String suffix) throws IOException {
        return testDir.newFile(prefix + "-" + System.nanoTime() + suffix);
    }

}
