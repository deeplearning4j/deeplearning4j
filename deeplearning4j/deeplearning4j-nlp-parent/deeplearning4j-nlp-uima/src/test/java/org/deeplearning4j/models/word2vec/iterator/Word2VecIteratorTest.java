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

package org.deeplearning4j.models.word2vec.iterator;

import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareFileSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.DataSet;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.assertNotEquals;

/**
 * Created by agibsonccc on 3/5/15.
 */
public class Word2VecIteratorTest {
    private Word2Vec vec;

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Before
    public void before() throws Exception {
        if (vec == null) {
            ClassPathResource resource = new ClassPathResource("/labeled/");
            File dir = testDir.newFolder();
            resource.copyDirectory(dir);
            SentenceIterator iter = UimaSentenceIterator.createWithPath(dir.getAbsolutePath());
            new File("cache.ser").delete();

            TokenizerFactory t = new UimaTokenizerFactory();

            vec = new Word2Vec.Builder().minWordFrequency(1).iterations(5).layerSize(100)
                            .stopWords(new ArrayList<String>()).useUnknown(true).windowSize(5).iterate(iter)
                            .tokenizerFactory(t).build();
            vec.fit();

        }
    }

    @Test
    public void testLabeledExample() throws Exception {

        INDArray unk = vec.getWordVectorMatrix(Word2Vec.DEFAULT_UNK);
        assertNotEquals(null, unk);

        unk = vec.getWordVectorMatrix("2131241sdasdas");
        assertNotEquals(null, unk);

        ClassPathResource resource = new ClassPathResource("/labeled/");
        File dir = testDir.newFolder();
        resource.copyDirectory(dir);

        Word2VecDataSetIterator iter = new Word2VecDataSetIterator(vec,
                        new LabelAwareFileSentenceIterator(null, dir),
                        Arrays.asList("negative", "positive", "neutral"));
        DataSet next = iter.next();

    }

}

