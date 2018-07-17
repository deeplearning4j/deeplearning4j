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

package org.deeplearning4j.spark.models.word2vec;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.junit.After;
import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * Tests for new Spark Word2Vec implementation
 *
 * @author raver119@gmail.com
 */
public class SparkWord2VecTest {
    private static List<String> sentences;
    private JavaSparkContext sc;

    @Before
    public void setUp() throws Exception {
        if (sentences == null) {
            sentences = new ArrayList<>();

            sentences.add("one two thee four");
            sentences.add("some once again");
            sentences.add("one another sentence");
        }

        SparkConf sparkConf = new SparkConf().setMaster("local[8]").setAppName("SeqVecTests");
        sc = new JavaSparkContext(sparkConf);
    }

    @After
    public void tearDown() throws Exception {
        sc.stop();
    }

    @Test
    public void testStringsTokenization1() throws Exception {
        JavaRDD<String> rddSentences = sc.parallelize(sentences);

        SparkWord2Vec word2Vec = new SparkWord2Vec();

        word2Vec.fitSentences(rddSentences);

        VocabCache<ShallowSequenceElement> vocabCache = word2Vec.getShallowVocabCache();

        assertNotEquals(null, vocabCache);

        assertEquals(9, vocabCache.numWords());
        assertEquals(2.0, vocabCache.wordFor(SequenceElement.getLongHash("one")).getElementFrequency(), 1e-5);
        assertEquals(1.0, vocabCache.wordFor(SequenceElement.getLongHash("two")).getElementFrequency(), 1e-5);
    }
}
