/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.spark.models.word2vec;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.sequencevectors.sequence.SequenceElement;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.sequencevectors.export.ExportContainer;
import org.deeplearning4j.spark.models.sequencevectors.export.SparkModelExporter;
import org.deeplearning4j.spark.models.sequencevectors.learning.elements.SparkSkipGram;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.junit.jupiter.api.*;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.parameterserver.distributed.conf.VoidConfiguration;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.*;
@Tag(TagNames.FILE_IO)
@Tag(TagNames.SPARK)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
public class SparkWord2VecTest extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 120000L;
    }

    private static List<String> sentences;
    private JavaSparkContext sc;

    @BeforeEach
    public void setUp() throws Exception {
        if (sentences == null) {
            sentences = new ArrayList<>();

            sentences.add("one two thee four");
            sentences.add("some once again");
            sentences.add("one another sentence");
        }

        SparkConf sparkConf = new SparkConf().setMaster("local[8]")
                .set("spark.driver.host", "localhost")
                .setAppName("SeqVecTests");
        sc = new JavaSparkContext(sparkConf);
    }

    @AfterEach
    public void tearDown() throws Exception {
        sc.stop();
    }

    @Test
    @Disabled("AB 2019/05/21 - Failing - Issue #7657")
    public void testStringsTokenization1() throws Exception {
        JavaRDD<String> rddSentences = sc.parallelize(sentences);

        SparkWord2Vec word2Vec = new SparkWord2Vec();
        word2Vec.getConfiguration().setTokenizerFactory(DefaultTokenizerFactory.class.getCanonicalName());
        word2Vec.getConfiguration().setElementsLearningAlgorithm("org.deeplearning4j.spark.models.sequencevectors.learning.elements.SparkSkipGram");
        word2Vec.setExporter(new SparkModelExporter<VocabWord>() {
            @Override
            public void export(JavaRDD<ExportContainer<VocabWord>> rdd) {
                rdd.foreach(new TestFn());
            }
        });


        word2Vec.fitSentences(rddSentences);

        VocabCache<ShallowSequenceElement> vocabCache = word2Vec.getShallowVocabCache();

        assertNotEquals(null, vocabCache);

        assertEquals(9, vocabCache.numWords());
        assertEquals(2.0, vocabCache.wordFor(SequenceElement.getLongHash("one")).getElementFrequency(), 1e-5);
        assertEquals(1.0, vocabCache.wordFor(SequenceElement.getLongHash("two")).getElementFrequency(), 1e-5);
    }

    public static class TestFn implements VoidFunction<ExportContainer<VocabWord>>, Serializable {
        @Override
        public void call(ExportContainer<VocabWord> v) throws Exception {
            assertNotNull(v.getElement());
            assertNotNull(v.getArray());
//            System.out.println(v.getElement() + " - " + v.getArray());
        }
    }
}
