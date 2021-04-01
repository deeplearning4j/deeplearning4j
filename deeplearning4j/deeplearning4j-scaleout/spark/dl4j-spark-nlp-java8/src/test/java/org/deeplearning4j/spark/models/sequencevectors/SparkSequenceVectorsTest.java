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

package org.deeplearning4j.spark.models.sequencevectors;

import com.sun.jna.Platform;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.sequence.ShallowSequenceElement;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.spark.models.sequencevectors.export.ExportContainer;
import org.deeplearning4j.spark.models.sequencevectors.export.SparkModelExporter;
import org.deeplearning4j.spark.models.word2vec.SparkWord2VecTest;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.junit.jupiter.api.*;
import org.nd4j.common.primitives.Counter;
import org.nd4j.common.resources.Downloader;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;

import java.io.File;
import java.net.URI;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNotEquals;
@Tag(TagNames.FILE_IO)
@Tag(TagNames.SPARK)
@Tag(TagNames.DIST_SYSTEMS)
@NativeTag
@Slf4j
public class SparkSequenceVectorsTest extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 120000L;
    }

    protected static List<Sequence<VocabWord>> sequencesCyclic;
    private JavaSparkContext sc;


    @BeforeAll
    @SneakyThrows
    public static void beforeAll() {
        if(Platform.isWindows()) {
            File hadoopHome = new File(System.getProperty("java.io.tmpdir"),"hadoop-tmp");
            File binDir = new File(hadoopHome,"bin");
            if(!binDir.exists())
                binDir.mkdirs();
            File outputFile = new File(binDir,"winutils.exe");
            if(!outputFile.exists()) {
                log.info("Fixing spark for windows");
                Downloader.download("winutils.exe",
                        URI.create("https://github.com/cdarlint/winutils/blob/master/hadoop-2.6.5/bin/winutils.exe?raw=true").toURL(),
                        outputFile,"db24b404d2331a1bec7443336a5171f1",3);
            }

            System.setProperty("hadoop.home.dir", hadoopHome.getAbsolutePath());
        }
    }

    @BeforeEach
    public void setUp() throws Exception {
        if (sequencesCyclic == null) {
            sequencesCyclic = new ArrayList<>();

            // 10 sequences in total
            for (int с = 0; с < 10; с++) {

                Sequence<VocabWord> sequence = new Sequence<>();

                for (int e = 0; e < 10; e++) {
                    // we will have 9 equal elements, with total frequency of 10
                    sequence.addElement(new VocabWord(1.0, "" + e, (long) e));
                }

                // and 1 element with frequency of 20
                sequence.addElement(new VocabWord(1.0, "0", 0L));
                sequencesCyclic.add(sequence);
            }
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
    @Disabled("Timeout issue")
    public void testFrequenciesCount() throws Exception {

        if(Platform.isWindows()) {
            //Spark tests don't run on windows
            return;
        }
        JavaRDD<Sequence<VocabWord>> sequences = sc.parallelize(sequencesCyclic);

        SparkSequenceVectors<VocabWord> seqVec = new SparkSequenceVectors<>();

        seqVec.getConfiguration().setTokenizerFactory(DefaultTokenizerFactory.class.getCanonicalName());
        seqVec.getConfiguration().setElementsLearningAlgorithm("org.deeplearning4j.spark.models.sequencevectors.learning.elements.SparkSkipGram");
        seqVec.setExporter(new SparkModelExporter<VocabWord>() {
            @Override
            public void export(JavaRDD<ExportContainer<VocabWord>> rdd) {
                rdd.foreach(new SparkWord2VecTest.TestFn());
            }
        });

        seqVec.fitSequences(sequences);

        Counter<Long> counter = seqVec.getCounter();

        // element "0" should have frequency of 20
        assertEquals(20, counter.getCount(0L), 1e-5);

        // elements 1 - 9 should have frequencies of 10
        for (int e = 1; e < sequencesCyclic.get(0).getElements().size() - 1; e++) {
            assertEquals(10, counter.getCount(sequencesCyclic.get(0).getElementByIndex(e).getStorageId()), 1e-5);
        }


        VocabCache<ShallowSequenceElement> shallowVocab = seqVec.getShallowVocabCache();

        assertEquals(10, shallowVocab.numWords());

        ShallowSequenceElement zero = shallowVocab.tokenFor(0L);
        ShallowSequenceElement first = shallowVocab.tokenFor(1L);

        assertNotEquals(null, zero);
        assertEquals(20.0, zero.getElementFrequency(), 1e-5);
        assertEquals(0, zero.getIndex());

        assertEquals(10.0, first.getElementFrequency(), 1e-5);
    }

}
