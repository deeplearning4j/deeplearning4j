/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.deeplearning4j.models.embeddings.loader;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.FlatModelUtils;
import org.deeplearning4j.models.fasttext.FastText;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.junit.Before;
import org.junit.Rule;
import org.junit.Test;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.IOException;
import java.util.Collections;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertNotNull;
import static org.junit.Assert.assertTrue;
import static org.junit.Assert.fail;

@Slf4j
public class WordVectorSerializerTest extends BaseDL4JTest {
    private AbstractCache<VocabWord> cache;

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    @Before
    public void setUp() throws Exception {
        cache = new AbstractCache.Builder<VocabWord>().build();

        val words = new VocabWord[3];
        words[0] = new VocabWord(1.0, "word");
        words[1] = new VocabWord(2.0, "test");
        words[2] = new VocabWord(3.0, "tester");

        for (int i = 0; i < words.length; ++i) {
            cache.addToken(words[i]);
            cache.addWordToIndex(i, words[i].getLabel());
        }
    }

    @Test
    public void sequenceVectorsCorrect_WhenDeserialized() {

        INDArray syn0 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1Neg = Nd4j.rand(DataType.FLOAT, 10, 2);

        InMemoryLookupTable<VocabWord> lookupTable = new InMemoryLookupTable
                .Builder<VocabWord>()
                .useAdaGrad(false)
                .cache(cache)
                .build();

        lookupTable.setSyn0(syn0);
        lookupTable.setSyn1(syn1);
        lookupTable.setSyn1Neg(syn1Neg);

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration()).
                vocabCache(cache).
                lookupTable(lookupTable).
                build();
        SequenceVectors<VocabWord> deser = null;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            WordVectorSerializer.writeSequenceVectors(vectors, baos);
            byte[] bytesResult = baos.toByteArray();
            deser = WordVectorSerializer.readSequenceVectors(new ByteArrayInputStream(bytesResult), true);
        } catch (Exception e) {
            log.error("",e);
            fail();
        }

        assertNotNull(vectors.getConfiguration());
        assertEquals(vectors.getConfiguration(), deser.getConfiguration());

        assertEquals(cache.totalWordOccurrences(),deser.vocab().totalWordOccurrences());
        assertEquals(cache.totalNumberOfDocs(), deser.vocab().totalNumberOfDocs());
        assertEquals(cache.numWords(), deser.vocab().numWords());

        for (int i = 0; i < cache.words().size(); ++i) {
            val cached = cache.wordAtIndex(i);
            val restored = deser.vocab().wordAtIndex(i);
            assertNotNull(cached);
            assertEquals(cached, restored);
        }

    }

    @Test
    public void W2V_Correct_WhenDeserialized() {

        INDArray syn0 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1Neg = Nd4j.rand(DataType.FLOAT, 10, 2);

        InMemoryLookupTable<VocabWord> lookupTable = new InMemoryLookupTable
                .Builder<VocabWord>()
                .useAdaGrad(false)
                .cache(cache)
                .build();

        lookupTable.setSyn0(syn0);
        lookupTable.setSyn1(syn1);
        lookupTable.setSyn1Neg(syn1Neg);

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration()).
                vocabCache(cache).
                lookupTable(lookupTable).
                layerSize(200).
                modelUtils(new BasicModelUtils<VocabWord>()).
                build();

        Word2Vec word2Vec = new Word2Vec.Builder(vectors.getConfiguration())
                .vocabCache(vectors.vocab())
                .lookupTable(lookupTable)
                .modelUtils(new FlatModelUtils<VocabWord>())
                .limitVocabularySize(1000)
                .elementsLearningAlgorithm(CBOW.class.getCanonicalName())
                .allowParallelTokenization(true)
                .usePreciseMode(true)
                .batchSize(1024)
                .windowSize(23)
                .minWordFrequency(24)
                .iterations(54)
                .seed(45)
                .learningRate(0.08)
                .epochs(45)
                .stopWords(Collections.singletonList("NOT"))
                .sampling(44)
                .workers(45)
                .negativeSample(56)
                .useAdaGrad(true)
                .useHierarchicSoftmax(false)
                .minLearningRate(0.002)
                .resetModel(true)
                .useUnknown(true)
                .enableScavenger(true)
                .usePreciseWeightInit(true)
                .build();

        Word2Vec deser = null;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            WordVectorSerializer.writeWord2Vec(word2Vec, baos);
            byte[] bytesResult = baos.toByteArray();
            deser = WordVectorSerializer.readWord2Vec(new ByteArrayInputStream(bytesResult), true);
        } catch (Exception e) {
            log.error("",e);
            fail();
        }

        assertNotNull(word2Vec.getConfiguration());
        assertEquals(word2Vec.getConfiguration(), deser.getConfiguration());

        assertEquals(cache.totalWordOccurrences(),deser.vocab().totalWordOccurrences());
        assertEquals(cache.totalNumberOfDocs(), deser.vocab().totalNumberOfDocs());
        assertEquals(cache.numWords(), deser.vocab().numWords());

        for (int i = 0; i < cache.words().size(); ++i) {
            val cached = cache.wordAtIndex(i);
            val restored = deser.vocab().wordAtIndex(i);
            assertNotNull(cached);
            assertEquals(cached, restored);
        }

    }

    @Test
    public void ParaVec_Correct_WhenDeserialized() {

        INDArray syn0 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1Neg = Nd4j.rand(DataType.FLOAT, 10, 2);

        InMemoryLookupTable<VocabWord> lookupTable = new InMemoryLookupTable
                .Builder<VocabWord>()
                .useAdaGrad(false)
                .cache(cache)
                .build();

        lookupTable.setSyn0(syn0);
        lookupTable.setSyn1(syn1);
        lookupTable.setSyn1Neg(syn1Neg);

        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .vocabCache(cache)
                .lookupTable(lookupTable)
                .build();

        Word2Vec deser = null;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            WordVectorSerializer.writeWord2Vec(paragraphVectors, baos);
            byte[] bytesResult = baos.toByteArray();
            deser = WordVectorSerializer.readWord2Vec(new ByteArrayInputStream(bytesResult), true);
        } catch (Exception e) {
            log.error("",e);
            fail();
        }

        assertNotNull(paragraphVectors.getConfiguration());
        assertEquals(paragraphVectors.getConfiguration(), deser.getConfiguration());

        assertEquals(cache.totalWordOccurrences(),deser.vocab().totalWordOccurrences());
        assertEquals(cache.totalNumberOfDocs(), deser.vocab().totalNumberOfDocs());
        assertEquals(cache.numWords(), deser.vocab().numWords());

        for (int i = 0; i < cache.words().size(); ++i) {
            val cached = cache.wordAtIndex(i);
            val restored = deser.vocab().wordAtIndex(i);
            assertNotNull(cached);
            assertEquals(cached, restored);
        }

    }

    @Test
    public void weightLookupTable_Correct_WhenDeserialized() throws Exception {

        INDArray syn0 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1Neg = Nd4j.rand(DataType.FLOAT, 10, 2);

        InMemoryLookupTable<VocabWord> lookupTable = new InMemoryLookupTable
                .Builder<VocabWord>()
                .useAdaGrad(false)
                .cache(cache)
                .build();

        lookupTable.setSyn0(syn0);
        lookupTable.setSyn1(syn1);
        lookupTable.setSyn1Neg(syn1Neg);

        File dir = testDir.newFolder();
        File file = new File(dir, "lookupTable.txt");

        WeightLookupTable<VocabWord> deser = null;
        try {
            WordVectorSerializer.writeLookupTable(lookupTable, file);
            deser = WordVectorSerializer.readLookupTable(file);
        } catch (Exception e) {
            log.error("",e);
            fail();
        }
        assertEquals(lookupTable.getVocab().totalWordOccurrences(), ((InMemoryLookupTable<VocabWord>)deser).getVocab().totalWordOccurrences());
        assertEquals(cache.totalNumberOfDocs(), ((InMemoryLookupTable<VocabWord>)deser).getVocab().totalNumberOfDocs());
        assertEquals(cache.numWords(), ((InMemoryLookupTable<VocabWord>)deser).getVocab().numWords());

        for (int i = 0; i < cache.words().size(); ++i) {
            val cached = cache.wordAtIndex(i);
            val restored = ((InMemoryLookupTable<VocabWord>)deser).getVocab().wordAtIndex(i);
            assertNotNull(cached);
            assertEquals(cached, restored);
        }

        assertEquals(lookupTable.getSyn0().columns(), ((InMemoryLookupTable<VocabWord>) deser).getSyn0().columns());
        assertEquals(lookupTable.getSyn0().rows(), ((InMemoryLookupTable<VocabWord>) deser).getSyn0().rows());
        for (int c = 0; c < ((InMemoryLookupTable<VocabWord>) deser).getSyn0().columns(); ++c) {
            for (int r = 0; r < ((InMemoryLookupTable<VocabWord>) deser).getSyn0().rows(); ++r) {
                assertEquals(lookupTable.getSyn0().getDouble(r,c),
                            ((InMemoryLookupTable<VocabWord>) deser).getSyn0().getDouble(r,c), 1e-5);
            }
        }
    }

    @Test
    public void FastText_Correct_WhenDeserialized() throws IOException {

        FastText fastText =
                FastText.builder().cbow(true).build();

        File dir = testDir.newFolder();
        WordVectorSerializer.writeWordVectors(fastText, new File(dir, "some.data"));

        FastText deser = null;
        try {
            deser = WordVectorSerializer.readWordVectors(new File(dir, "some.data"));
        } catch (Exception e) {
            log.error("",e);
            fail();
        }

        assertNotNull(deser);
        assertEquals(fastText.isCbow(), deser.isCbow());
        assertEquals(fastText.isModelLoaded(), deser.isModelLoaded());
        assertEquals(fastText.isAnalogies(), deser.isAnalogies());
        assertEquals(fastText.isNn(), deser.isNn());
        assertEquals(fastText.isPredict(), deser.isPredict());
        assertEquals(fastText.isPredict_prob(), deser.isPredict_prob());
        assertEquals(fastText.isQuantize(), deser.isQuantize());
        assertEquals(fastText.getInputFile(), deser.getInputFile());
        assertEquals(fastText.getOutputFile(), deser.getOutputFile());
    }

    @Test
    public void testIsHeader_withValidHeader () {

        /* Given */
        AbstractCache<VocabWord> cache = new AbstractCache<>();
        String line = "48 100";

        /* When */
        boolean isHeader = WordVectorSerializer.isHeader(line, cache);

        /* Then */
        assertTrue(isHeader);
    }

    @Test
    public void testIsHeader_notHeader () {

        /* Given */
        AbstractCache<VocabWord> cache = new AbstractCache<>();
        String line = "your -0.0017603 0.0030831 0.00069072 0.0020581 -0.0050952 -2.2573e-05 -0.001141";

        /* When */
        boolean isHeader = WordVectorSerializer.isHeader(line, cache);

        /* Then */
        assertFalse(isHeader);
    }
}
