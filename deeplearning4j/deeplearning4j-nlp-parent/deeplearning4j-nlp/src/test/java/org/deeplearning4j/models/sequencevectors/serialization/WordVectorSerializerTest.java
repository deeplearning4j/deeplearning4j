package org.deeplearning4j.models.sequencevectors.serialization;

import lombok.val;
import org.apache.commons.lang.StringUtils;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.FlatModelUtils;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.sequencevectors.SequenceVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.junit.Before;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.ByteArrayInputStream;
import java.io.ByteArrayOutputStream;
import java.io.File;
import java.util.Collections;

import static org.junit.Assert.*;

public class WordVectorSerializerTest {
    private AbstractCache<VocabWord> cache;

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

        InMemoryLookupTable<VocabWord> lookupTable =
                (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                        .useAdaGrad(false).cache(cache)
                        .build();

        lookupTable.setSyn0(syn0);
        lookupTable.setSyn1(syn1);
        lookupTable.setSyn1Neg(syn1Neg);

        SequenceVectors<VocabWord> vectors = new SequenceVectors.Builder<VocabWord>(new VectorsConfiguration()).
                vocabCache(cache).
                lookupTable(lookupTable).
                build();
        SequenceVectors<VocabWord> deser = null;
        String json = StringUtils.EMPTY;
        try {
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            WordVectorSerializer.writeSequenceVectors(vectors, baos);
            byte[] bytesResult = baos.toByteArray();
            deser = WordVectorSerializer.readSequenceVectors(new ByteArrayInputStream(bytesResult), true);
        } catch (Exception e) {
            e.printStackTrace();
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

        InMemoryLookupTable<VocabWord> lookupTable =
                (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                        .useAdaGrad(false).cache(cache)
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
            e.printStackTrace();
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

        InMemoryLookupTable<VocabWord> lookupTable =
                (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                        .useAdaGrad(false).cache(cache)
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
            e.printStackTrace();
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
    public void weightLookupTable_Correct_WhenDeserialized() {

        INDArray syn0 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1 = Nd4j.rand(DataType.FLOAT, 10, 2),
                syn1Neg = Nd4j.rand(DataType.FLOAT, 10, 2);

        InMemoryLookupTable<VocabWord> lookupTable =
                (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                        .useAdaGrad(false).cache(cache)
                        .build();

        lookupTable.setSyn0(syn0);
        lookupTable.setSyn1(syn1);
        lookupTable.setSyn1Neg(syn1Neg);

        File file = new File("lookupTable.txt");

        WeightLookupTable<VocabWord> deser = null;
        try {
            WordVectorSerializer.writeLookupTable(lookupTable, file);
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            deser = WordVectorSerializer.readLookupTable(file);
        } catch (Exception e) {
            e.printStackTrace();
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
                assertEquals(lookupTable.getSyn0().getDouble(c,r),
                            ((InMemoryLookupTable<VocabWord>) deser).getSyn0().getDouble(c,r), 1e-5);
            }
        }
    }
}
