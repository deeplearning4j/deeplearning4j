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

package org.deeplearning4j.models;

import com.google.common.primitives.Doubles;
import org.apache.commons.io.FileUtils;
import org.apache.commons.lang.ArrayUtils;
import org.junit.Rule;
import org.junit.rules.TemporaryFolder;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.embeddings.WeightLookupTable;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.RandomUtils;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.paragraphvectors.ParagraphVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.*;

/**
 * @author jeffreytang
 * @author raver119@gmail.com
 */
public class WordVectorSerializerTest {

    @Rule
    public TemporaryFolder testDir = new TemporaryFolder();

    private File textFile, binaryFile, textFile2;
    String pathToWriteto;

    private Logger logger = LoggerFactory.getLogger(WordVectorSerializerTest.class);

    @Before
    public void before() throws Exception {
        if (textFile == null) {
            textFile = new ClassPathResource("word2vecserialization/google_news_30.txt").getFile();
        }
        if (binaryFile == null) {
            File dir = testDir.newFolder();
            binaryFile = new File(dir, "google_news_30.bin.gz");
            FileUtils.copyFile(new ClassPathResource("word2vecserialization/google_news_30.bin.gz").getFile(), binaryFile);
        }
        pathToWriteto = new ClassPathResource("word2vecserialization/testing_word2vec_serialization.txt").getFile()
                        .getAbsolutePath();
        FileUtils.deleteDirectory(new File("word2vec-index"));
    }

    @Test
    @Ignore
    public void testLoaderTextSmall() throws Exception {
        INDArray vec = Nd4j.create(new double[] {0.002001, 0.002210, -0.001915, -0.001639, 0.000683, 0.001511, 0.000470,
                        0.000106, -0.001802, 0.001109, -0.002178, 0.000625, -0.000376, -0.000479, -0.001658, -0.000941,
                        0.001290, 0.001513, 0.001485, 0.000799, 0.000772, -0.001901, -0.002048, 0.002485, 0.001901,
                        0.001545, -0.000302, 0.002008, -0.000247, 0.000367, -0.000075, -0.001492, 0.000656, -0.000669,
                        -0.001913, 0.002377, 0.002190, -0.000548, -0.000113, 0.000255, -0.001819, -0.002004, 0.002277,
                        0.000032, -0.001291, -0.001521, -0.001538, 0.000848, 0.000101, 0.000666, -0.002107, -0.001904,
                        -0.000065, 0.000572, 0.001275, -0.001585, 0.002040, 0.000463, 0.000560, -0.000304, 0.001493,
                        -0.001144, -0.001049, 0.001079, -0.000377, 0.000515, 0.000902, -0.002044, -0.000992, 0.001457,
                        0.002116, 0.001966, -0.001523, -0.001054, -0.000455, 0.001001, -0.001894, 0.001499, 0.001394,
                        -0.000799, -0.000776, -0.001119, 0.002114, 0.001956, -0.000590, 0.002107, 0.002410, 0.000908,
                        0.002491, -0.001556, -0.000766, -0.001054, -0.001454, 0.001407, 0.000790, 0.000212, -0.001097,
                        0.000762, 0.001530, 0.000097, 0.001140, -0.002476, 0.002157, 0.000240, -0.000916, -0.001042,
                        -0.000374, -0.001468, -0.002185, -0.001419, 0.002139, -0.000885, -0.001340, 0.001159, -0.000852,
                        0.002378, -0.000802, -0.002294, 0.001358, -0.000037, -0.001744, 0.000488, 0.000721, -0.000241,
                        0.000912, -0.001979, 0.000441, 0.000908, -0.001505, 0.000071, -0.000030, -0.001200, -0.001416,
                        -0.002347, 0.000011, 0.000076, 0.000005, -0.001967, -0.002481, -0.002373, -0.002163, -0.000274,
                        0.000696, 0.000592, -0.001591, 0.002499, -0.001006, -0.000637, -0.000702, 0.002366, -0.001882,
                        0.000581, -0.000668, 0.001594, 0.000020, 0.002135, -0.001410, -0.001303, -0.002096, -0.001833,
                        -0.001600, -0.001557, 0.001222, -0.000933, 0.001340, 0.001845, 0.000678, 0.001475, 0.001238,
                        0.001170, -0.001775, -0.001717, -0.001828, -0.000066, 0.002065, -0.001368, -0.001530, -0.002098,
                        0.001653, -0.002089, -0.000290, 0.001089, -0.002309, -0.002239, 0.000721, 0.001762, 0.002132,
                        0.001073, 0.001581, -0.001564, -0.001820, 0.001987, -0.001382, 0.000877, 0.000287, 0.000895,
                        -0.000591, 0.000099, -0.000843, -0.000563});
        String w1 = "database";
        String w2 = "DBMS";
        WordVectors vecModel = WordVectorSerializer.readWord2VecModel(new ClassPathResource("word2vec/googleload/sample_vec.txt").getFile());
        WordVectors vectorsBinary = WordVectorSerializer.readWord2VecModel(new ClassPathResource("word2vec/googleload/sample_vec.bin").getFile());
        INDArray textWeights = vecModel.lookupTable().getWeights();
        INDArray binaryWeights = vectorsBinary.lookupTable().getWeights();
        Collection<String> nearest = vecModel.wordsNearest("database", 10);
        Collection<String> nearestBinary = vectorsBinary.wordsNearest("database", 10);
        System.out.println(nearestBinary);
        assertEquals(vecModel.similarity("DBMS", "DBMS's"), vectorsBinary.similarity("DBMS", "DBMS's"), 1e-1);

    }

    @Test
    public void testLoaderText() throws IOException {
        WordVectors vec = WordVectorSerializer.readWord2VecModel(textFile);
        assertEquals(vec.vocab().numWords(), 30);
        assertTrue(vec.vocab().hasToken("Morgan_Freeman"));
        assertTrue(vec.vocab().hasToken("JA_Montalbano"));
    }

    @Test
    public void testLoaderStream() throws IOException {
        WordVectors vec = WordVectorSerializer.readWord2VecModel(textFile);

        assertEquals(vec.vocab().numWords(), 30);
        assertTrue(vec.vocab().hasToken("Morgan_Freeman"));
        assertTrue(vec.vocab().hasToken("JA_Montalbano"));
    }

    @Test
    public void testLoaderBinary() throws IOException {
        WordVectors vec = WordVectorSerializer.readWord2VecModel(binaryFile);
        assertEquals(vec.vocab().numWords(), 30);
        assertTrue(vec.vocab().hasToken("Morgan_Freeman"));
        assertTrue(vec.vocab().hasToken("JA_Montalbano"));
        double[] wordVector1 = vec.getWordVector("Morgan_Freeman");
        double[] wordVector2 = vec.getWordVector("JA_Montalbano");
        assertTrue(wordVector1.length == 300);
        assertTrue(wordVector2.length == 300);
        assertEquals(Doubles.asList(wordVector1).get(0), 0.044423, 1e-3);
        assertEquals(Doubles.asList(wordVector2).get(0), 0.051964, 1e-3);
    }

    @Test
    @Ignore
    public void testWriteWordVectors() throws IOException {
        WordVectors vec = WordVectorSerializer.readWord2VecModel(binaryFile);
        InMemoryLookupTable lookupTable = (InMemoryLookupTable) vec.lookupTable();
        InMemoryLookupCache lookupCache = (InMemoryLookupCache) vec.vocab();
        WordVectorSerializer.writeWordVectors(lookupTable, lookupCache, pathToWriteto);

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(pathToWriteto));
        double[] wordVector1 = wordVectors.getWordVector("Morgan_Freeman");
        double[] wordVector2 = wordVectors.getWordVector("JA_Montalbano");
        assertTrue(wordVector1.length == 300);
        assertTrue(wordVector2.length == 300);
        assertEquals(Doubles.asList(wordVector1).get(0), 0.044423, 1e-3);
        assertEquals(Doubles.asList(wordVector2).get(0), 0.051964, 1e-3);
    }

    @Test
    @Ignore
    public void testWriteWordVectorsFromWord2Vec() throws IOException {
        WordVectors vec = WordVectorSerializer.readWord2VecModel(binaryFile, true);
        WordVectorSerializer.writeWordVectors((Word2Vec) vec, pathToWriteto);

        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(new File(pathToWriteto));
        INDArray wordVector1 = wordVectors.getWordVectorMatrix("Morgan_Freeman");
        INDArray wordVector2 = wordVectors.getWordVectorMatrix("JA_Montalbano");
        assertEquals(vec.getWordVectorMatrix("Morgan_Freeman"), wordVector1);
        assertEquals(vec.getWordVectorMatrix("JA_Montalbano"), wordVector2);
        assertTrue(wordVector1.length() == 300);
        assertTrue(wordVector2.length() == 300);
        assertEquals(wordVector1.getDouble(0), 0.044423, 1e-3);
        assertEquals(wordVector2.getDouble(0), 0.051964, 1e-3);
    }

    @Test
    @Ignore
    public void testFromTableAndVocab() throws IOException {

        WordVectors vec = WordVectorSerializer.readWord2VecModel(textFile);
        InMemoryLookupTable lookupTable = (InMemoryLookupTable) vec.lookupTable();
        InMemoryLookupCache lookupCache = (InMemoryLookupCache) vec.vocab();

        WordVectors wordVectors = WordVectorSerializer.fromTableAndVocab(lookupTable, lookupCache);
        double[] wordVector1 = wordVectors.getWordVector("Morgan_Freeman");
        double[] wordVector2 = wordVectors.getWordVector("JA_Montalbano");
        assertTrue(wordVector1.length == 300);
        assertTrue(wordVector2.length == 300);
        assertEquals(Doubles.asList(wordVector1).get(0), 0.044423, 1e-3);
        assertEquals(Doubles.asList(wordVector2).get(0), 0.051964, 1e-3);
    }

    @Test
    public void testIndexPersistence() throws Exception {
        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5).iterations(1).epochs(1).layerSize(100)
                        .stopWords(new ArrayList<String>()).useAdaGrad(false).negativeSample(5).seed(42).windowSize(5)
                        .iterate(iter).tokenizerFactory(t).build();

        vec.fit();

        VocabCache orig = vec.getVocab();

        File tempFile = File.createTempFile("temp", "w2v");
        tempFile.deleteOnExit();

        WordVectorSerializer.writeWordVectors(vec, tempFile);

        WordVectors vec2 = WordVectorSerializer.loadTxtVectors(tempFile);

        VocabCache rest = vec2.vocab();

        assertEquals(orig.totalNumberOfDocs(), rest.totalNumberOfDocs());

        for (VocabWord word : vec.getVocab().vocabWords()) {
            INDArray array1 = vec.getWordVectorMatrix(word.getLabel());
            INDArray array2 = vec2.getWordVectorMatrix(word.getLabel());

            assertEquals(array1, array2);
        }
    }

    @Test
    public void testFullModelSerialization() throws Exception {
        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        InMemoryLookupCache cache = new InMemoryLookupCache(false);
        WeightLookupTable table = new InMemoryLookupTable.Builder().vectorLength(100).useAdaGrad(false).negative(5.0)
                        .cache(cache).lr(0.025f).build();

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5).iterations(1).epochs(1).layerSize(100)
                        .lookupTable(table).stopWords(new ArrayList<String>()).useAdaGrad(false).negativeSample(5)
                        .vocabCache(cache).seed(42)
                        //                .workers(6)
                        .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());
        vec.fit();

        //logger.info("Original word 0: " + cache.wordFor(cache.wordAtIndex(0)));

        //logger.info("Closest Words:");
        Collection<String> lst = vec.wordsNearest("day", 10);
        System.out.println(lst);

        WordVectorSerializer.writeFullModel(vec, "tempModel.txt");

        File modelFile = new File("tempModel.txt");
        modelFile.deleteOnExit();

        assertTrue(modelFile.exists());
        assertTrue(modelFile.length() > 0);


        Word2Vec vec2 = WordVectorSerializer.loadFullModel("tempModel.txt");

        assertNotEquals(null, vec2);

        assertEquals(vec.getConfiguration(), vec2.getConfiguration());

        //logger.info("Source ExpTable: " + ArrayUtils.toString(((InMemoryLookupTable) table).getExpTable()));
        //logger.info("Dest  ExpTable: " + ArrayUtils.toString(((InMemoryLookupTable)  vec2.getLookupTable()).getExpTable()));
        assertTrue(ArrayUtils.isEquals(((InMemoryLookupTable) table).getExpTable(),
                        ((InMemoryLookupTable) vec2.getLookupTable()).getExpTable()));


        InMemoryLookupTable restoredTable = (InMemoryLookupTable) vec2.lookupTable();

        /*
        logger.info("Restored word 1: " + restoredTable.getVocab().wordFor(restoredTable.getVocab().wordAtIndex(1)));
        logger.info("Restored word 'it': " + restoredTable.getVocab().wordFor("it"));
        logger.info("Original word 1: " + cache.wordFor(cache.wordAtIndex(1)));
        logger.info("Original word 'i': " + cache.wordFor("i"));
        logger.info("Original word 0: " + cache.wordFor(cache.wordAtIndex(0)));
        logger.info("Restored word 0: " + restoredTable.getVocab().wordFor(restoredTable.getVocab().wordAtIndex(0)));
        */
        assertEquals(cache.wordAtIndex(1), restoredTable.getVocab().wordAtIndex(1));
        assertEquals(cache.wordAtIndex(7), restoredTable.getVocab().wordAtIndex(7));
        assertEquals(cache.wordAtIndex(15), restoredTable.getVocab().wordAtIndex(15));

        /*
            these tests needed only to make sure INDArray equality is working properly
         */
        double[] array1 = new double[] {0.323232325, 0.65756575, 0.12315, 0.12312315, 0.1232135, 0.12312315,
                        0.4343423425, 0.15};
        double[] array2 = new double[] {0.423232325, 0.25756575, 0.12375, 0.12311315, 0.1232035, 0.12318315,
                        0.4343493425, 0.25};
        assertNotEquals(Nd4j.create(array1), Nd4j.create(array2));
        assertEquals(Nd4j.create(array1), Nd4j.create(array1));


        INDArray rSyn0_1 = restoredTable.getSyn0().slice(1);
        INDArray oSyn0_1 = ((InMemoryLookupTable) table).getSyn0().slice(1);

        //logger.info("Restored syn0: " + rSyn0_1);
        //logger.info("Original syn0: " + oSyn0_1);

        assertEquals(oSyn0_1, rSyn0_1);

        // just checking $^###! syn0/syn1 order
        int cnt = 0;
        for (VocabWord word : cache.vocabWords()) {
            INDArray rSyn0 = restoredTable.getSyn0().slice(word.getIndex());
            INDArray oSyn0 = ((InMemoryLookupTable) table).getSyn0().slice(word.getIndex());

            assertEquals(rSyn0, oSyn0);
            assertEquals(1.0, arraysSimilarity(rSyn0, oSyn0), 0.001);

            INDArray rSyn1 = restoredTable.getSyn1().slice(word.getIndex());
            INDArray oSyn1 = ((InMemoryLookupTable) table).getSyn1().slice(word.getIndex());

            assertEquals(rSyn1, oSyn1);
            if (arraysSimilarity(rSyn1, oSyn1) < 0.98) {
                //   logger.info("Restored syn1: " + rSyn1);
                //   logger.info("Original  syn1: " + oSyn1);
            }
            // we exclude word 222 since it has syn1 full of zeroes
            if (cnt != 222)
                assertEquals(1.0, arraysSimilarity(rSyn1, oSyn1), 0.001);



            if (((InMemoryLookupTable) table).getSyn1Neg() != null) {
                INDArray rSyn1Neg = restoredTable.getSyn1Neg().slice(word.getIndex());
                INDArray oSyn1Neg = ((InMemoryLookupTable) table).getSyn1Neg().slice(word.getIndex());

                assertEquals(rSyn1Neg, oSyn1Neg);
                //                assertEquals(1.0, arraysSimilarity(rSyn1Neg, oSyn1Neg), 0.001);
            }
            assertEquals(word.getHistoricalGradient(),
                            restoredTable.getVocab().wordFor(word.getWord()).getHistoricalGradient());

            cnt++;
        }

        // at this moment we can assume that whole model is transferred, and we can call fit over new model
        //        iter.reset();

        iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());

        vec2.setTokenizerFactory(t);
        vec2.setSentenceIterator(iter);

        vec2.fit();

        INDArray day1 = vec.getWordVectorMatrix("day");
        INDArray day2 = vec2.getWordVectorMatrix("day");

        INDArray night1 = vec.getWordVectorMatrix("night");
        INDArray night2 = vec2.getWordVectorMatrix("night");

        double simD = arraysSimilarity(day1, day2);
        double simN = arraysSimilarity(night1, night2);

        logger.info("Vec1 day: " + day1);
        logger.info("Vec2 day: " + day2);

        logger.info("Vec1 night: " + night1);
        logger.info("Vec2 night: " + night2);

        logger.info("Day/day cross-model similarity: " + simD);
        logger.info("Night/night cross-model similarity: " + simN);



        logger.info("Vec1 day/night similiraty: " + vec.similarity("day", "night"));
        logger.info("Vec2 day/night similiraty: " + vec2.similarity("day", "night"));

        // check if cross-model values are not the same
        assertNotEquals(1.0, simD, 0.001);
        assertNotEquals(1.0, simN, 0.001);

        // check if cross-model values are still close to each other
        assertTrue(simD > 0.70);
        assertTrue(simN > 0.70);

        modelFile.delete();
    }

    @Test
    @Ignore
    public void testLoader() throws Exception {
        WordVectors vec = WordVectorSerializer.loadTxtVectors(new File("/home/raver119/Downloads/_vectors.txt"));

        logger.info("Rewinding: " + Arrays.toString(vec.getWordVector("rewinding")));
    }


    @Test
    public void testOutputStream() throws Exception {
        File file = File.createTempFile("tmp_ser", "ssa");
        file.deleteOnExit();

        File inputFile = new ClassPathResource("/big/raw_sentences.txt").getFile();
        SentenceIterator iter = new BasicLineIterator(inputFile);
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        InMemoryLookupCache cache = new InMemoryLookupCache(false);
        WeightLookupTable table = new InMemoryLookupTable.Builder().vectorLength(100).useAdaGrad(false).negative(5.0)
                        .cache(cache).lr(0.025f).build();

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5).iterations(1).epochs(1).layerSize(100)
                        .lookupTable(table).stopWords(new ArrayList<String>()).useAdaGrad(false).negativeSample(5)
                        .vocabCache(cache).seed(42)
                        //                .workers(6)
                        .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());
        vec.fit();

        INDArray day1 = vec.getWordVectorMatrix("day");

        WordVectorSerializer.writeWordVectors(vec, new FileOutputStream(file));

        WordVectors vec2 = WordVectorSerializer.loadTxtVectors(file);

        INDArray day2 = vec2.getWordVectorMatrix("day");

        assertEquals(day1, day2);

        File tempFile = File.createTempFile("tetsts", "Fdfs");
        tempFile.deleteOnExit();

        WordVectorSerializer.writeWord2VecModel(vec, tempFile);

        Word2Vec vec3 = WordVectorSerializer.readWord2VecModel(tempFile);
    }

    @Test
    public void testParaVecSerialization1() throws Exception {
        VectorsConfiguration configuration = new VectorsConfiguration();
        configuration.setIterations(14123);
        configuration.setLayersSize(156);

        INDArray syn0 = Nd4j.rand(100, configuration.getLayersSize());
        INDArray syn1 = Nd4j.rand(100, configuration.getLayersSize());

        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        for (int i = 0; i < 100; i++) {
            VocabWord word = new VocabWord((float) i, "word_" + i);
            List<Integer> points = new ArrayList<>();
            List<Byte> codes = new ArrayList<>();
            int num = org.apache.commons.lang3.RandomUtils.nextInt(1, 20);
            for (int x = 0; x < num; x++) {
                points.add(org.apache.commons.lang3.RandomUtils.nextInt(1, 100000));
                codes.add(org.apache.commons.lang3.RandomUtils.nextBytes(10)[0]);
            }
            if (RandomUtils.nextInt(10) < 3) {
                word.markAsLabel(true);
            }
            word.setIndex(i);
            word.setPoints(points);
            word.setCodes(codes);
            cache.addToken(word);
            cache.addWordToIndex(i, word.getLabel());
        }

        InMemoryLookupTable<VocabWord> lookupTable =
                        (InMemoryLookupTable<VocabWord>) new InMemoryLookupTable.Builder<VocabWord>()
                                        .vectorLength(configuration.getLayersSize()).cache(cache).build();

        lookupTable.setSyn0(syn0);
        lookupTable.setSyn1(syn1);

        ParagraphVectors originalVectors =
                        new ParagraphVectors.Builder(configuration).vocabCache(cache).lookupTable(lookupTable).build();

        File tempFile = File.createTempFile("paravec", "tests");
        tempFile.deleteOnExit();

        WordVectorSerializer.writeParagraphVectors(originalVectors, tempFile);

        ParagraphVectors restoredVectors = WordVectorSerializer.readParagraphVectors(tempFile);

        InMemoryLookupTable<VocabWord> restoredLookupTable =
                        (InMemoryLookupTable<VocabWord>) restoredVectors.getLookupTable();
        AbstractCache<VocabWord> restoredVocab = (AbstractCache<VocabWord>) restoredVectors.getVocab();

        assertEquals(restoredLookupTable.getSyn0(), lookupTable.getSyn0());
        assertEquals(restoredLookupTable.getSyn1(), lookupTable.getSyn1());

        for (int i = 0; i < cache.numWords(); i++) {
            assertEquals(cache.elementAtIndex(i).isLabel(), restoredVocab.elementAtIndex(i).isLabel());
            assertEquals(cache.wordAtIndex(i), restoredVocab.wordAtIndex(i));
            assertEquals(cache.elementAtIndex(i).getElementFrequency(),
                            restoredVocab.elementAtIndex(i).getElementFrequency(), 0.1f);
            List<Integer> originalPoints = cache.elementAtIndex(i).getPoints();
            List<Integer> restoredPoints = restoredVocab.elementAtIndex(i).getPoints();
            assertEquals(originalPoints.size(), restoredPoints.size());
            for (int x = 0; x < originalPoints.size(); x++) {
                assertEquals(originalPoints.get(x), restoredPoints.get(x));
            }

            List<Byte> originalCodes = cache.elementAtIndex(i).getCodes();
            List<Byte> restoredCodes = restoredVocab.elementAtIndex(i).getCodes();
            assertEquals(originalCodes.size(), restoredCodes.size());
            for (int x = 0; x < originalCodes.size(); x++) {
                assertEquals(originalCodes.get(x), restoredCodes.get(x));
            }
        }
    }

    private double arraysSimilarity(INDArray array1, INDArray array2) {
        if (array1.equals(array2))
            return 1.0;

        INDArray vector = Transforms.unitVec(array1);
        INDArray vector2 = Transforms.unitVec(array2);
        if (vector == null || vector2 == null)
            return -1;
        return Nd4j.getBlasWrapper().dot(vector, vector2);

    }


    /**
     * This method here is only to test real google model few gigabytes worth
     * Keep it ignored, since it requirs full google model being present in system, which is 1.6gb compressed
     *
     * @throws Exception
     */
    @Test
    @Ignore
    public void testStaticLoaderGoogleModel() throws Exception {
        logger.info("Executor name: {}", Nd4j.getExecutioner().getClass().getSimpleName());

        long time1 = System.currentTimeMillis();
        WordVectors vectors = WordVectorSerializer
                        .loadStaticModel(new File("C:\\Users\\raver\\develop\\GoogleNews-vectors-negative300.bin.gz"));
        long time2 = System.currentTimeMillis();

        logger.info("Loading time: {} ms", (time2 - time1));
    }

    /**
     * This method tests binary file loading as static model
     *
     * @throws Exception
     */
    @Test
    public void testStaticLoaderBinary() throws Exception {

        logger.info("Executor name: {}", Nd4j.getExecutioner().getClass().getSimpleName());

        WordVectors vectorsLive = WordVectorSerializer.readWord2VecModel(binaryFile);
        WordVectors vectorsStatic = WordVectorSerializer.loadStaticModel(binaryFile);

        INDArray arrayLive = vectorsLive.getWordVectorMatrix("Morgan_Freeman");
        INDArray arrayStatic = vectorsStatic.getWordVectorMatrix("Morgan_Freeman");

        assertNotEquals(null, arrayLive);
        assertEquals(arrayLive, arrayStatic);
    }

    /**
     * This method tests CSV file loading as static model
     *
     * @throws Exception
     */
    @Test
    public void testStaticLoaderText() throws Exception {
        logger.info("Executor name: {}", Nd4j.getExecutioner().getClass().getSimpleName());

        WordVectors vectorsLive = WordVectorSerializer.loadTxtVectors(textFile);
        WordVectors vectorsStatic = WordVectorSerializer.loadStaticModel(textFile);

        INDArray arrayLive = vectorsLive.getWordVectorMatrix("Morgan_Freeman");
        INDArray arrayStatic = vectorsStatic.getWordVectorMatrix("Morgan_Freeman");

        assertNotEquals(null, arrayLive);
        assertEquals(arrayLive, arrayStatic);
    }

    /**
     * This method tests ZIP file loading as static model
     *
     * @throws Exception
     */
    @Test
    public void testStaticLoaderArchive() throws Exception {
        logger.info("Executor name: {}", Nd4j.getExecutioner().getClass().getSimpleName());

        File w2v = new ClassPathResource("word2vec.dl4j/file.w2v").getFile();

        WordVectors vectorsLive = WordVectorSerializer.readWord2Vec(w2v);
        WordVectors vectorsStatic = WordVectorSerializer.loadStaticModel(w2v);

        INDArray arrayLive = vectorsLive.getWordVectorMatrix("night");
        INDArray arrayStatic = vectorsStatic.getWordVectorMatrix("night");

        assertNotEquals(null, arrayLive);
        assertEquals(arrayLive, arrayStatic);
    }

    @Test
    public void testUnifiedLoaderArchive1() throws Exception {
        logger.info("Executor name: {}", Nd4j.getExecutioner().getClass().getSimpleName());

        File w2v = new ClassPathResource("word2vec.dl4j/file.w2v").getFile();

        WordVectors vectorsLive = WordVectorSerializer.readWord2Vec(w2v);
        WordVectors vectorsUnified = WordVectorSerializer.readWord2VecModel(w2v, false);

        INDArray arrayLive = vectorsLive.getWordVectorMatrix("night");
        INDArray arrayStatic = vectorsUnified.getWordVectorMatrix("night");

        assertNotEquals(null, arrayLive);
        assertEquals(arrayLive, arrayStatic);

        assertEquals(null, ((InMemoryLookupTable) vectorsUnified.lookupTable()).getSyn1());
        assertEquals(null, ((InMemoryLookupTable) vectorsUnified.lookupTable()).getSyn1Neg());
    }

    @Test
    public void testUnifiedLoaderArchive2() throws Exception {
        logger.info("Executor name: {}", Nd4j.getExecutioner().getClass().getSimpleName());

        File w2v = new ClassPathResource("word2vec.dl4j/file.w2v").getFile();

        WordVectors vectorsLive = WordVectorSerializer.readWord2Vec(w2v);
        WordVectors vectorsUnified = WordVectorSerializer.readWord2VecModel(w2v, true);

        INDArray arrayLive = vectorsLive.getWordVectorMatrix("night");
        INDArray arrayStatic = vectorsUnified.getWordVectorMatrix("night");

        assertNotEquals(null, arrayLive);
        assertEquals(arrayLive, arrayStatic);

        assertNotEquals(null, ((InMemoryLookupTable) vectorsUnified.lookupTable()).getSyn1());
    }

    /**
     * This method tests CSV file loading via unified loader
     *
     * @throws Exception
     */
    @Test
    public void testUnifiedLoaderText() throws Exception {
        logger.info("Executor name: {}", Nd4j.getExecutioner().getClass().getSimpleName());

        WordVectors vectorsLive = WordVectorSerializer.loadTxtVectors(textFile);
        WordVectors vectorsUnified = WordVectorSerializer.readWord2VecModel(textFile, true);

        INDArray arrayLive = vectorsLive.getWordVectorMatrix("Morgan_Freeman");
        INDArray arrayStatic = vectorsUnified.getWordVectorMatrix("Morgan_Freeman");

        assertNotEquals(null, arrayLive);
        assertEquals(arrayLive, arrayStatic);

        // we're trying EXTENDED model, but file doesn't have syn1/huffman info, so it should be silently degraded to simplified model
        assertEquals(null, ((InMemoryLookupTable) vectorsUnified.lookupTable()).getSyn1());
    }

    /**
     * This method tests binary file loading via unified loader
     *
     * @throws Exception
     */
    @Test
    public void testUnifiedLoaderBinary() throws Exception {

        logger.info("Executor name: {}", Nd4j.getExecutioner().getClass().getSimpleName());

        WordVectors vectorsLive = WordVectorSerializer.readWord2VecModel(binaryFile);
        WordVectors vectorsStatic = WordVectorSerializer.readWord2VecModel(binaryFile, false);

        INDArray arrayLive = vectorsLive.getWordVectorMatrix("Morgan_Freeman");
        INDArray arrayStatic = vectorsStatic.getWordVectorMatrix("Morgan_Freeman");

        assertNotEquals(null, arrayLive);
        assertEquals(arrayLive, arrayStatic);
    }

    @Ignore
    @Test
    public void testBiggerParavecLoader() throws Exception {
        ParagraphVectors vectors =
                        WordVectorSerializer.readParagraphVectors("C:\\Users\\raver\\Downloads\\10kNews.zip");
    }

    @Test
    public void testVocabPeristence() throws Exception {
        // we build vocab save it, and confirm equality

    }

    @Test
    public void testMalformedLabels1() throws Exception {
        List<String> words = new ArrayList<>();
        words.add("test A");
        words.add("test B");
        words.add("test\nC");
        words.add("test`D");
        words.add("test_E");
        words.add("test 5");

        AbstractCache<VocabWord> vocabCache = new AbstractCache<>();
        int cnt = 0;
        for (String word : words) {
            vocabCache.addToken(new VocabWord(1.0, word));
            vocabCache.addWordToIndex(cnt, word);
            cnt++;
        }

        vocabCache.elementAtIndex(1).markAsLabel(true);

        InMemoryLookupTable<VocabWord> lookupTable =
                        new InMemoryLookupTable<>(vocabCache, 10, false, 0.01, Nd4j.getRandom(), 0.0);
        lookupTable.resetWeights(true);

        assertNotEquals(null, lookupTable.getSyn0());
        assertNotEquals(null, lookupTable.getSyn1());
        assertNotEquals(null, lookupTable.getExpTable());
        assertEquals(null, lookupTable.getSyn1Neg());

        ParagraphVectors vec = new ParagraphVectors.Builder().lookupTable(lookupTable).vocabCache(vocabCache).build();

        File tempFile = File.createTempFile("temp", "w2v");
        tempFile.deleteOnExit();

        WordVectorSerializer.writeParagraphVectors(vec, tempFile);


        ParagraphVectors restoredVec = WordVectorSerializer.readParagraphVectors(tempFile);

        for (String word : words) {
            assertEquals(true, restoredVec.hasWord(word));
        }

        assertTrue(restoredVec.getVocab().elementAtIndex(1).isLabel());
    }

    @Test
    public void testB64_1() throws Exception {
        String wordA = "night";
        String wordB = "night day";
        String encA = WordVectorSerializer.encodeB64(wordA);
        String encB = WordVectorSerializer.encodeB64(wordB);

        assertEquals(wordA, WordVectorSerializer.decodeB64(encA));
        assertEquals(wordB, WordVectorSerializer.decodeB64(encB));

        assertEquals(wordA, WordVectorSerializer.decodeB64(wordA));
        assertEquals(wordB, WordVectorSerializer.decodeB64(wordB));

    }
}
