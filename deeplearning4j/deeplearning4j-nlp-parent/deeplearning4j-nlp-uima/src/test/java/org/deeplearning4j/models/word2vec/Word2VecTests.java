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

package org.deeplearning4j.models.word2vec;

import com.google.common.primitives.Doubles;
import com.google.common.primitives.Ints;
import lombok.val;
import net.didion.jwnl.data.Word;
import org.apache.commons.io.FileUtils;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.nn.modelimport.keras.utils.KerasModelUtils;
import org.nd4j.linalg.io.ClassPathResource;
import org.deeplearning4j.models.embeddings.learning.impl.elements.CBOW;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.FlatModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
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
import org.nd4j.resources.Resources;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.*;

import static org.junit.Assert.*;


/**
 * @author jeffreytang
 */
public class Word2VecTests {

    private static final Logger log = LoggerFactory.getLogger(Word2VecTests.class);

    private File inputFile;
    private File inputFile2;
    private String pathToWriteto;
    private WordVectors googleModel;

    @Before
    public void before() throws Exception {
        File googleModelTextFile = Resources.asFile("word2vecserialization/google_news_30.txt");
        googleModel = WordVectorSerializer.readWord2VecModel(googleModelTextFile);
        inputFile = Resources.asFile("big/raw_sentences.txt");
        inputFile2 = Resources.asFile("big/raw_sentences_2.txt");

        File ptwt = new File(System.getProperty("java.io.tmpdir"), "testing_word2vec_serialization.txt");

        pathToWriteto = ptwt.getAbsolutePath();



        FileUtils.deleteDirectory(new File("word2vec-index"));
    }

    @Test
    public void testGoogleModelLoaded() throws Exception {
        assertEquals(googleModel.vocab().numWords(), 30);
        assertTrue(googleModel.hasWord("Morgan_Freeman"));
        double[] wordVector = googleModel.getWordVector("Morgan_Freeman");
        assertTrue(wordVector.length == 300);
        assertEquals(Doubles.asList(wordVector).get(0), 0.044423, 1e-3);
    }

    @Test
    public void testSimilarity() throws Exception {
        testGoogleModelLoaded();
        assertEquals(googleModel.similarity("Benkovic", "Boeremag_trialists"), 0.1204, 1e-2);
        assertEquals(googleModel.similarity("Benkovic", "Gopie"), 0.3350, 1e-2);
        assertEquals(googleModel.similarity("Benkovic", "Youku.com"), 0.0116, 1e-2);
    }

    @Test
    public void testWordsNearest() throws Exception {
        testGoogleModelLoaded();
        List<Object> lst = Arrays.asList(googleModel.wordsNearest("Benkovic", 10).toArray());

        assertTrue(lst.contains("Gopie"));
        assertTrue(lst.contains("JIM_HOOK_Senior"));
        /*
        assertEquals(lst.get(0), "Gopie");
        assertEquals(lst.get(1), "JIM_HOOK_Senior");
        */
    }

    @Test
    public void testUIMAIterator() throws Exception {
        SentenceIterator iter = UimaSentenceIterator.createWithPath(inputFile.getAbsolutePath());
        assertEquals(iter.nextSentence(), "No ,  he says now .");
    }

    @Test
    @Ignore // no adagrad these days
    public void testWord2VecAdaGrad() throws Exception {
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(5).iterations(5).learningRate(0.025).layerSize(100)
                        .seed(42).batchSize(13500).sampling(0).negativeSample(0)
                        //.epochs(10)
                        .windowSize(5).modelUtils(new BasicModelUtils<VocabWord>()).useAdaGrad(false)
                        .useHierarchicSoftmax(true).iterate(iter).workers(4).tokenizerFactory(t).build();

        vec.fit();

        Collection<String> lst = vec.wordsNearest("day", 10);
        log.info(Arrays.toString(lst.toArray()));

        //   assertEquals(10, lst.size());

        double sim = vec.similarity("day", "night");
        log.info("Day/night similarity: " + sim);

        assertTrue(lst.contains("week"));
        assertTrue(lst.contains("night"));
        assertTrue(lst.contains("year"));
    }

    @Test
    public void testWord2VecCBOW() throws Exception {
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(1).iterations(1).learningRate(0.025).layerSize(150)
                        .seed(42).sampling(0).negativeSample(0).useHierarchicSoftmax(true).windowSize(5)
                        .modelUtils(new BasicModelUtils<VocabWord>()).useAdaGrad(false).iterate(iter).workers(4)
                        .tokenizerFactory(t).elementsLearningAlgorithm(new CBOW<VocabWord>()).build();

        vec.fit();

        Collection<String> lst = vec.wordsNearest("day", 10);
        log.info(Arrays.toString(lst.toArray()));

        //   assertEquals(10, lst.size());

        double sim = vec.similarity("day", "night");
        log.info("Day/night similarity: " + sim);

        assertTrue(lst.contains("week"));
        assertTrue(lst.contains("night"));
        assertTrue(lst.contains("year"));
        assertTrue(sim > 0.65f);
    }


    @Test
    public void testWord2VecMultiEpoch() throws Exception {
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(1).iterations(5).learningRate(0.025).layerSize(150)
                        .seed(42).sampling(0).negativeSample(0).useHierarchicSoftmax(true).windowSize(5).epochs(3)
                        .modelUtils(new BasicModelUtils<VocabWord>()).useAdaGrad(false).iterate(iter).workers(8)
                        .tokenizerFactory(t).elementsLearningAlgorithm(new CBOW<VocabWord>()).build();

        vec.fit();

        Collection<String> lst = vec.wordsNearest("day", 10);
        log.info(Arrays.toString(lst.toArray()));

        //   assertEquals(10, lst.size());

        double sim = vec.similarity("day", "night");
        log.info("Day/night similarity: " + sim);

        assertTrue(lst.contains("week"));
        assertTrue(lst.contains("night"));
        assertTrue(lst.contains("year"));
    }

    @Test
    public void reproducibleResults_ForMultipleRuns() throws Exception {
        log.info("reproducibleResults_ForMultipleRuns");
        val shakespear = new ClassPathResource("big/rnj.txt");
        val basic = new ClassPathResource("big/rnj.txt");
        SentenceIterator iter = new BasicLineIterator(inputFile);
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec1 = new Word2Vec.Builder().minWordFrequency(1).iterations(1).batchSize(8192).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .epochs(1).windowSize(5).allowParallelTokenization(true)
                .workers(1)
                .useHierarchicSoftmax(true)
                .modelUtils(new BasicModelUtils<VocabWord>()).iterate(iter).tokenizerFactory(t).build();

        Word2Vec vec2 = new Word2Vec.Builder().minWordFrequency(1).iterations(1).batchSize(8192).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .epochs(1).windowSize(5).allowParallelTokenization(true)
                .workers(1)
                .useHierarchicSoftmax(true)
                .modelUtils(new BasicModelUtils<VocabWord>()).iterate(iter).tokenizerFactory(t).build();

        vec1.fit();

        iter.reset();

        vec2.fit();

        for (int e = 0; e < vec1.getVocab().numWords(); e++) {
            val w1 = vec1.getVocab().elementAtIndex(e);
            val w2 = vec2.getVocab().elementAtIndex(e);

            assertNotNull(w1);
            assertNotNull(w2);

            assertEquals(w1.getLabel(), w2.getLabel());

            assertArrayEquals("Failed for token [" + w1.getLabel() + "] at index [" + e + "]", Ints.toArray(w1.getPoints()), Ints.toArray(w2.getPoints()));
            assertArrayEquals("Failed for token [" + w1.getLabel() + "] at index [" + e + "]", Ints.toArray(w1.getCodes()), Ints.toArray(w2.getCodes()));
        }

        val syn0_from_vec1 = ((InMemoryLookupTable<VocabWord>) vec1.getLookupTable()).getSyn0();
        val syn0_from_vec2 = ((InMemoryLookupTable<VocabWord>) vec2.getLookupTable()).getSyn0();

        assertEquals(syn0_from_vec1, syn0_from_vec2);

        log.info("Day/night similarity: {}", vec1.similarity("day", "night"));
        val result = vec1.wordsNearest("day", 10);
        printWords("day", result, vec1);
    }
    
    @Test
    public void testRunWord2Vec() throws Exception {
        // Strip white space before and after for each line
        /*val shakespear = new ClassPathResource("big/rnj.txt");
        SentenceIterator iter = new BasicLineIterator(shakespear.getFile());*/
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());


        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(1).iterations(1).batchSize(8192).layerSize(100)
                        .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                        .sampling(0).elementsLearningAlgorithm(new SkipGram<VocabWord>())
                        //.negativeSample(10)
                        .epochs(1).windowSize(5).allowParallelTokenization(true)
                        .workers(6)
                        .usePreciseMode(true)
                        .modelUtils(new BasicModelUtils<VocabWord>()).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());
        vec.fit();
        File tempFile = File.createTempFile("temp", "temp");
        tempFile.deleteOnExit();

        WordVectorSerializer.writeFullModel(vec, tempFile.getAbsolutePath());
        Collection<String> lst = vec.wordsNearest("day", 10);
        //log.info(Arrays.toString(lst.toArray()));
        printWords("day", lst, vec);

        assertEquals(10, lst.size());

        double sim = vec.similarity("day", "night");
        log.info("Day/night similarity: " + sim);

        assertTrue(sim < 1.0);
        assertTrue(sim > 0.4);


        assertTrue(lst.contains("week"));
        assertTrue(lst.contains("night"));
        assertTrue(lst.contains("year"));

        assertFalse(lst.contains(null));


        lst = vec.wordsNearest("day", 10);
        //log.info(Arrays.toString(lst.toArray()));
        printWords("day", lst, vec);

        assertTrue(lst.contains("week"));
        assertTrue(lst.contains("night"));
        assertTrue(lst.contains("year"));

        new File("cache.ser").delete();

        ArrayList<String> labels = new ArrayList<>();
        labels.add("day");
        labels.add("night");
        labels.add("week");

        INDArray matrix = vec.getWordVectors(labels);
        assertEquals(matrix.getRow(0, true), vec.getWordVectorMatrix("day"));
        assertEquals(matrix.getRow(1, true), vec.getWordVectorMatrix("night"));
        assertEquals(matrix.getRow(2, true), vec.getWordVectorMatrix("week"));

        WordVectorSerializer.writeWordVectors(vec, pathToWriteto);
    }

    /**
     * Adding test for cosine similarity, to track changes in Transforms.cosineSim()
     */
    @Test
    public void testCosineSim() {
        double[] array1 = new double[] {1.01, 0.91, 0.81, 0.71};
        double[] array2 = new double[] {1.01, 0.91, 0.81, 0.71};
        double[] array3 = new double[] {1.0, 0.9, 0.8, 0.7};

        double sim12 = Transforms.cosineSim(Nd4j.create(array1), Nd4j.create(array2));
        double sim23 = Transforms.cosineSim(Nd4j.create(array2), Nd4j.create(array3));
        log.info("Arrays 1/2 cosineSim: " + sim12);
        log.info("Arrays 2/3 cosineSim: " + sim23);
        log.info("Arrays 1/2 dot: " + Nd4j.getBlasWrapper().dot(Nd4j.create(array1), Nd4j.create(array2)));
        log.info("Arrays 2/3 dot: " + Nd4j.getBlasWrapper().dot(Nd4j.create(array2), Nd4j.create(array3)));

        assertEquals(1.0d, sim12, 0.01d);
        assertEquals(0.99d, sim23, 0.01d);
    }

    @Test
    public void testLoadingWordVectors() throws Exception {
        File modelFile = new File(pathToWriteto);
        if (!modelFile.exists()) {
            testRunWord2Vec();
        }
        WordVectors wordVectors = WordVectorSerializer.loadTxtVectors(modelFile);
        Collection<String> lst = wordVectors.wordsNearest("day", 10);
        System.out.println(Arrays.toString(lst.toArray()));
    }

    @Ignore
    @Test
    public void testWord2VecGoogleModelUptraining() throws Exception {
        long time1 = System.currentTimeMillis();
        Word2Vec vec = WordVectorSerializer.readWord2VecModel(
                        new File("C:\\Users\\raver\\Downloads\\GoogleNews-vectors-negative300.bin.gz"), false);
        long time2 = System.currentTimeMillis();
        log.info("Model loaded in {} msec", time2 - time1);
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        vec.setTokenizerFactory(t);
        vec.setSentenceIterator(iter);
        vec.getConfiguration().setUseHierarchicSoftmax(false);
        vec.getConfiguration().setNegative(5.0);
        vec.setElementsLearningAlgorithm(new CBOW<VocabWord>());

        vec.fit();
    }

    @Test
    public void testW2VnegativeOnRestore() throws Exception {
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());


        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(1).iterations(3).batchSize(8192).layerSize(100)
                        .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                        .sampling(0).elementsLearningAlgorithm(new SkipGram<VocabWord>()).negativeSample(10).epochs(1)
                        .windowSize(5).useHierarchicSoftmax(false).allowParallelTokenization(true)
                        .modelUtils(new FlatModelUtils<VocabWord>()).iterate(iter).tokenizerFactory(t).build();


        assertEquals(false, vec.getConfiguration().isUseHierarchicSoftmax());

        log.info("Fit 1");
        vec.fit();

        File tmpFile = File.createTempFile("temp", "file");
        tmpFile.deleteOnExit();

        WordVectorSerializer.writeWord2VecModel(vec, tmpFile);

        iter.reset();

        Word2Vec restoredVec = WordVectorSerializer.readWord2VecModel(tmpFile, true);
        restoredVec.setTokenizerFactory(t);
        restoredVec.setSentenceIterator(iter);

        assertEquals(false, restoredVec.getConfiguration().isUseHierarchicSoftmax());
        assertTrue(restoredVec.getModelUtils() instanceof FlatModelUtils);
        assertTrue(restoredVec.getConfiguration().isAllowParallelTokenization());

        log.info("Fit 2");
        restoredVec.fit();


        iter.reset();
        restoredVec = WordVectorSerializer.readWord2VecModel(tmpFile, false);
        restoredVec.setTokenizerFactory(t);
        restoredVec.setSentenceIterator(iter);

        assertEquals(false, restoredVec.getConfiguration().isUseHierarchicSoftmax());
        assertTrue(restoredVec.getModelUtils() instanceof BasicModelUtils);

        log.info("Fit 3");
        restoredVec.fit();
    }

    @Test
    public void testUnknown1() throws Exception {
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(10).useUnknown(true)
                        .unknownElement(new VocabWord(1.0, "PEWPEW")).iterations(1).layerSize(100)
                        .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                        .sampling(0).elementsLearningAlgorithm(new CBOW<VocabWord>()).epochs(1).windowSize(5)
                        .useHierarchicSoftmax(true).allowParallelTokenization(true)
                        .modelUtils(new FlatModelUtils<VocabWord>()).iterate(iter).tokenizerFactory(t).build();

        vec.fit();

        assertTrue(vec.hasWord("PEWPEW"));
        assertTrue(vec.getVocab().containsWord("PEWPEW"));

        INDArray unk = vec.getWordVectorMatrix("PEWPEW");
        assertNotEquals(null, unk);

        File tempFile = File.createTempFile("temp", "file");
        tempFile.deleteOnExit();

        WordVectorSerializer.writeWord2VecModel(vec, tempFile);

        log.info("Original configuration: {}", vec.getConfiguration());

        Word2Vec restored = WordVectorSerializer.readWord2VecModel(tempFile);

        assertTrue(restored.hasWord("PEWPEW"));
        assertTrue(restored.getVocab().containsWord("PEWPEW"));
        INDArray unk_restored = restored.getWordVectorMatrix("PEWPEW");

        assertEquals(unk, unk_restored);



        // now we're getting some junk word
        INDArray random = vec.getWordVectorMatrix("hhsd7d7sdnnmxc_SDsda");
        INDArray randomRestored = restored.getWordVectorMatrix("hhsd7d7sdnnmxc_SDsda");

        log.info("Restored configuration: {}", restored.getConfiguration());

        assertEquals(unk, random);
        assertEquals(unk, randomRestored);
    }

    @Test
    public void orderIsCorrect_WhenParallelized() throws Exception {
        // Strip white space before and after for each line
        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());


        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(1).iterations(3).batchSize(64).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new SkipGram<VocabWord>())
                //.negativeSample(10)
                .epochs(1).windowSize(5).allowParallelTokenization(true)
                .workers(1)
                .modelUtils(new BasicModelUtils<VocabWord>()).iterate(iter).tokenizerFactory(t).build();


        vec.fit();
        System.out.println(vec.getVocab().numWords());

        val words = vec.getVocab().words();
        for (val word : words) {
            System.out.println(word);
        }
    }

    @Test
    public void testJSONSerialization() {
        Word2Vec word2Vec = new Word2Vec.Builder()
                .layerSize(1000)
                .limitVocabularySize(1000)
                .elementsLearningAlgorithm(CBOW.class.getCanonicalName())
                .allowParallelTokenization(true)
                .modelUtils(new FlatModelUtils<VocabWord>())
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


        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        val words = new VocabWord[3];
        words[0] = new VocabWord(1.0, "word");
        words[1] = new VocabWord(2.0, "test");
        words[2] = new VocabWord(3.0, "tester");

        for (int i = 0; i < words.length; ++i) {
            cache.addToken(words[i]);
            cache.addWordToIndex(i, words[i].getLabel());
        }
        word2Vec.setVocab(cache);

        String json = null;
        Word2Vec unserialized = null;
        try {
            json = word2Vec.toJson();
            log.info("{}", json.toString());

            unserialized = Word2Vec.fromJson(json);
        }
        catch (Exception e) {
            e.printStackTrace();
            fail();
        }

        assertEquals(cache.totalWordOccurrences(),((Word2Vec) unserialized).getVocab().totalWordOccurrences());
        assertEquals(cache.totalNumberOfDocs(), ((Word2Vec) unserialized).getVocab().totalNumberOfDocs());

        for (int i = 0; i < words.length; ++i) {
            val cached = cache.wordAtIndex(i);
            val restored = ((Word2Vec) unserialized).getVocab().wordAtIndex(i);
            assertNotNull(cached);
            assertEquals(cached, restored);
        }
    }

    @Test
    public void testWord2VecConfigurationConsistency() {
        VectorsConfiguration configuration = new VectorsConfiguration();

        assertEquals(configuration.getLayersSize(), 200);
        assertEquals(configuration.getLayersSize(), 200);
        assert(configuration.getElementsLearningAlgorithm() == null);
        assertEquals(configuration.isAllowParallelTokenization(), false);
        assertEquals(configuration.isPreciseMode(), false);
        assertEquals(configuration.getBatchSize(), 512);
        assert(configuration.getModelUtils() == null);
        assertTrue(!configuration.isPreciseMode());
        assertEquals(configuration.getBatchSize(), 512);
        assertEquals(configuration.getWindow(), 5);
        assertEquals(configuration.getMinWordFrequency(), 5);
        assertEquals(configuration.getIterations(), 1);
        assertEquals(configuration.getSeed(), 0);
        assertEquals(configuration.getLearningRate(), 0.025, 1e-5f);
        assertEquals(configuration.getEpochs(), 1);
        assertTrue(configuration.getStopList().isEmpty());
        assertEquals(configuration.getSampling(), 0.0, 1e-5f);
        assertEquals(configuration.getNegative(), 0, 1e-5f);
        assertTrue(!configuration.isUseAdaGrad());
        assertTrue(configuration.isUseHierarchicSoftmax());
        assertEquals(configuration.getMinLearningRate(), 1.0E-4, 1e-5f);
        assertTrue(!configuration.isUseUnknown());


        Word2Vec word2Vec = new Word2Vec.Builder(configuration)
                .layerSize(1000)
                .limitVocabularySize(1000)
                .elementsLearningAlgorithm(CBOW.class.getCanonicalName())
                .allowParallelTokenization(true)
                .modelUtils(new FlatModelUtils<VocabWord>())
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

        assertEquals(word2Vec.getConfiguration().getLayersSize(), word2Vec.getLayerSize());
        assertEquals(word2Vec.getConfiguration().getLayersSize(), 1000);
        assertEquals(word2Vec.getConfiguration().getElementsLearningAlgorithm(), CBOW.class.getCanonicalName());
        assertEquals(word2Vec.getConfiguration().isAllowParallelTokenization(), true);
        assertEquals(word2Vec.getConfiguration().isPreciseMode(), true);
        assertEquals(word2Vec.getConfiguration().getBatchSize(), 1024);

        String modelUtilsName = word2Vec.getConfiguration().getModelUtils();
        assertEquals(modelUtilsName, FlatModelUtils.class.getCanonicalName());

        assertTrue(word2Vec.getConfiguration().isPreciseMode());
        assertEquals(word2Vec.getConfiguration().getBatchSize(), 1024);

        assertEquals(word2Vec.getConfiguration().getWindow(), 23);
        assertEquals(word2Vec.getConfiguration().getMinWordFrequency(), 24);
        assertEquals(word2Vec.getConfiguration().getIterations(), 54);
        assertEquals(word2Vec.getConfiguration().getSeed(), 45);
        assertEquals(word2Vec.getConfiguration().getLearningRate(), 0.08, 1e-5f);
        assertEquals(word2Vec.getConfiguration().getEpochs(), 45);

        assertEquals(word2Vec.getConfiguration().getStopList().size(), 1);

        assertEquals(configuration.getSampling(), 44.0, 1e-5f);
        assertEquals(configuration.getNegative(), 56.0, 1e-5f);
        assertTrue(configuration.isUseAdaGrad());
        assertTrue(!configuration.isUseHierarchicSoftmax());
        assertEquals(configuration.getMinLearningRate(), 0.002, 1e-5f);
        assertTrue(configuration.isUseUnknown());
    }

    @Test
    public void testWordVectorsPartiallyAbsentLabels() throws Exception {

        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(10).useUnknown(true)
                .iterations(1).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new CBOW<VocabWord>()).epochs(1).windowSize(5)
                .useHierarchicSoftmax(true).allowParallelTokenization(true)
                .useUnknown(false)
                .modelUtils(new FlatModelUtils<VocabWord>()).iterate(iter).tokenizerFactory(t).build();

        vec.fit();

        ArrayList<String> labels = new ArrayList<>();
        labels.add("fewfew");
        labels.add("day");
        labels.add("night");
        labels.add("week");

        INDArray matrix = vec.getWordVectors(labels);
        assertEquals(3, matrix.rows());
        assertEquals(matrix.getRow(0, true), vec.getWordVectorMatrix("day"));
        assertEquals(matrix.getRow(1, true), vec.getWordVectorMatrix("night"));
        assertEquals(matrix.getRow(2, true), vec.getWordVectorMatrix("week"));
    }


    @Test
    public void testWordVectorsAbsentLabels() throws Exception {

        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(10).useUnknown(true)
                .iterations(1).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new CBOW<VocabWord>()).epochs(1).windowSize(5)
                .useHierarchicSoftmax(true).allowParallelTokenization(true)
                .useUnknown(false)
                .modelUtils(new FlatModelUtils<VocabWord>()).iterate(iter).tokenizerFactory(t).build();

        vec.fit();

        ArrayList<String> labels = new ArrayList<>();
        labels.add("fewfew");

        INDArray matrix = vec.getWordVectors(labels);
        assertTrue(matrix.isEmpty());
    }

    @Test
    public void testWordVectorsAbsentLabels_WithUnknown() throws Exception {

        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());
        // Split on white spaces in the line to get words
        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec vec = new Word2Vec.Builder().minWordFrequency(1).iterations(1).batchSize(8192).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new SkipGram<VocabWord>())
                //.negativeSample(10)
                .epochs(1).windowSize(5).allowParallelTokenization(true)
                .workers(4)
                .modelUtils(new BasicModelUtils<VocabWord>()).iterate(iter).tokenizerFactory(t)
                .useUnknown(true).unknownElement(new VocabWord(1, "UNKOWN")).build();

        vec.fit();

        ArrayList<String> labels = new ArrayList<>();
        labels.add("bus");
        labels.add("car");

        INDArray matrix = vec.getWordVectors(labels);
        for (int i = 0; i < labels.size(); ++i)
            assertEquals(matrix.getRow(i, true), vec.getWordVectorMatrix("UNKNOWN"));
    }

    @Test
    public void weightsNotUpdated_WhenLocked() throws Exception {

        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        Word2Vec vec1 = new Word2Vec.Builder().minWordFrequency(1).iterations(3).batchSize(64).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .epochs(1).windowSize(5).allowParallelTokenization(true)
                .workers(1)
                .iterate(iter)
                .modelUtils(new BasicModelUtils<VocabWord>()).build();

        vec1.fit();

        iter = new BasicLineIterator(inputFile2.getAbsolutePath());
        Word2Vec vec2 = new Word2Vec.Builder().minWordFrequency(1).iterations(3).batchSize(32).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(32).learningRate(0.021).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .epochs(1).windowSize(5).allowParallelTokenization(true)
                .workers(1)
                .iterate(iter)
                .intersectModel(vec1, true)
                .modelUtils(new BasicModelUtils<VocabWord>()).build();

        vec2.fit();

        assertEquals(vec1.getWordVectorMatrix("put"), vec2.getWordVectorMatrix("put"));
        assertEquals(vec1.getWordVectorMatrix("part"), vec2.getWordVectorMatrix("part"));
        assertEquals(vec1.getWordVectorMatrix("made"), vec2.getWordVectorMatrix("made"));
        assertEquals(vec1.getWordVectorMatrix("money"), vec2.getWordVectorMatrix("money"));
    }

    @Test
    public void weightsNotUpdated_WhenLocked_CBOW() throws Exception {

        SentenceIterator iter = new BasicLineIterator(inputFile.getAbsolutePath());

        Word2Vec vec1 = new Word2Vec.Builder().minWordFrequency(1).iterations(1).batchSize(8192).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(42).learningRate(0.025).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new CBOW<VocabWord>())
                .epochs(1).windowSize(5).allowParallelTokenization(true)
                .workers(1)
                .iterate(iter)
                .modelUtils(new BasicModelUtils<VocabWord>()).build();

        vec1.fit();

        log.info("Fit 1 finished");

        iter = new BasicLineIterator(inputFile2.getAbsolutePath());
        Word2Vec vec2 = new Word2Vec.Builder().minWordFrequency(1).iterations(1).batchSize(8192).layerSize(100)
                .stopWords(new ArrayList<String>()).seed(32).learningRate(0.021).minLearningRate(0.001)
                .sampling(0).elementsLearningAlgorithm(new CBOW<VocabWord>())
                .epochs(1).windowSize(5).allowParallelTokenization(true)
                .workers(1)
                .iterate(iter)
                .intersectModel(vec1, true)
                .modelUtils(new BasicModelUtils<VocabWord>()).build();

        vec2.fit();

        log.info("Fit 2 finished");

        assertEquals(vec1.getWordVectorMatrix("put"), vec2.getWordVectorMatrix("put"));
        assertEquals(vec1.getWordVectorMatrix("part"), vec2.getWordVectorMatrix("part"));
        assertEquals(vec1.getWordVectorMatrix("made"), vec2.getWordVectorMatrix("made"));
        assertEquals(vec1.getWordVectorMatrix("money"), vec2.getWordVectorMatrix("money"));
    }

    private static void printWords(String target, Collection<String> list, Word2Vec vec) {
        System.out.println("Words close to [" + target + "]:");
        for (String word : list) {
            double sim = vec.similarity(target, word);
            System.out.print("'" + word + "': [" + sim + "]");
        }
        System.out.print("\n");
    }
    //
}

