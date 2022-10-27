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

package org.deeplearning4j.models.paragraphvectors;


import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.apache.commons.io.IOUtils;
import org.apache.commons.io.LineIterator;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DBOW;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DM;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.sequencevectors.sequence.Sequence;
import org.deeplearning4j.models.sequencevectors.transformers.impl.SentenceTransformer;
import org.deeplearning4j.models.sequencevectors.transformers.impl.iterables.BasicTransformerIterator;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.*;
import org.deeplearning4j.text.sentenceiterator.*;
import org.deeplearning4j.text.sentenceiterator.interoperability.SentenceIteratorConverter;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.Timeout;
import org.junit.jupiter.api.io.TempDir;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.io.ClassPathResource;
import org.nd4j.common.io.CollectionUtils;
import org.nd4j.common.resources.Resources;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.common.util.SerializationUtils;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.io.*;
import java.nio.charset.StandardCharsets;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

import static org.junit.jupiter.api.Assertions.*;

@Slf4j
@Tag(TagNames.FILE_IO)
@NativeTag
public class ParagraphVectorsTest extends BaseDL4JTest {

    @Override
    public long getTimeoutMilliseconds() {
        return isIntegrationTests() ? 600_000 : 240_000;
    }


    @Override
    public DataType getDataType() {
        return DataType.FLOAT;
    }

    @Override
    public DataType getDefaultFPDataType() {
        return DataType.FLOAT;
    }




    /**
     * This test checks, how vocab is built using SentenceIterator provided, without labels.
     *
     * @throws Exception
     */
    @Timeout(2400000)
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testParagraphVectorsVocabBuilding1() throws Exception {
        File file = Resources.asFile("/big/raw_sentences.txt");
        SentenceIterator iter = new BasicLineIterator(file); //UimaSentenceIterator.createWithPath(file.getAbsolutePath());

        int numberOfLines = 0;
        while (iter.hasNext()) {
            iter.nextSentence();
            numberOfLines++;
        }

        iter.reset();

        InMemoryLookupCache cache = new InMemoryLookupCache(false);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        // LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder().minWordFrequency(1).iterations(5).layerSize(100)
                //      .labelsGenerator(source)
                .windowSize(5).iterate(iter).vocabCache(cache).tokenizerFactory(t).build();

        vec.buildVocab();

        LabelsSource source = vec.getLabelsSource();


        //VocabCache cache = vec.getVocab();
        log.info("Number of lines in corpus: " + numberOfLines);
        assertEquals(numberOfLines, source.getLabels().size());
        assertEquals(97162, source.getLabels().size());

        assertNotEquals(null, cache);
        assertEquals(97406, cache.numWords());

        // proper number of words for minWordsFrequency = 1 is 244
        assertEquals(244, cache.numWords() - source.getLabels().size());
    }


    /**
     * This test doesn't really care about actual results. We only care about equality between live model & restored models
     *
     * @throws Exception
     */
    @Timeout(3000000)
    @Tag(TagNames.LONG_TEST)
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testParagraphVectorsModelling1(Nd4jBackend backend) throws Exception {
        for(boolean binary : new boolean[] {true,false}) {
            File file = Resources.asFile("/big/raw_sentences.txt");
            SentenceIterator iter = new BasicLineIterator(file);

            TokenizerFactory t = new DefaultTokenizerFactory();
            t.setTokenPreProcessor(new CommonPreprocessor());

            LabelsSource source = new LabelsSource("DOC_");

            ParagraphVectors vec = new ParagraphVectors.Builder().minWordFrequency(1).iterations(5).seed(119).epochs(1)
                    .layerSize(150).learningRate(0.025).labelsSource(source).windowSize(5)
                    .sequenceLearningAlgorithm(new DM<>())
                    .iterate(iter)
                    .trainWordVectors(true)
                    .usePreciseWeightInit(true)
                    .batchSize(8192)
                    .tokenizerFactory(t)
                    .workers(4)
                    .sampling(0)
                    .build();

            vec.fit();

            VocabCache<VocabWord> cache = vec.getVocab();

            File fullFile = File.createTempFile("paravec", "tests");
            fullFile.deleteOnExit();

            INDArray originalSyn1_17 = ((InMemoryLookupTable) vec.getLookupTable()).getSyn1().getRow(17, true).dup();

            if(binary)
                WordVectorSerializer.writeParagraphVectorsBinary(vec,fullFile);
            else
                WordVectorSerializer.writeParagraphVectors(vec, fullFile);

            ParagraphVectors paragraphVectors = binary ?
                    WordVectorSerializer.readParagraphVectorsBinary(fullFile) :
                    WordVectorSerializer.readParagraphVectors(fullFile);


            int cnt1 = cache.wordFrequency("day");
            int cnt2 = cache.wordFrequency("me");

            assertNotEquals(1, cnt1);
            assertNotEquals(1, cnt2);
            assertNotEquals(cnt1, cnt2);

            assertEquals(97406, cache.numWords());

            assertTrue(vec.hasWord("DOC_16392"));
            assertTrue(vec.hasWord("DOC_3720"));

            List<String> result = new ArrayList<>(vec.nearestLabels(vec.getWordVectorMatrix("DOC_16392"), 10));
            System.out.println("nearest labels: " + result);
            for (String label : result) {
                System.out.println(label + "/DOC_16392: " + vec.similarity(label, "DOC_16392"));
            }
            assertTrue(result.contains("DOC_16392"));
            //assertTrue(result.contains("DOC_21383"));



        /*
            We have few lines that contain pretty close words invloved.
            These sentences should be pretty close to each other in vector space
         */
            // line 3721: This is my way .
            // line 6348: This is my case .
            // line 9836: This is my house .
            // line 12493: This is my world .
            // line 16393: This is my work .

            // this is special sentence, that has nothing common with previous sentences
            // line 9853: We now have one .

            double similarityD = vec.similarity("day", "night");
            log.info("day/night similarity: " + similarityD);

            if (similarityD < 0.0) {
                log.info("Day: " + Arrays.toString(vec.getWordVectorMatrix("day").dup().data().asDouble()));
                log.info("Night: " + Arrays.toString(vec.getWordVectorMatrix("night").dup().data().asDouble()));
            }


            List<String> labelsOriginal = vec.labelsSource.getLabels();

            double similarityW = vec.similarity("way", "work");
            log.info("way/work similarity: " + similarityW);

            double similarityH = vec.similarity("house", "world");
            log.info("house/world similarity: " + similarityH);

            double similarityC = vec.similarity("case", "way");
            log.info("case/way similarity: " + similarityC);

            double similarity1 = vec.similarity("DOC_9835", "DOC_12492");
            log.info("9835/12492 similarity: " + similarity1);
            //        assertTrue(similarity1 > 0.7d);

            double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
            log.info("3720/16392 similarity: " + similarity2);
            //        assertTrue(similarity2 > 0.7d);

            double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
            log.info("6347/3720 similarity: " + similarity3);
            //        assertTrue(similarity2 > 0.7d);

            // likelihood in this case should be significantly lower
            double similarityX = vec.similarity("DOC_3720", "DOC_9852");
            log.info("3720/9852 similarity: " + similarityX);
            assertTrue(similarityX < 0.5d);

            File tempFile = File.createTempFile("paravec", "ser");
            tempFile.deleteOnExit();

            INDArray day = vec.getWordVectorMatrix("day").dup();

        /*
            Testing txt serialization
         */
            File tempFile2 = File.createTempFile("paravec", "ser");
            tempFile2.deleteOnExit();

            if(!binary)
                WordVectorSerializer.writeWordVectors(vec, tempFile2);
            else
                WordVectorSerializer.writeParagraphVectorsBinary(vec,tempFile2);
            ParagraphVectors vec3 = binary ? WordVectorSerializer.readParagraphVectorsBinary(tempFile2)
                    : WordVectorSerializer.readParagraphVectorsFromText(tempFile2);

            INDArray day3 = vec3.getWordVectorMatrix("day").dup();

            List<String> labelsRestored = vec3.labelsSource.getLabels();

            assertEquals(day, day3);

            assertEquals(labelsOriginal.size(), labelsRestored.size());

        /*
         Testing binary serialization
        */
            SerializationUtils.saveObject(vec, tempFile);


            ParagraphVectors vec2 = SerializationUtils.readObject(tempFile);
            INDArray day2 = vec2.getWordVectorMatrix("day").dup();

            List<String> labelsBinary = vec2.labelsSource.getLabels();

            assertEquals(day, day2);

            tempFile.delete();


            assertEquals(labelsOriginal.size(), labelsBinary.size());

            INDArray original = vec.getWordVectorMatrix("DOC_16392").dup();
            INDArray originalPreserved = original.dup();
            INDArray inferredA1 = vec.inferVector("This is my work .");
            INDArray inferredB1 = vec.inferVector("This is my work .");

            double cosAO1 = Transforms.cosineSim(inferredA1.dup(), original.dup());
            double cosAB1 = Transforms.cosineSim(inferredA1.dup(), inferredB1.dup());

            log.info("Cos O/A: {}", cosAO1);
            log.info("Cos A/B: {}", cosAB1);
            log.info("Inferred: {}", inferredA1);
            //        assertTrue(cosAO1 > 0.45);
            assertTrue(cosAB1 > 0.95);

            //assertArrayEquals(inferredA.data().asDouble(), inferredB.data().asDouble(), 0.01);

            ParagraphVectors restoredVectors = binary ? WordVectorSerializer.readParagraphVectorsBinary(fullFile)  :
                    WordVectorSerializer.readParagraphVectors(fullFile);
            restoredVectors.setTokenizerFactory(t);

            INDArray restoredSyn1_17 = ((InMemoryLookupTable) restoredVectors.getLookupTable()).getSyn1().getRow(17, true).dup();

            assertEquals(originalSyn1_17, restoredSyn1_17);

            INDArray originalRestored = vec.getWordVectorMatrix("DOC_16392").dup();

            assertEquals(originalPreserved, originalRestored);

            INDArray inferredA2 = restoredVectors.inferVector("This is my work .");
            INDArray inferredB2 = restoredVectors.inferVector("This is my work .");
            INDArray inferredC2 = restoredVectors.inferVector("world way case .");

            double cosAO2 = Transforms.cosineSim(inferredA2.dup(), original.dup());
            double cosAB2 = Transforms.cosineSim(inferredA2.dup(), inferredB2.dup());
            double cosAAX = Transforms.cosineSim(inferredA1.dup(), inferredA2.dup());
            double cosAC2 = Transforms.cosineSim(inferredC2.dup(), inferredA2.dup());

            log.info("Cos A2/B2: {}", cosAB2);
            log.info("Cos A1/A2: {}", cosAAX);
            log.info("Cos O/A2: {}", cosAO2);
            log.info("Cos C2/A2: {}", cosAC2);

            log.info("Vector: {}", Arrays.toString(inferredA1.data().asFloat()));

            log.info("cosAO2: {}", cosAO2);

            //  assertTrue(cosAO2 > 0.45);
            assertTrue(cosAB2 > 0.95);
            assertTrue(cosAAX > 0.95);
        }

    }


    @Test
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testParagraphVectorsDM() throws Exception {
        File file = Resources.asFile("/big/raw_sentences.txt");
        SentenceIterator iter = new BasicLineIterator(file);

        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder().minWordFrequency(1).iterations(2).seed(119).epochs(1)
                .layerSize(100).learningRate(0.025).labelsSource(source).windowSize(5).iterate(iter)
                .trainWordVectors(true).vocabCache(cache).tokenizerFactory(t).negativeSample(0)
                .useHierarchicSoftmax(true).sampling(0).workers(1).usePreciseWeightInit(true)
                .sequenceLearningAlgorithm(new DM<VocabWord>()).build();

        vec.fit();


        int cnt1 = cache.wordFrequency("day");
        int cnt2 = cache.wordFrequency("me");

        assertNotEquals(1, cnt1);
        assertNotEquals(1, cnt2);
        assertNotEquals(cnt1, cnt2);

        double simDN = vec.similarity("day", "night");
        log.info("day/night similariry: {}", simDN);

        double similarity1 = vec.similarity("DOC_9835", "DOC_12492");
        log.info("9835/12492 similarity: " + similarity1);
        //        assertTrue(similarity1 > 0.2d);

        double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
        log.info("3720/16392 similarity: " + similarity2);
        //      assertTrue(similarity2 > 0.2d);

        double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
        log.info("6347/3720 similarity: " + similarity3);
        //        assertTrue(similarity3 > 0.6d);

        double similarityX = vec.similarity("DOC_3720", "DOC_9852");
        log.info("3720/9852 similarity: " + similarityX);
        if(isIntegrationTests()) {
            assertTrue(similarityX < 0.5d);
        }


        // testing DM inference now

        INDArray original = vec.getWordVectorMatrix("DOC_16392").dup();
        INDArray inferredA1 = vec.inferVector("This is my work");
        INDArray inferredB1 = vec.inferVector("This is my work .");

        double cosAO1 = Transforms.cosineSim(inferredA1.dup(), original.dup());
        double cosAB1 = Transforms.cosineSim(inferredA1.dup(), inferredB1.dup());

        log.info("Cos O/A: {}", cosAO1);
        log.info("Cos A/B: {}", cosAB1);
    }


    @Timeout(300000)
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testParagraphVectorsDBOW() throws Exception {
        skipUnlessIntegrationTests();

        File file = Resources.asFile("/big/raw_sentences.txt");
        SentenceIterator iter = new BasicLineIterator(file);

        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder().minWordFrequency(1).iterations(5).seed(119).epochs(1)
                .layerSize(100).learningRate(0.025).labelsSource(source).windowSize(5).iterate(iter)
                .trainWordVectors(true).vocabCache(cache).tokenizerFactory(t).negativeSample(0)
                .allowParallelTokenization(true).useHierarchicSoftmax(true).sampling(0).workers(4)
                .usePreciseWeightInit(true).sequenceLearningAlgorithm(new DBOW<VocabWord>()).build();

        vec.fit();

        assertFalse(((InMemoryLookupTable<VocabWord>)vec.getLookupTable()).getSyn0().isAttached());
        assertFalse(((InMemoryLookupTable<VocabWord>)vec.getLookupTable()).getSyn1().isAttached());

        int cnt1 = cache.wordFrequency("day");
        int cnt2 = cache.wordFrequency("me");

        assertNotEquals(1, cnt1);
        assertNotEquals(1, cnt2);
        assertNotEquals(cnt1, cnt2);

        double simDN = vec.similarity("day", "night");
        log.info("day/night similariry: {}", simDN);

        double similarity1 = vec.similarity("DOC_9835", "DOC_12492");
        log.info("9835/12492 similarity: " + similarity1);
        //        assertTrue(similarity1 > 0.2d);

        double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
        log.info("3720/16392 similarity: " + similarity2);
        //      assertTrue(similarity2 > 0.2d);

        double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
        log.info("6347/3720 similarity: " + similarity3);
        //        assertTrue(similarity3 > 0.6d);

        double similarityX = vec.similarity("DOC_3720", "DOC_9852");
        log.info("3720/9852 similarity: " + similarityX);
        assertTrue(similarityX < 0.5d);


        // testing DM inference now

        INDArray original = vec.getWordVectorMatrix("DOC_16392").dup();
        INDArray inferredA1 = vec.inferVector("This is my work");
        INDArray inferredB1 = vec.inferVector("This is my work .");
        INDArray inferredC1 = vec.inferVector("This is my day");
        INDArray inferredD1 = vec.inferVector("This is my night");

        log.info("A: {}", Arrays.toString(inferredA1.data().asFloat()));
        log.info("C: {}", Arrays.toString(inferredC1.data().asFloat()));

        assertNotEquals(inferredA1, inferredC1);

        double cosAO1 = Transforms.cosineSim(inferredA1.dup(), original.dup());
        double cosAB1 = Transforms.cosineSim(inferredA1.dup(), inferredB1.dup());
        double cosAC1 = Transforms.cosineSim(inferredA1.dup(), inferredC1.dup());
        double cosCD1 = Transforms.cosineSim(inferredD1.dup(), inferredC1.dup());

        log.info("Cos O/A: {}", cosAO1);
        log.info("Cos A/B: {}", cosAB1);
        log.info("Cos A/C: {}", cosAC1);
        log.info("Cos C/D: {}", cosCD1);

    }

    @Test()
    @Timeout(300000)
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testParagraphVectorsWithWordVectorsModelling1() throws Exception {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if(!isIntegrationTests() && "CUDA".equalsIgnoreCase(backend)) {
            skipUnlessIntegrationTests(); //Skip CUDA except for integration tests due to very slow test speed
        }

        File file = Resources.asFile("/big/raw_sentences.txt");
        SentenceIterator iter = new BasicLineIterator(file);

        //        InMemoryLookupCache cache = new InMemoryLookupCache(false);
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder().minWordFrequency(1).iterations(3).epochs(1).layerSize(100)
                .learningRate(0.025).labelsSource(source).windowSize(5).iterate(iter).trainWordVectors(true)
                .vocabCache(cache).tokenizerFactory(t).sampling(0).build();

        vec.fit();


        int cnt1 = cache.wordFrequency("day");
        int cnt2 = cache.wordFrequency("me");

        assertNotEquals(1, cnt1);
        assertNotEquals(1, cnt2);
        assertNotEquals(cnt1, cnt2);

        /*
            We have few lines that contain pretty close words invloved.
            These sentences should be pretty close to each other in vector space
         */
        // line 3721: This is my way .
        // line 6348: This is my case .
        // line 9836: This is my house .
        // line 12493: This is my world .
        // line 16393: This is my work .

        // this is special sentence, that has nothing common with previous sentences
        // line 9853: We now have one .

        assertTrue(vec.hasWord("DOC_3720"));

        double similarityD = vec.similarity("day", "night");
        log.info("day/night similarity: " + similarityD);

        double similarityW = vec.similarity("way", "work");
        log.info("way/work similarity: " + similarityW);

        double similarityH = vec.similarity("house", "world");
        log.info("house/world similarity: " + similarityH);

        double similarityC = vec.similarity("case", "way");
        log.info("case/way similarity: " + similarityC);

        double similarity1 = vec.similarity("DOC_9835", "DOC_12492");
        log.info("9835/12492 similarity: " + similarity1);
        //        assertTrue(similarity1 > 0.7d);

        double similarity2 = vec.similarity("DOC_3720", "DOC_16392");
        log.info("3720/16392 similarity: " + similarity2);
        //        assertTrue(similarity2 > 0.7d);

        double similarity3 = vec.similarity("DOC_6347", "DOC_3720");
        log.info("6347/3720 similarity: " + similarity3);
        //        assertTrue(similarity2 > 0.7d);

        // likelihood in this case should be significantly lower
        // however, since corpus is small, and weight initialization is random-based, sometimes this test CAN fail
        double similarityX = vec.similarity("DOC_3720", "DOC_9852");
        log.info("3720/9852 similarity: " + similarityX);
        assertTrue(similarityX < 0.5d);


        double sim119 = vec.similarityToLabel("This is my case .", "DOC_6347");
        double sim120 = vec.similarityToLabel("This is my case .", "DOC_3720");
        log.info("1/2: " + sim119 + "/" + sim120);
        //assertEquals(similarity3, sim119, 0.001);
    }


    /**
     * This test is not indicative.
     * there's no need in this test within travis, use it manually only for problems detection
     *
     * @throws Exception
     */
    @Test
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testParagraphVectorsReducedLabels1(@TempDir Path testDir) throws Exception {
        val tempDir = testDir.toFile();
        ClassPathResource resource = new ClassPathResource("/labeled");
        resource.copyDirectory(tempDir);

        LabelAwareIterator iter = new FileLabelAwareIterator.Builder().addSourceFolder(tempDir).build();

        TokenizerFactory t = new DefaultTokenizerFactory();

        /**
         * Please note: text corpus is REALLY small, and some kind of "results" could be received with HIGH epochs number, like 30.
         * But there's no reason to keep at that high
         */

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .epochs(3)
                .layerSize(100)
                .stopWords(new ArrayList<String>())
                .windowSize(5).iterate(iter)
                .tokenizerFactory(t)
                .build();

        vec.fit();

        //WordVectorSerializer.writeWordVectors(vec, "vectors.txt");

        INDArray w1 = vec.lookupTable().vector("I");
        INDArray w2 = vec.lookupTable().vector("am");
        INDArray w3 = vec.lookupTable().vector("sad.");

        INDArray words = Nd4j.create(3, vec.lookupTable().layerSize());

        words.putRow(0, w1);
        words.putRow(1, w2);
        words.putRow(2, w3);


        INDArray mean = words.isMatrix() ? words.mean(0) : words;

        log.info("Mean" + Arrays.toString(mean.dup().data().asDouble()));
        log.info("Array" + Arrays.toString(vec.lookupTable().vector("negative").dup().data().asDouble()));

        double simN = Transforms.cosineSim(mean, vec.lookupTable().vector("negative"));
        log.info("Similarity negative: " + simN);


        double simP = Transforms.cosineSim(mean, vec.lookupTable().vector("neutral"));
        log.info("Similarity neutral: " + simP);

        double simV = Transforms.cosineSim(mean, vec.lookupTable().vector("positive"));
        log.info("Similarity positive: " + simV);
    }


    @Test()
    @Timeout(300000)
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testParallelIterator() throws IOException {
        TokenizerFactory factory = new DefaultTokenizerFactory();
        SentenceIterator iterator = new BasicLineIterator(Resources.asFile("big/raw_sentences.txt"));

        SentenceTransformer transformer = new SentenceTransformer.Builder().iterator(iterator).allowMultithreading(true)
                .tokenizerFactory(factory).build();

        BasicTransformerIterator iter = (BasicTransformerIterator)transformer.iterator();
        for (int i = 0; i < 100; ++i) {
            int cnt = 0;
            long counter = 0;
            Sequence<VocabWord> sequence = null;
            while (iter.hasNext()) {
                sequence = iter.next();
                counter += sequence.size();
                cnt++;
            }
            iter.reset();
            assertEquals(757172, counter);
        }
    }

    @Test
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testIterator(@TempDir Path testDir) throws IOException {
        val folder_labeled = new File(testDir.toFile(),"labeled");
        val folder_unlabeled = new File(testDir.toFile(),"unlabeled");
        assertTrue(folder_labeled.mkdirs());
        assertTrue(folder_unlabeled.mkdirs());
        new ClassPathResource("/paravec/labeled/").copyDirectory(folder_labeled);
        new ClassPathResource("/paravec/unlabeled/").copyDirectory(folder_unlabeled);


        FileLabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(folder_labeled).build();

        File resource_sentences = Resources.asFile("/big/raw_sentences.txt");
        SentenceIterator iter = new BasicLineIterator(resource_sentences);

        int i = 0;
        for (; i < 10; ++i) {
            int j = 0;
            int labels = 0;
            int words = 0;
            while (labelAwareIterator.hasNextDocument()) {
                ++j;
                LabelledDocument document = labelAwareIterator.nextDocument();
                labels += document.getLabels().size();
                List<VocabWord> lst =  document.getReferencedContent();
                if (!CollectionUtils.isEmpty(lst))
                    words += lst.size();
            }
            labelAwareIterator.reset();
            assertEquals(0, words);
            assertEquals(30, labels);
            assertEquals(30, j);
            j = 0;
            while (iter.hasNext()) {
                ++j;
                iter.nextSentence();
            }
            assertEquals(97162, j);
            iter.reset();
        }

    }

    /*
        In this test we'll build w2v model, and will use it's vocab and weights for ParagraphVectors.
        there's no need in this test within travis, use it manually only for problems detection
    */
    @Test
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testParagraphVectorsOverExistingWordVectorsModel(@TempDir Path testDir) throws Exception {
        String backend = Nd4j.getExecutioner().getEnvironmentInformation().getProperty("backend");
        if(!isIntegrationTests() && "CUDA".equalsIgnoreCase(backend)) {
            skipUnlessIntegrationTests(); //Skip CUDA except for integration tests due to very slow test speed
        }

        // we build w2v from multiple sources, to cover everything
        File resource_sentences = Resources.asFile("/big/raw_sentences.txt");

        val folder_mixed = testDir.toFile();
        ClassPathResource resource_mixed = new ClassPathResource("paravec/");
        resource_mixed.copyDirectory(folder_mixed);

        SentenceIterator iter = new AggregatingSentenceIterator.Builder()
                .addSentenceIterator(new BasicLineIterator(resource_sentences))
                .addSentenceIterator(new FileSentenceIterator(folder_mixed)).build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec wordVectors = new Word2Vec.Builder().seed(119).minWordFrequency(1).batchSize(250).iterations(1).epochs(1)
                .learningRate(0.025).layerSize(150).minLearningRate(0.001)
                .elementsLearningAlgorithm(new SkipGram<VocabWord>()).useHierarchicSoftmax(true).windowSize(5)
                .allowParallelTokenization(true)
                .workers(1)
                .iterate(iter).tokenizerFactory(t).build();

        wordVectors.fit();

        VocabWord day_A = wordVectors.getVocab().tokenFor("day");

        INDArray vector_day1 = wordVectors.getWordVectorMatrix("day").dup();

        // At this moment we have ready w2v model. It's time to use it for ParagraphVectors

        val folder_labeled = new File(testDir.toFile(),"labeled");
        val folder_unlabeled = new File(testDir.toFile(),"unlabeled");
        new ClassPathResource("/paravec/labeled/").copyDirectory(folder_labeled);
        new ClassPathResource("/paravec/unlabeled/").copyDirectory(folder_unlabeled);


        FileLabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(folder_labeled).build();


        // documents from this iterator will be used for classification
        FileLabelAwareIterator unlabeledIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(folder_unlabeled).build();


        // we're building classifier now, with pre-built w2v model passed in
        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder().seed(119).iterate(labelAwareIterator)
                .learningRate(0.025).minLearningRate(0.001).iterations(10).epochs(1).layerSize(150)
                .tokenizerFactory(t).sequenceLearningAlgorithm(new DBOW<VocabWord>()).useHierarchicSoftmax(true)
                .allowParallelTokenization(true)
                .workers(1)
                .trainWordVectors(false).useExistingWordVectors(wordVectors).build();

        paragraphVectors.fit();

        VocabWord day_B = paragraphVectors.getVocab().tokenFor("day");

        assertEquals(day_A.getIndex(), day_B.getIndex());

        /*
        double similarityD = wordVectors.similarity("day", "night");
        log.info("day/night similarity: " + similarityD);
        assertTrue(similarityD > 0.5d);
        */

        INDArray vector_day2 = paragraphVectors.getWordVectorMatrix("day").dup();
        double crossDay = arraysSimilarity(vector_day1, vector_day2);

        log.info("Day1: " + vector_day1);
        log.info("Day2: " + vector_day2);
        log.info("Cross-Day similarity: " + crossDay);
        log.info("Cross-Day similiarity 2: " + Transforms.cosineSim(Transforms.unitVec(vector_day1), Transforms.unitVec(vector_day2)));

        assertTrue(crossDay > 0.9d);

        /**
         *
         * Here we're checking cross-vocabulary equality
         *
         */
        /*
        Random rnd = new Random();
        VocabCache<VocabWord> cacheP = paragraphVectors.getVocab();
        VocabCache<VocabWord> cacheW = wordVectors.getVocab();
        for (int x = 0; x < 1000; x++) {
            int idx = rnd.nextInt(cacheW.numWords());

            String wordW = cacheW.wordAtIndex(idx);
            String wordP = cacheP.wordAtIndex(idx);

            assertEquals(wordW, wordP);

            INDArray arrayW = wordVectors.getWordVectorMatrix(wordW);
            INDArray arrayP = paragraphVectors.getWordVectorMatrix(wordP);

            double simWP = Transforms.cosineSim(arrayW, arrayP);
            assertTrue(simWP >= 0.9);
        }
        */

        log.info("Zfinance: " + paragraphVectors.getWordVectorMatrix("Zfinance"));
        log.info("Zhealth: " + paragraphVectors.getWordVectorMatrix("Zhealth"));
        log.info("Zscience: " + paragraphVectors.getWordVectorMatrix("Zscience"));

        assertTrue(unlabeledIterator.hasNext());
        LabelledDocument document = unlabeledIterator.nextDocument();

        log.info("Results for document '" + document.getLabel() + "'");

        List<String> results = new ArrayList<>(paragraphVectors.predictSeveral(document, 3));
        for (String result : results) {
            double sim = paragraphVectors.similarityToLabel(document, result);
            log.info("Similarity to [" + result + "] is [" + sim + "]");
        }

        String topPrediction = paragraphVectors.predict(document);
        assertEquals("Z"+document.getLabel(), topPrediction);
    }

    /*
        Left as reference implementation, before stuff was changed in w2v
     */
    @Deprecated
    private double arraysSimilarity(@NonNull INDArray array1, @NonNull INDArray array2) {
        if (array1.equals(array2))
            return 1.0;

        INDArray vector = Transforms.unitVec(array1);
        INDArray vector2 = Transforms.unitVec(array2);

        if (vector == null || vector2 == null)
            return -1;

        return Transforms.cosineSim(vector, vector2);

    }

    @Test
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testDirectInference(@TempDir Path testDir) throws Exception {
        boolean isIntegration = isIntegrationTests();
        File resource = Resources.asFile("/big/raw_sentences.txt");
        SentenceIterator sentencesIter = getIterator(isIntegration, resource);

        ClassPathResource resource_mixed = new ClassPathResource("paravec/");
        File local_resource_mixed = testDir.toFile();
        resource_mixed.copyDirectory(local_resource_mixed);
        SentenceIterator iter = new AggregatingSentenceIterator.Builder()
                .addSentenceIterator(sentencesIter)
                .addSentenceIterator(new FileSentenceIterator(local_resource_mixed)).build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec wordVectors = new Word2Vec.Builder().minWordFrequency(1).batchSize(250).iterations(1).epochs(1)
                .learningRate(0.025).layerSize(150).minLearningRate(0.001)
                .elementsLearningAlgorithm(new SkipGram<VocabWord>()).useHierarchicSoftmax(true).windowSize(5)
                .iterate(iter).tokenizerFactory(t).build();

        wordVectors.fit();

        ParagraphVectors pv = new ParagraphVectors.Builder().tokenizerFactory(t).iterations(10)
                .useHierarchicSoftmax(true).trainWordVectors(true).useExistingWordVectors(wordVectors)
                .negativeSample(0).sequenceLearningAlgorithm(new DM<VocabWord>()).build();

        INDArray vec1 = pv.inferVector("This text is pretty awesome");
        INDArray vec2 = pv.inferVector("Fantastic process of crazy things happening inside just for history purposes");

        log.info("vec1/vec2: {}", Transforms.cosineSim(vec1, vec2));
    }


    @Test()
    @Timeout(300000)
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testHash() {
        VocabWord w1 = new VocabWord(1.0, "D1");
        VocabWord w2 = new VocabWord(1.0, "Bo");



        log.info("W1 > Short hash: {}; Long hash: {}", w1.getLabel().hashCode(), w1.getStorageId());
        log.info("W2 > Short hash: {}; Long hash: {}", w2.getLabel().hashCode(), w2.getStorageId());

        assertNotEquals(w1.getStorageId(), w2.getStorageId());
    }


    /**
     * This is very long test, to track memory consumption over time
     *
     * @throws Exception
     */
    @Tag(TagNames.LONG_TEST)
    @ParameterizedTest
    @MethodSource("org.nd4j.linalg.BaseNd4jTestWithBackends#configs")
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    @Disabled("Takes too long for CI")
    @Tag(TagNames.NEEDS_VERIFY)
    public void testsParallelFit1(Nd4jBackend backend) throws Exception {
        final File file = Resources.asFile("big/raw_sentences.txt");

        for (int i = 0; i < 1000; i++) {
            List<Thread> threads = new ArrayList<>();
            for (int t = 0; t < 3; t++) {
                threads.add(new Thread(() -> {
                    try {
                        TokenizerFactory t1 = new DefaultTokenizerFactory();

                        LabelsSource source = new LabelsSource("DOC_");

                        SentenceIteratorConverter sic =
                                new SentenceIteratorConverter(new BasicLineIterator(file), source);

                        ParagraphVectors vec = new ParagraphVectors.Builder().seed(42)
                                //.batchSize(10)
                                .minWordFrequency(1).iterations(1).epochs(5).layerSize(100)
                                .learningRate(0.05)
                                //.labelsSource(source)
                                .windowSize(5).trainWordVectors(true).allowParallelTokenization(false)
                                //.vocabCache(cache)
                                .tokenizerFactory(t1).workers(1).iterate(sic).build();

                        vec.fit();
                    } catch (Exception e) {
                        throw new RuntimeException(e);
                    }
                }));
            }

            for (Thread t : threads) {
                t.start();
            }

            for (Thread t : threads) {
                t.join();
            }
        }
    }

    @Test()
    @Timeout(300000)
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testJSONSerialization() {
        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder().build();
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        val words = new VocabWord[3];
        words[0] = new VocabWord(1.0, "word");
        words[1] = new VocabWord(2.0, "test");
        words[2] = new VocabWord(3.0, "tester");

        for (int i = 0; i < words.length; ++i) {
            cache.addToken(words[i]);
            cache.addWordToIndex(i, words[i].getLabel());
        }
        paragraphVectors.setVocab(cache);

        String json = null;
        Word2Vec unserialized = null;
        try {
            json = paragraphVectors.toJson();
            log.info("{}", json.toString());

            unserialized = ParagraphVectors.fromJson(json);
        } catch (Exception e) {
            log.error("",e);
            fail();
        }

        assertEquals(cache.totalWordOccurrences(), ((ParagraphVectors) unserialized).getVocab().totalWordOccurrences());
        assertEquals(cache.totalNumberOfDocs(), ((ParagraphVectors) unserialized).getVocab().totalNumberOfDocs());

        for (int i = 0; i < words.length; ++i) {
            val cached = cache.wordAtIndex(i);
            val restored = ((ParagraphVectors) unserialized).getVocab().wordAtIndex(i);
            assertNotNull(cached);
            assertEquals(cached, restored);
        }
    }

    @Test()
    @Timeout(300000)
    @Tag(TagNames.LONG_TEST)
    @Tag(TagNames.LARGE_RESOURCES)
    public void testDoubleFit() throws Exception {
        boolean isIntegration = isIntegrationTests();
        File resource = Resources.asFile("/big/raw_sentences.txt");
        SentenceIterator iter = getIterator(isIntegration, resource);


        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");

        val builder = new ParagraphVectors.Builder();
        ParagraphVectors vec = builder.minWordFrequency(1).iterations(5).seed(119).epochs(1)
                .layerSize(150).learningRate(0.025).labelsSource(source).windowSize(5)
                .sequenceLearningAlgorithm(new DM<VocabWord>()).iterate(iter).trainWordVectors(true)
                .usePreciseWeightInit(true)
                .batchSize(8192)
                .allowParallelTokenization(false)
                .tokenizerFactory(t).workers(1).sampling(0).build();

        vec.fit();
        long num1 = vec.vocab().totalNumberOfDocs();

        vec.fit();
        System.out.println(vec.vocab().totalNumberOfDocs());
        long num2 = vec.vocab().totalNumberOfDocs();

        assertEquals(num1, num2);
    }

    public static SentenceIterator getIterator(boolean isIntegration, File file) throws IOException {
        return getIterator(isIntegration, file, 500);
    }

    public static SentenceIterator getIterator(boolean isIntegration, File file, int linesForUnitTest) throws IOException {
        if(isIntegration){
            return new BasicLineIterator(file);
        } else {
            List<String> lines = new ArrayList<>();
            try(InputStream is = new BufferedInputStream(new FileInputStream(file))){
                LineIterator lineIter = IOUtils.lineIterator(is, StandardCharsets.UTF_8);
                try{
                    for( int i=0; i<linesForUnitTest && lineIter.hasNext(); i++ ){
                        lines.add(lineIter.next());
                    }
                } finally {
                    lineIter.close();
                }
            }

            return new CollectionSentenceIterator(lines);
        }
    }
}


