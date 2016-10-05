/*
 *
 *  * Copyright 2015 Skymind,Inc.
 *  *
 *  *    Licensed under the Apache License, Version 2.0 (the "License");
 *  *    you may not use this file except in compliance with the License.
 *  *    You may obtain a copy of the License at
 *  *
 *  *        http://www.apache.org/licenses/LICENSE-2.0
 *  *
 *  *    Unless required by applicable law or agreed to in writing, software
 *  *    distributed under the License is distributed on an "AS IS" BASIS,
 *  *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  *    See the License for the specific language governing permissions and
 *  *    limitations under the License.
 *
 */

package org.deeplearning4j.models.paragraphvectors;


import lombok.NonNull;
import org.datavec.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.learning.impl.elements.SkipGram;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DBOW;
import org.deeplearning4j.models.embeddings.learning.impl.sequence.DM;
import org.deeplearning4j.models.embeddings.loader.VectorsConfiguration;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.AbstractCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.FileLabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelAwareIterator;
import org.deeplearning4j.text.documentiterator.LabelledDocument;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.AggregatingSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.FileSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.util.SerializationUtils;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collection;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by agibsonccc on 12/3/14.
 */
public class ParagraphVectorsTest {
    private static final Logger log = LoggerFactory.getLogger(ParagraphVectorsTest.class);

    @Before
    public void before() {
        new File("word2vec-index").delete();
    }


/*
    @Test
    public void testWord2VecRunThroughVectors() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile().getParentFile();
        LabelAwareSentenceIterator iter = LabelAwareUimaSentenceIterator.createWithPath(file.getAbsolutePath());


        TokenizerFactory t = new UimaTokenizerFactory();


        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1).iterations(5).labels(Arrays.asList("label1", "deeple"))
                .layerSize(100)
                .stopWords(new ArrayList<String>())
                .windowSize(5).iterate(iter).tokenizerFactory(t).build();

        assertEquals(new ArrayList<String>(), vec.getStopWords());


        vec.fit();
        double sim = vec.similarity("day","night");
        log.info("day/night similarity: " + sim);
        new File("cache.ser").delete();

    }
*/

    /**
     * This test checks, how vocab is built using SentenceIterator provided, without labels.
     *
     * @throws Exception
     */
    @Test
    public void testParagraphVectorsVocabBuilding1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();//.getParentFile();
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

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1).iterations(5)
                .layerSize(100)
          //      .labelsGenerator(source)
                .windowSize(5)
                .iterate(iter)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .build();

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

    @Test
    public void testParagraphVectorsModelling1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

        InMemoryLookupCache cache = new InMemoryLookupCache(false);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(3)
                .epochs(1)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .elementsLearningAlgorithm(new SkipGram<VocabWord>())
                .sequenceLearningAlgorithm(new DBOW<VocabWord>())
                .iterate(iter)
                .trainWordVectors(true)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build();

        vec.fit();

        File fullFile = File.createTempFile("paravec", "tests");
        fullFile.deleteOnExit();

        WordVectorSerializer.writeParagraphVectors(vec, fullFile);

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
        for(String label: result) {
            System.out.println(label + "/DOC_16392: " + vec.similarity(label, "DOC_16392"));
        }
        assertTrue(result.contains("DOC_16392"));
        assertTrue(result.contains("DOC_21383"));



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

        WordVectorSerializer.writeWordVectors(vec, tempFile2);

        ParagraphVectors vec3 = WordVectorSerializer.readParagraphVectorsFromText(tempFile2);

        INDArray day3 = vec3.getWordVectorMatrix("day").dup();

        List<String> labelsRestored = vec3.labelsSource.getLabels();

        assertEquals(day, day3);

        assertEquals(labelsOriginal.size(), labelsRestored.size());

           /*
            Testing binary serialization
         */
        SerializationUtils.saveObject(vec, tempFile);


        ParagraphVectors vec2 = (ParagraphVectors) SerializationUtils.readObject(tempFile);
        INDArray day2 = vec2.getWordVectorMatrix("day").dup();

        List<String> labelsBinary = vec2.labelsSource.getLabels();

        assertEquals(day, day2);

        tempFile.delete();


        assertEquals(labelsOriginal.size(), labelsBinary.size());

        INDArray original = vec.getWordVectorMatrix("DOC_16392").dup();
        INDArray inferredA1 = vec.inferVector("This is my world .");
        INDArray inferredB1 = vec.inferVector("This is my world .");

        double cosAO1 = Transforms.cosineSim(inferredA1.dup(), original.dup());
        double cosAB1 = Transforms.cosineSim(inferredA1.dup(), inferredB1.dup());

        log.info("Cos O/A: {}", cosAO1);
        log.info("Cos A/B: {}", cosAB1);
        assertTrue(cosAO1 > 0.7);
        assertTrue(cosAB1 > 0.95);

        //assertArrayEquals(inferredA.data().asDouble(), inferredB.data().asDouble(), 0.01);

        ParagraphVectors restoredVectors = WordVectorSerializer.readParagraphVectors(fullFile);
        restoredVectors.setTokenizerFactory(t);

        INDArray inferredA2 = restoredVectors.inferVector("This is my world .");
        INDArray inferredB2 = restoredVectors.inferVector("This is my world .");
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

        assertTrue(cosAO2 > 0.7);
        assertTrue(cosAB2 > 0.95);
        assertTrue(cosAAX > 0.95);
    }


    @Test
    public void testParagraphVectorsDM() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

//        InMemoryLookupCache cache = new InMemoryLookupCache(false);
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(3)
                .epochs(1)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(true)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .negativeSample(0)
                .sampling(0)
                .sequenceLearningAlgorithm(new DM<VocabWord>())
                .build();

        vec.fit();


        int cnt1 = cache.wordFrequency("day");
        int cnt2 = cache.wordFrequency("me");

        assertNotEquals(1, cnt1);
        assertNotEquals(1, cnt2);
        assertNotEquals(cnt1, cnt2);

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

    }


    @Test
    public void testParagraphVectorsWithWordVectorsModelling1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

//        InMemoryLookupCache cache = new InMemoryLookupCache(false);
        AbstractCache<VocabWord> cache = new AbstractCache.Builder<VocabWord>().build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        LabelsSource source = new LabelsSource("DOC_");

        ParagraphVectors vec = new ParagraphVectors.Builder()
                .minWordFrequency(1)
                .iterations(3)
                .epochs(1)
                .layerSize(100)
                .learningRate(0.025)
                .labelsSource(source)
                .windowSize(5)
                .iterate(iter)
                .trainWordVectors(true)
                .vocabCache(cache)
                .tokenizerFactory(t)
                .sampling(0)
                .build();

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
    @Ignore
    public void testParagraphVectorsReducedLabels1() throws Exception {
        ClassPathResource resource = new ClassPathResource("/labeled");
        File file = resource.getFile();

        LabelAwareIterator iter = new FileLabelAwareIterator.Builder()
                .addSourceFolder(file)
                .build();

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
                .windowSize(5)
                .iterate(iter)
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


    /*
        In this test we'll build w2v model, and will use it's vocab and weights for ParagraphVectors.
        there's no need in this test within travis, use it manually only for problems detection
    */
    @Test
    @Ignore
    public void testParagraphVectorsOverExistingWordVectorsModel() throws Exception {


        // we build w2v from multiple sources, to cover everything
        ClassPathResource resource_sentences = new ClassPathResource("/big/raw_sentences.txt");
        ClassPathResource resource_mixed = new ClassPathResource("/paravec");
        SentenceIterator iter = new AggregatingSentenceIterator.Builder()
                .addSentenceIterator(new BasicLineIterator(resource_sentences.getFile()))
                .addSentenceIterator(new FileSentenceIterator(resource_mixed.getFile()))
                .build();

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec wordVectors = new Word2Vec.Builder()
                .minWordFrequency(1)
                .batchSize(250)
                .iterations(3)
                .epochs(1)
                .learningRate(0.025)
                .layerSize(150)
                .minLearningRate(0.001)
                .windowSize(5)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        wordVectors.fit();

        INDArray vector_day1 = wordVectors.getWordVectorMatrix("day").dup();

        // At this moment we have ready w2v model. It's time to use it for ParagraphVectors

        FileLabelAwareIterator labelAwareIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(new ClassPathResource("/paravec/labeled").getFile())
                .build();


        // documents from this iterator will be used for classification
        FileLabelAwareIterator unlabeledIterator = new FileLabelAwareIterator.Builder()
                .addSourceFolder(new ClassPathResource("/paravec/unlabeled").getFile())
                .build();


        // we're building classifier now, with pre-built w2v model passed in
        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .iterate(labelAwareIterator)
                .learningRate(0.025)
                .minLearningRate(0.001)
                .iterations(1)
                .epochs(10)
                .layerSize(150)
                .tokenizerFactory(t)
                .trainWordVectors(true)
                .useExistingWordVectors(wordVectors)
                .build();

        paragraphVectors.fit();


        /*
        double similarityD = wordVectors.similarity("day", "night");
        log.info("day/night similarity: " + similarityD);
        assertTrue(similarityD > 0.5d);
        */

        INDArray vector_day2 = paragraphVectors.getWordVectorMatrix("day").dup();
        double crossDay = arraysSimilarity(vector_day1, vector_day2);
/*
        log.info("Day1: " + vector_day1);
        log.info("Day2: " + vector_day2);
        log.info("Cross-Day similarity: " + crossDay);
        log.info("Cross-Day similiarity 2: " + Transforms.cosineSim(vector_day1, vector_day2));

        assertTrue(crossDay > 0.9d);
*/
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

        LabelledDocument document = unlabeledIterator.nextDocument();

        log.info("Results for document '" + document.getLabel() + "'");

        List<String> results = new ArrayList<>(paragraphVectors.predictSeveral(document, 3));
        for (String result: results) {
            double sim = paragraphVectors.similarityToLabel(document, result);
            log.info("Similarity to ["+result+"] is ["+ sim +"]");
        }

        String topPrediction = paragraphVectors.predict(document);
        assertEquals("Zhealth", topPrediction);
    }

    /*
        Left as reference implementation, before stuff was changed in w2v
     */
    @Deprecated
    private double arraysSimilarity(@NonNull INDArray array1,@NonNull INDArray array2) {
        if (array1.equals(array2)) return 1.0;

        INDArray vector = Transforms.unitVec(array1);
        INDArray vector2 = Transforms.unitVec(array2);

        if(vector == null || vector2 == null)
            return -1;

        return  Nd4j.getBlasWrapper().dot(vector, vector2);

    }

    /**
     * Special test to check d2v inference against pre-trained gensim model and
     */
    @Ignore
    @Test
    public void testGensimEquality() throws Exception {

        INDArray expB = Nd4j.create(new double[]{-0.02465764,  0.00756337, -0.0268607 ,  0.01588023,  0.01580242, -0.00150542,  0.00116652,  0.0021577 , -0.00754891, -0.02441176, -0.01271976, -0.02015191,  0.00220599,  0.03722657, -0.01629612, -0.02779619, -0.01157856, -0.01937938, -0.00744667,  0.01990043, -0.00505888,  0.00573646,  0.00385467, -0.0282531 ,  0.03484593, -0.05528606,  0.02428633, -0.01510474,  0.00153177, -0.03637344, 0.01747423, -0.00090738, -0.02199888,  0.01410434, -0.01710641, -0.01446697, -0.04225266,  0.00262217,  0.00871943,  0.00471594, 0.0101348 , -0.01991908,  0.00874325, -0.00606416, -0.01035323, -0.01376545,  0.00451507, -0.01220307, -0.04361237,  0.00026028, -0.02401881,  0.00580314,  0.00238946, -0.01325974,  0.01879044, -0.00335623, -0.01631887,  0.02222102, -0.02998703,  0.03190075, -0.01675236, -0.01799807, -0.01314015,  0.01950069,  0.0011723 , 0.01013178,  0.01093296, -0.034143  ,  0.00420227,  0.01449351, -0.00629987,  0.01652851, -0.01286825,  0.03314656,  0.03485073, 0.01120341,  0.01298241,  0.0019494 , -0.02420256, -0.0063762, 0.01527091, -0.00732881,  0.0060427 ,  0.019327  , -0.02068196, 0.00876712,  0.00292274,  0.01312969, -0.01529114,  0.0021757 , -0.00565621, -0.01093122,  0.02758765, -0.01342688,  0.01606117, -0.02666447,  0.00541112,  0.00375426, -0.00761796,  0.00136015, -0.01169962, -0.03012749,  0.03012953, -0.05491332, -0.01137303, -0.01392103,  0.01370098, -0.00794501,  0.0248435 ,  0.00319645, 0.04261713, -0.00364211,  0.00780485,  0.01182583, -0.00647098, 0.03291231, -0.02515565,  0.03480943,  0.00119836, -0.00490694, 0.02615346, -0.00152456,  0.00196142, -0.02326461,  0.00603225, -0.02414703, -0.02540966,  0.0072112 , -0.01090273, -0.00505061, -0.02196866,  0.00515245,  0.04981546, -0.02237269, -0.00189305, 0.0169786 ,  0.01782372, -0.00430022,  0.00551226,  0.00293861, -0.01337168, -0.00302476, -0.01869966,  0.00270757,  0.03199976, -0.01614617, -0.02716484,  0.01560035, -0.01312686, -0.01604082, 0.01347521,  0.03229654,  0.00707219, -0.00588392,  0.02444809, -0.01068742, -0.0190814 , -0.00556385, -0.00462766,  0.01283929, 0.02001247, -0.00837629, -0.00041943, -0.02298774,  0.00874839, 0.00434907, -0.00963332,  0.00476905,  0.00793049, -0.00212557, -0.01839353,  0.03345517,  0.00838255, -0.0157447 , -0.0376134 , 0.01059611, -0.02323246, -0.01326356, -0.01116734,  0.00598869, 0.0211626 ,  0.01872963, -0.0038276 , -0.01208279, -0.00989125, 0.04147648,  0.00181867, -0.00369355,  0.02312465,  0.0048396 , 0.00564515,  0.01317832, -0.0057621 , -0.01882041, -0.02869064, -0.00670661,  0.02585443, -0.01108428,  0.01411031,  0.01204507, -0.01244726, -0.00962342, -0.00205239, -0.01653971,  0.02871559, -0.00772978,  0.0214524 ,  0.02035478, -0.01324312,  0.00169302, -0.00064739,  0.00531795,  0.01059279, -0.02455794, -0.00002782, -0.0068906 , -0.0160858 , -0.0031842 , -0.02295724,  0.01481094, 0.01769004, -0.02925742,  0.02050495, -0.00029003, -0.02815636, 0.02467367,  0.03419458,  0.00654938, -0.01847546,  0.00999932, 0.00059222, -0.01722176,  0.05172159, -0.01548486,  0.01746444, 0.007871  ,  0.0078471 , -0.02414417,  0.01898077, -0.01470176, -0.00299465,  0.00368212, -0.02474656,  0.01317451,  0.03706085, -0.00032923,  0.02655881,  0.0013586 , -0.0120303 , -0.05030316, 0.0222294 , -0.0070967 , -0.02150935,  0.03254268,  0.01369857, 0.00246183, -0.02253576, -0.00551247,  0.00787363,  0.01215617, 0.02439827, -0.01104699, -0.00774596, -0.01898127, -0.01407653, 0.00195514, -0.03466602,  0.01560903, -0.01239944, -0.02474852, 0.00155114,  0.00089324, -0.01725949, -0.00011816,  0.00742845, 0.01247074, -0.02467943, -0.00679623,  0.01988366, -0.00626181, -0.02396477,  0.01052101, -0.01123178, -0.00386291, -0.00349261, -0.02714747, -0.00563315,  0.00228767, -0.01303677, -0.01971108, 0.00014759, -0.00346399,  0.02220698,  0.01979946, -0.00526076, 0.00647453,  0.01428513,  0.00223467, -0.01690172, -0.0081715 });

        VectorsConfiguration configuration = new VectorsConfiguration();
        Word2Vec w2v = WordVectorSerializer.readWord2VecFromText(
                new File("/home/raver119/Downloads/gensim_models_for_dl4j/word"),
                new File("/home/raver119/Downloads/gensim_models_for_dl4j/hs"),
                new File("/home/raver119/Downloads/gensim_models_for_dl4j/hs_code"),
                new File("/home/raver119/Downloads/gensim_models_for_dl4j/hs_mapping"),
                configuration
        );


        assertNotEquals(null, w2v.getLookupTable());
        assertNotEquals(null, w2v.getVocab());

        ParagraphVectors d2v = new ParagraphVectors.Builder(configuration)
                                                        .useExistingWordVectors(w2v)
                                                        .sequenceLearningAlgorithm(new DM<VocabWord>())
                                                        .tokenizerFactory(new DefaultTokenizerFactory())
                                                        .resetModel(false)
                                                        .build();


        assertNotEquals(null, d2v.getLookupTable());
        assertNotEquals(null, d2v.getVocab());

        assertTrue(d2v.getVocab() == w2v.getVocab());
        assertTrue(d2v.getLookupTable() == w2v.getLookupTable());

        String textA = "Donald Trump referred to President Obama as “your president” during the first presidential debate on Monday, much to many people’s chagrin on social media.\n"
                + "\n"
                + "Trump, made the reference after saying that the greatest threat facing the world is nuclear weapons. He then turned to Hillary Clinton and said, “Not global warming like you think and your President thinks,” referring to Obama.";

        String textB = "The comment followed Trump doubling down on his false claims about the so-called birther conspiracy theory about Obama. People following the debate were immediately angered that Trump implied Obama is not his president.";


        INDArray arrayA = d2v.inferVector(textA);
        INDArray arrayB = d2v.inferVector(textB);

        assertNotEquals(null, arrayA);
        assertNotEquals(null, arrayB);

        double simB = Transforms.cosineSim(arrayB, expB);

        log.info("SimilarityB: {}", simB);
    }
}
