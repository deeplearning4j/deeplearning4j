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


import org.deeplearning4j.models.word2vec.Word2Vec;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.models.word2vec.wordstore.inmemory.InMemoryLookupCache;
import org.deeplearning4j.text.documentiterator.LabelsSource;
import org.deeplearning4j.text.sentenceiterator.BasicLineIterator;
import org.deeplearning4j.text.sentenceiterator.LineSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.SentenceIterator;
import org.deeplearning4j.text.sentenceiterator.UimaSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareSentenceIterator;
import org.deeplearning4j.text.sentenceiterator.labelaware.LabelAwareUimaSentenceIterator;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.UimaTokenizerFactory;
import org.junit.Before;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.core.io.ClassPathResource;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;
import static org.junit.Assert.assertTrue;
import static org.junit.Assume.*;

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
                .iterate(iter)
                .trainWordVectors(false)
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

        assertEquals(97406, cache.numWords());

        assertTrue(vec.hasWord("DOC_16392"));
        assertTrue(vec.hasWord("DOC_3720"));

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
    }

    @Test
    public void testParagraphVectorsWithWordVectorsModelling1() throws Exception {
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
    }

    /*
        In this test we'll build w2v model, and will use it's vocab and weights for ParagraphVectors.
        IS NOT READY YET
    */
    @Test
    public void testParagraphVectorsOverExistingWordVectorsModel() throws Exception {
        /*
        ClassPathResource resource = new ClassPathResource("/big/raw_sentences.txt");
        File file = resource.getFile();
        SentenceIterator iter = new BasicLineIterator(file);

        InMemoryLookupCache cache = new InMemoryLookupCache(false);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec wordVectors = new Word2Vec.Builder()
                .minWordFrequency(1)
                .iterations(1)
                .epochs(1)
                .iterate(iter)
                .tokenizerFactory(t)
                .build();

        wordVectors.fit();

        INDArray vector_day1 = wordVectors.getWordVectorMatrix("day");
        double similarityD = wordVectors.similarity("day", "night");
        log.info("day/night similarity: " + similarityD);
        assertTrue(similarityD > 0.65d);

        // At this moment we have ready w2v model. It's time to use it for ParagraphVectors

        ParagraphVectors paragraphVectors = new ParagraphVectors.Builder()
                .iterate(iter)
                .iterations(1)
                .epochs(1)
                .tokenizerFactory(t)
                .trainWordVectors(false)
                .wordVectorsModel(wordVectors)
                .labelsTemplate("SNT_%d")
                .build();

        paragraphVectors.fit();
        INDArray vector_day2 = paragraphVectors.getWordVectorMatrix("day");
        double crossDay = arraysSimilarity(vector_day1, vector_day2);

        log.info("Day1: " + vector_day1);
        log.info("Day2: " + vector_day2);
        log.info("Cross-Day similarity: " + crossDay);

        assertTrue(crossDay > 0.9d);
        */
    }

    private double arraysSimilarity(INDArray array1, INDArray array2) {
        if (array1.equals(array2)) return 1.0;

        INDArray vector = Transforms.unitVec(array1);
        INDArray vector2 = Transforms.unitVec(array2);
        if(vector == null || vector2 == null)
            return -1;
        return  Nd4j.getBlasWrapper().dot(vector, vector2);

    }

    /*
        This is specific test to address cosineSim returning values < 0
     */
    @Ignore
    @Test
    public void testCosineSim() {
        Nd4j.dtype = DataBuffer.Type.DOUBLE;

        INDArray vec1 = Nd4j.create(new double[]{0.003980884328484535, 0.004026215523481369, -0.002371709095314145, 0.0010937106562778354, 0.0037080496549606323, 0.0023899234365671873, -0.0018686642870306969, -0.0030053425580263138, 0.001236464362591505, 9.734630293678492E-5, -0.0021808871533721685, -0.0020659095607697964, 0.002859584055840969, -0.0011606198968365788, -0.0038343463093042374, -0.0012979545863345265, -0.004094329662621021, -6.616020109504461E-4, 0.003920151386409998, -0.004590961150825024, 0.0033251738641411066, -3.6827087751589715E-4, -3.8962066173553467E-4, 0.004977192729711533, 0.0030343933030962944, -0.004943990148603916, -0.0022036072332412004, 0.0026519971434026957, -0.004640073049813509, -0.0014640631852671504, -0.0016898266039788723, 0.0011454367777332664, 9.043508907780051E-4, 0.0032472468446940184, 0.0037935907021164894, 0.0029152887873351574, -0.001198892598040402, 4.49799292255193E-4, 0.004544049501419067, 0.003468784037977457, -0.0029381709173321724, -0.00295071629807353, 0.004341112449765205, -0.004342120606452227, 0.004364471882581711, 0.0026899611111730337, -0.0011863819090649486, -0.004878778476268053, 8.598554413765669E-4, 0.002314662327989936, -0.003850248409435153, -4.904219531454146E-4, -0.0019613897893577814, -0.0043832045048475266, -8.378771017305553E-4, -0.0014581093564629555, -0.0022472806740552187, 0.0032676178961992264, 0.0033953296951949596, -8.839166257530451E-4, -0.002930713351815939, 2.698141324799508E-4, 0.0018378662643954158, -4.3097735033370554E-4, -6.99130876455456E-4, -0.0014571232022717595, -0.0022388570941984653, -0.0019292819779366255, -0.002768272068351507, -0.0037709292955696583, 6.755036301910877E-4, -0.003722618566825986, -0.0021332211326807737, -0.0026716203428804874, 3.8120447425171733E-4, 0.0037821209989488125, -0.0013625153806060553, -0.003543735248968005, 0.0048402040265500546, 0.0049340021796524525, 0.004159070085734129, -0.0018142855260521173, -0.00230741361156106, -0.0029703439213335514, 0.004570094868540764, -0.004900769796222448, -8.555767126381397E-4, -6.330466130748391E-4, 0.0013538813218474388, 4.1678131674416363E-4, -0.0041860006749629974, -0.001024525729008019, 0.004634086973965168, 7.914912421256304E-4, 0.0019921553321182728, -0.004206840880215168, 0.002674715593457222, 0.003861530451104045, -1.8840283155441284E-4, -0.0027044592425227165});
        INDArray vec2 = Nd4j.create(new double[]{-0.004710178356617689, -0.0016639563255012035, 0.001614249893464148, 0.004550485406070948, 1.025587334879674E-4, 0.004438179079443216, 0.003850518958643079, -0.0014919099630787969, 0.0021361345425248146, 0.004757246468216181, 0.004916144534945488, -0.0013914993032813072, -0.0026037085335701704, 0.004360968247056007, 0.0033731074072420597, 2.3143172438722104E-4, -0.003948550671339035, 0.002746849087998271, -4.448062099982053E-4, -0.004125682637095451, 0.0018341654213145375, 0.0028860217425972223, -0.0015598050085827708, -0.0019836118444800377, -0.003761543659493327, -8.517122478224337E-4, -6.945389322936535E-4, 0.0011574954260140657, 0.0029957324732095003, 0.0038646620232611895, -6.238603382371366E-4, -0.004308169241994619, -0.0017946800217032433, -0.00426481245085597, -0.0019806099589914083, -0.004050673451274633, 7.484317029593512E-5, 6.393367075361311E-4, 0.0026820702478289604, -0.00199546804651618, -1.8262595403939486E-4, -3.5763889900408685E-4, -0.004338039550930262, 6.385201122611761E-4, 0.003947985824197531, -0.003883859608322382, -0.0013746136100962758, -0.0016976085025817156, 0.0014504576101899147, -0.0022944719530642033, 0.0037672948092222214, -0.0047386703081429005, -0.00176273996476084, -0.004887159913778305, 0.004951943177729845, 0.0024230850394815207, -0.004242335446178913, 0.0034522039350122213, 0.004227067809551954, 0.004995960742235184, 7.847768138162792E-4, -0.001968711381778121, -7.779946899972856E-4, 0.0014542710268869996, 5.344665260054171E-4, 0.00292672635987401, -0.002540807705372572, 6.172555731609464E-4, 0.002198762260377407, 0.003972230013459921, 0.004826603922992945, 0.004627300426363945, -0.0025357455015182495, 0.004285604227334261, 0.0048736464232206345, 7.388484664261341E-4, 0.0041493638418614864, 2.751261054072529E-4, 0.001197249861434102, -0.0018036726396530867, -0.003288744017481804, -0.0011432680767029524, -7.104897522367537E-4, 7.275205571204424E-4, -0.004354830831289291, -2.1340310922823846E-4, -0.0042639062739908695, 0.002083777217194438, -0.004598635248839855, 0.0013592117466032505, 0.00473822234198451, 0.0022358030546456575, -0.0048327939584851265, 4.6066820505075157E-4, 0.004781856667250395, -0.004700486548244953, -4.069915448781103E-4, -0.004120219964534044, 8.657992002554238E-4, -0.0027251881547272205});

        double sim = Transforms.cosineSim(vec1, vec2);
        assertEquals(0, sim, 1e-1);
    }
}
