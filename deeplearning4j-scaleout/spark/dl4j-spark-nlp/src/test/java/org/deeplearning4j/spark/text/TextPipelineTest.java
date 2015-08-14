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

package org.deeplearning4j.spark.text;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.berkeley.Counter;
import org.deeplearning4j.berkeley.Pair;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.junit.Before;
import org.junit.Test;

import java.io.IOException;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.concurrent.atomic.AtomicLong;

import static org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecVariables.*;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * @author Jeffrey Tang
 */
public class TextPipelineTest {

    private List<String> sentenceList;
    private SparkConf conf;


    public SparkConf getConfClone() {
        return conf.clone();
    }

    public JavaRDD<String> getCorpusRDD(JavaSparkContext sc) {
        return sc.parallelize(sentenceList);
    }

    @Before
    public void before() throws IOException {
        conf = new SparkConf().setMaster("local[4]")
                .setAppName("sparktest")
                .set(NUM_WORDS, String.valueOf(1))
                .set(N_GRAMS, String.valueOf(1))
                .set(TOKENIZER, "org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory")
                .set(TOKEN_PREPROCESSOR, "org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor")
                .set(REMOVE_STOPWORDS, String.valueOf(true));
        sentenceList = Arrays.asList("This is a strange strange world.", "Flowers are red.");
    }

    @Test
    public void testTokenizeFunction() throws Exception {
        String tokenizerString = (String) defaultVals.get(TOKENIZER);
        String tokenizerPreprocessString = (String) defaultVals.get(TOKEN_PREPROCESSOR);

        TokenizerFunction tokenizerFunction = new TokenizerFunction(tokenizerString, tokenizerPreprocessString, 1);

        List<String> tokenList = tokenizerFunction.call(sentenceList.get(0));

        assertTrue(tokenList.equals(Arrays.asList("this", "is", "a", "strange", "strange", "world")));
    }

    @Test
    public void testTokenizer() throws Exception {
        JavaSparkContext sc = new JavaSparkContext(getConfClone());
        JavaRDD<String> corpusRDD = getCorpusRDD(sc);

        TextPipeline pipeline = new TextPipeline(corpusRDD);
        JavaRDD<List<String>> tokenizedRDD = pipeline.tokenize();
        assertEquals(tokenizedRDD.first(), Arrays.asList("this", "is", "a", "strange", "strange", "world"));

        sc.stop();
    }

    @Test
    public void testWordFreqAccIdentifyStopWords() throws Exception {
        JavaSparkContext sc = new JavaSparkContext(getConfClone());
        JavaRDD<String> corpusRDD = getCorpusRDD(sc);

        TextPipeline pipeline = new TextPipeline(corpusRDD);
        JavaRDD<List<String>> tokenizedRDD = pipeline.tokenize();
        JavaRDD<Pair<List<String>, AtomicLong>> sentenceWordsCountRDD =
                pipeline.updateAndReturnAccumulatorVal(tokenizedRDD);

        Counter<String> wordFreqCounter = pipeline.getWordFreqAcc().value();
        assertEquals(wordFreqCounter.getCount("STOP"), 4, 0);
        assertEquals(wordFreqCounter.getCount("strange"), 2, 0);
        assertEquals(wordFreqCounter.getCount("flowers"), 1, 0);
        assertEquals(wordFreqCounter.getCount("world"), 1, 0);
        assertEquals(wordFreqCounter.getCount("red"), 1, 0);

        List<Pair<List<String>, AtomicLong>> ret = sentenceWordsCountRDD.collect();
        assertEquals(ret.get(0).getFirst(), Arrays.asList("this", "is", "a", "strange", "strange", "world"));
        assertEquals(ret.get(1).getFirst(), Arrays.asList("flowers", "are", "red"));
        assertEquals(ret.get(0).getSecond().get(), 6);
        assertEquals(ret.get(1).getSecond().get(), 3);


        sc.stop();
    }

    @Test
    public void testWordFreqAccNotIdentifyingStopWords() throws Exception {
        SparkConf confCopy = getConfClone().set(REMOVE_STOPWORDS, String.valueOf(false));
        JavaSparkContext sc = new JavaSparkContext(confCopy);
        JavaRDD<String> corpusRDD = getCorpusRDD(sc);

        TextPipeline pipeline = new TextPipeline(corpusRDD);
        JavaRDD<List<String>> tokenizedRDD = pipeline.tokenize();
        pipeline.updateAndReturnAccumulatorVal(tokenizedRDD);

        Counter<String> wordFreqCounter = pipeline.getWordFreqAcc().value();
        assertEquals(wordFreqCounter.getCount("is"), 1, 0);
        assertEquals(wordFreqCounter.getCount("this"), 1, 0);
        assertEquals(wordFreqCounter.getCount("are"), 1, 0);
        assertEquals(wordFreqCounter.getCount("a"), 1, 0);
        assertEquals(wordFreqCounter.getCount("strange"), 2, 0);
        assertEquals(wordFreqCounter.getCount("flowers"), 1, 0);
        assertEquals(wordFreqCounter.getCount("world"), 1, 0);
        assertEquals(wordFreqCounter.getCount("red"), 1, 0);

        sc.stop();
    }

    @Test
    public void testFilterMinWordAddVocab() throws Exception {
        JavaSparkContext sc = new JavaSparkContext(getConfClone());
        JavaRDD<String> corpusRDD = getCorpusRDD(sc);

        TextPipeline pipeline = new TextPipeline(corpusRDD);
        JavaRDD<List<String>> tokenizedRDD = pipeline.tokenize();
        pipeline.updateAndReturnAccumulatorVal(tokenizedRDD);
        Counter<String> wordFreqCounter = pipeline.getWordFreqAcc().value();

        pipeline.filterMinWordAddVocab(wordFreqCounter);
        VocabCache vocabCache = pipeline.getVocabCache();

        assertTrue(vocabCache != null);
        VocabWord redVocab = vocabCache.tokenFor("red");
        VocabWord flowerVocab = vocabCache.tokenFor("flowers");
        VocabWord worldVocab = vocabCache.tokenFor("world");
        VocabWord strangeVocab = vocabCache.tokenFor("strange");


        assertEquals(redVocab.getWord(), "red");
        assertEquals(redVocab.getIndex(), 0);
        assertEquals(redVocab.getWordFrequency(), 1, 0);

        assertEquals(flowerVocab.getWord(), "flowers");
        assertEquals(flowerVocab.getIndex(), 1);
        assertEquals(flowerVocab.getWordFrequency(), 1, 0);

        assertEquals(worldVocab.getWord(), "world");
        assertEquals(worldVocab.getIndex(), 2);
        assertEquals(worldVocab.getWordFrequency(), 1, 0);

        assertEquals(strangeVocab.getWord(), "strange");
        assertEquals(strangeVocab.getIndex(), 3);
        assertEquals(strangeVocab.getWordFrequency(), 2, 0);

        sc.stop();
    }

    @Test
    public void testBuildVocabCache() throws  Exception{
        JavaSparkContext sc = new JavaSparkContext(getConfClone());
        JavaRDD<String> corpusRDD = getCorpusRDD(sc);
        TextPipeline pipeline = new TextPipeline(corpusRDD);
        pipeline.buildVocabCache();

        VocabCache vocabCache = pipeline.getVocabCache();

        assertTrue(vocabCache != null);
        VocabWord redVocab = vocabCache.tokenFor("red");
        VocabWord flowerVocab = vocabCache.tokenFor("flowers");
        VocabWord worldVocab = vocabCache.tokenFor("world");
        VocabWord strangeVocab = vocabCache.tokenFor("strange");


        assertEquals(redVocab.getWord(), "red");
        assertEquals(redVocab.getIndex(), 0);
        assertEquals(redVocab.getWordFrequency(), 1, 0);

        assertEquals(flowerVocab.getWord(), "flowers");
        assertEquals(flowerVocab.getIndex(), 1);
        assertEquals(flowerVocab.getWordFrequency(), 1, 0);

        assertEquals(worldVocab.getWord(), "world");
        assertEquals(worldVocab.getIndex(), 2);
        assertEquals(worldVocab.getWordFrequency(), 1, 0);

        assertEquals(strangeVocab.getWord(), "strange");
        assertEquals(strangeVocab.getIndex(), 3);
        assertEquals(strangeVocab.getWordFrequency(), 2, 0);

        sc.stop();
    }

    @Test
    public void testBuildVocabWordListRDD() throws Exception{
        JavaSparkContext sc = new JavaSparkContext(getConfClone());
        JavaRDD<String> corpusRDD = getCorpusRDD(sc);
        TextPipeline pipeline = new TextPipeline(corpusRDD);
        pipeline.buildVocabCache();
        pipeline.buildVocabWordListRDD();
        JavaRDD<AtomicLong> sentenceCountRDD = pipeline.getSentenceCountRDD();
        JavaRDD<List<VocabWord>> vocabWordListRDD = pipeline.getVocabWordListRDD();
        List<List<VocabWord>> vocabWordList = vocabWordListRDD.collect();
        List<VocabWord> firstSentenceVocabList = vocabWordList.get(0);
        List<VocabWord> secondSentenceVocabList = vocabWordList.get(1);

        System.out.println(Arrays.deepToString(firstSentenceVocabList.toArray()));

        List<String> firstSentenceTokenList = new ArrayList<>();
        List<String> secondSentenceTokenList = new ArrayList<>();
        for (VocabWord v : firstSentenceVocabList) {
            if (v != null) {
                firstSentenceTokenList.add(v.getWord());
            }
        }
        for (VocabWord v : secondSentenceVocabList) {
            if (v != null) {
                secondSentenceTokenList.add(v.getWord());
            }
        }

        assertEquals(pipeline.getTotalWordCount(), 9, 0);
        assertEquals(sentenceCountRDD.collect().get(0).get(), 6);
        assertEquals(sentenceCountRDD.collect().get(1).get(), 3);
        assertTrue(firstSentenceTokenList.containsAll(Arrays.asList("strange", "strange", "world")));
        assertTrue(secondSentenceTokenList.containsAll(Arrays.asList("flowers", "red")));

        sc.stop();
     }


//    /**
//     *
//     * Testing the TextPipeline on a toy example to make sure words and vocabs are counted correctly
//     *
//     * @throws Exception
//     */
//    @Test
//    public void testTextPipelineSimple() throws Exception {
//        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("basic/word2vec.txt").getFile().getAbsolutePath(), 3);
//        TextPipeline pipeline = new TextPipeline(corpus, 1); //Min word freq
//        Pair<VocabCache,Long> pair = pipeline.process();
//        InMemoryLookupCache lookupCache = (InMemoryLookupCache)pair.getFirst();
//
//        // vocabWord count
//        assertEquals(lookupCache.vocabs.size(), 6);
//        // Check vocabWord Index
//        ArrayList<Integer> vocabWordInds = new ArrayList<>();
//        for (VocabWord vw : lookupCache.vocabs.values()) {
//            vocabWordInds.add(vw.getIndex());
//        }
//        assertTrue(Collections.min(vocabWordInds) == 0);
//        assertTrue(Collections.max(vocabWordInds) == 5);
//        // Check vocabWord themselves
//        // STOP and UNK are not added to
//        assertTrue(lookupCache.vocabs.containsKey("She"));
//        assertTrue(lookupCache.vocabs.containsKey("found"));
//        assertTrue(lookupCache.vocabs.containsKey("one"));
//        assertTrue(lookupCache.vocabs.containsKey("two"));
//        assertTrue(lookupCache.vocabs.containsKey("ba"));
//        assertTrue(lookupCache.vocabs.containsKey("abab"));
//        // Check total word count
//        assertTrue(pair.getSecond() == 8);
//    }
//
//    /**
//     *
//     * Testing the TextPipeline on a bigger corpus to make sure words and vocabs are counted correctly
//     *
//     * @throws Exception
//     */
//    @Test
//    public void testTextPipelineFull() throws Exception {
//        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath(), 3);
//        TextPipeline pipeline = new TextPipeline(corpus, 1); //Min word freq
//        Pair<VocabCache, Long> pair = pipeline.process();
//        InMemoryLookupCache lookupCache = (InMemoryLookupCache) pair.getFirst();
//
//        assertEquals(lookupCache.vocabs.size(), 542);
//        assertEquals(lookupCache.vocabs.get("SHOULD").getIndex(), 5);
//        assertEquals((int)lookupCache.wordFrequencies.getCount("SHOULD"), 4);
//    }
//
//    /**
//     * Test Word2Vec on a toy dataset to make sure things at least run and check if words and vocab are
//     * counted properly after Word2Vec is trained
//     *
//     * @throws Exception
//     */
//    @Test
//    public void testWord2VecSimple() throws Exception {
//        // Train Word2Vec
//        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("basic/word2vec.txt").getFile().getAbsolutePath(), 3);
//        Word2Vec model = new Word2Vec();
//        model.train(corpus);
//
//        // Test lookupcache
//        InMemoryLookupCache lookupCache = (InMemoryLookupCache)model.getVocabCacheBroadcast().value();
//        // vocabWord count
//        assertEquals(lookupCache.vocabs.size(), 7);
//        // Check vocabWord Index
//        ArrayList<Integer> vocabWordInds = new ArrayList<>();
//        for (VocabWord vw : lookupCache.vocabs.values()) {
//            vocabWordInds.add(vw.getIndex());
//        }
//        assertTrue(Collections.min(vocabWordInds) == 1);
//        assertTrue(Collections.max(vocabWordInds) == 7);
//        // Check vocabWord themselves
//        assertTrue(lookupCache.vocabs.containsKey("She"));
//        assertTrue(lookupCache.vocabs.containsKey("found"));
//        assertTrue(lookupCache.vocabs.containsKey("STOP"));
//        assertTrue(lookupCache.vocabs.containsKey("one"));
//        assertTrue(lookupCache.vocabs.containsKey("two"));
//        assertTrue(lookupCache.vocabs.containsKey("ba"));
//        assertTrue(lookupCache.vocabs.containsKey("abab"));
//        // Check word frequencies
//        assertTrue(lookupCache.wordFrequencies.getCount("She") == 1.0);
//        assertTrue(lookupCache.wordFrequencies.getCount("found") == 1.0);
//        assertTrue(lookupCache.wordFrequencies.getCount("STOP") == 2.0);
//        assertTrue(lookupCache.wordFrequencies.getCount("one") == 1.0);
//        assertTrue(lookupCache.wordFrequencies.getCount("two") == 1.0);
//        assertTrue(lookupCache.wordFrequencies.getCount("ba") == 1.0);
//        assertTrue(lookupCache.wordFrequencies.getCount("abab") == 1.0);
//    }
//
//    /**
//     *
//     * Run Word2Vec to a bigger corpus to make sure the results are sensible
//     *
//     * @throws Exception
//     */
//    @Test
//    public void testWord2VecFull() throws Exception {
//        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath());
//        Word2Vec model = new Word2Vec();
//        model.train(corpus);
//        InMemoryLookupCache lookupCache = (InMemoryLookupCache)model.getVocabCacheBroadcast().value();
//        Collection<String> lst = model.wordsNearest("day", 10);
//
//    }

}

