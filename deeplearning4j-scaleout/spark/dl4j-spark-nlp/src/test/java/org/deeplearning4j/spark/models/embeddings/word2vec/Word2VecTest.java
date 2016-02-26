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

package org.deeplearning4j.spark.models.embeddings.word2vec;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.canova.api.util.ClassPathResource;
import org.deeplearning4j.models.embeddings.inmemory.InMemoryLookupTable;
import org.deeplearning4j.models.embeddings.loader.WordVectorSerializer;
import org.deeplearning4j.models.embeddings.reader.impl.BasicModelUtils;
import org.deeplearning4j.models.embeddings.reader.impl.FlatModelUtils;
import org.deeplearning4j.models.embeddings.wordvectors.WordVectors;
import org.deeplearning4j.models.word2vec.VocabWord;
import org.deeplearning4j.models.word2vec.wordstore.VocabCache;
import org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor;
import org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory;
import org.deeplearning4j.text.tokenization.tokenizerfactory.TokenizerFactory;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.io.File;
import java.util.Arrays;
import java.util.Collection;

import static org.junit.Assert.*;

/**
 * @author jeffreytang
 */
public class Word2VecTest {

    @Test
    public void testConcepts() throws Exception {
        // These are all default values for word2vec
       SparkConf sparkConf = new SparkConf().setMaster("local[8]").setAppName("sparktest");

        // Set SparkContext
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // Path of data part-00000
        String dataPath = new ClassPathResource("/big/raw_sentences.txt").getFile().getAbsolutePath();
        dataPath = "/ext/Temp/part-00000";
//        String dataPath = new ClassPathResource("spark_word2vec_test.txt").getFile().getAbsolutePath();

        // Read in data
        JavaRDD<String> corpus = sc.textFile(dataPath);

        TokenizerFactory t = new DefaultTokenizerFactory();
        t.setTokenPreProcessor(new CommonPreprocessor());

        Word2Vec word2Vec = new Word2Vec.Builder()
                .setNGrams(1)
           //     .setTokenizer("org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory")
           //     .setTokenPreprocessor("org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor")
           //     .setRemoveStop(false)
                .tokenizerFactory(t)
                .seed(42L)
                .negative(10)
                .useAdaGrad(false)
                .layerSize(150)
                .windowSize(5)
                .learningRate(0.025)
                .minLearningRate(0.0001)
                .iterations(1)
                .batchSize(100)
                .minWordFrequency(5)
                .stopWords(Arrays.asList("three"))
                .useUnknown(true)
                .build();

        word2Vec.train(corpus);

        //word2Vec.setModelUtils(new FlatModelUtils());

        System.out.println("UNK: " + word2Vec.getWordVectorMatrix("UNK"));

        InMemoryLookupTable<VocabWord> table = (InMemoryLookupTable<VocabWord>) word2Vec.lookupTable();

        double sim = word2Vec.similarity("day", "night");
        System.out.println("day/night similarity: " + sim);
/*
        System.out.println("Hornjo: " + word2Vec.getWordVectorMatrix("hornjoserbsce"));
        System.out.println("carro: " + word2Vec.getWordVectorMatrix("carro"));

        Collection<String> portu = word2Vec.wordsNearest("carro", 10);
        printWords("carro", portu, word2Vec);

        portu = word2Vec.wordsNearest("davi", 10);
        printWords("davi", portu, word2Vec);

        System.out.println("---------------------------------------");
        */

        Collection<String> words = word2Vec.wordsNearest("day", 10);
        printWords("day", words, word2Vec);

        assertTrue(words.contains("night"));
        assertTrue(words.contains("week"));
        assertTrue(words.contains("year"));

        sim = word2Vec.similarity("two", "four");
        System.out.println("two/four similarity: " + sim);

        words = word2Vec.wordsNearest("two", 10);
        printWords("two", words, word2Vec);

        // three should be absent due to stopWords
        assertFalse(words.contains("three"));

        assertTrue(words.contains("five"));
        assertTrue(words.contains("four"));

        sc.stop();


        // test serialization
        File tempFile = File.createTempFile("temp","tmp");
        tempFile.deleteOnExit();

        int idx1 = word2Vec.vocab().wordFor("day").getIndex();

        INDArray array1 = word2Vec.getWordVectorMatrix("day").dup();

        VocabWord word1 = word2Vec.vocab().elementAtIndex(0);

        WordVectorSerializer.writeWordVectors(word2Vec.getLookupTable(), tempFile);

        WordVectors vectors = WordVectorSerializer.loadTxtVectors(tempFile);

        VocabWord word2 = ((VocabCache<VocabWord>)vectors.vocab()).elementAtIndex(0);
        VocabWord wordIT = ((VocabCache<VocabWord>)vectors.vocab()).wordFor("it");
        int idx2 = vectors.vocab().wordFor("day").getIndex();

        INDArray array2 = vectors.getWordVectorMatrix("day").dup();

        System.out.println("word 'i': " + word2);
        System.out.println("word 'it': " + wordIT);

        assertEquals(idx1, idx2);
        assertEquals(word1, word2);
        assertEquals(array1, array2);
    }

    @Test
    @Ignore
    public void testPortugeseW2V() throws Exception {
        WordVectors word2Vec = WordVectorSerializer.loadTxtVectors(new File("/ext/Temp/para.txt"));
        word2Vec.setModelUtils(new FlatModelUtils());

        Collection<String> portu = word2Vec.wordsNearest("carro", 10);
        printWords("carro", portu, word2Vec);

        portu = word2Vec.wordsNearest("davi", 10);
        printWords("davi", portu, word2Vec);
    }

    private static void printWords(String target, Collection<String> list, WordVectors vec) {
        System.out.println("Words close to ["+target+"]:");
        for (String word: list) {
            double sim = vec.similarity(target, word);
            System.out.print("'"+ word+"': ["+ sim+"], ");
        }
        System.out.print("\n");
    }
}