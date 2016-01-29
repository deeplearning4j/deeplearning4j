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
import org.deeplearning4j.models.word2vec.VocabWord;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.util.Collection;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertNotEquals;

/**
 * @author jeffreytang
 */
@Ignore
public class Word2VecTest {

    @Test
    public void testConcepts() throws Exception {
        // These are all default values for word2vec
        /*SparkConf sparkConf = new SparkConf()
                .setMaster("spark://192.168.1.35:7077")
                .set("spark.executor.memory", "20G")
                .set("spark.driver.memory", "20G")
                .setAppName("sparktest");
        */
       SparkConf sparkConf = new SparkConf().setMaster("local").setAppName("sparktest");

        // Set SparkContext
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // Path of data
        String dataPath = new ClassPathResource("/big/raw_sentences.txt").getFile().getAbsolutePath();
//        String dataPath = new ClassPathResource("spark_word2vec_test.txt").getFile().getAbsolutePath();

        // Read in data
        JavaRDD<String> corpus = sc.textFile(dataPath);

        Word2Vec word2Vec = new Word2Vec()
                .setnGrams(1)
                .setTokenizer("org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory")
                .setTokenPreprocessor("org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor")
                .setRemoveStop(false)
                .setSeed(42L)
                .setNegative(0)
                .setUseAdaGrad(false)
                .setVectorLength(150)
                .setWindow(5)
                .setAlpha(0.025).setMinAlpha(0.0001)
                .setIterations(1)
                .setNumWords(5);

        word2Vec.train(corpus);

        System.out.println("two idx: " + word2Vec.getVocab().wordFor("two").getIndex());
        System.out.println("four idx: " + word2Vec.getVocab().wordFor("four").getIndex());


        InMemoryLookupTable<VocabWord> table = (InMemoryLookupTable<VocabWord>) word2Vec.lookupTable();

        System.out.println("Two by idx: " + table.getSyn0().getRow(126));
        System.out.println("Two by lookup: " + table.vector("two"));
        System.out.println("Two external: " + word2Vec.getWordVectorMatrix("two"));

        double sim = word2Vec.similarity("day", "night");
        System.out.println("day/night similarity: " + sim);

        Collection<String> words = word2Vec.wordsNearest("day", 10);
        printWords("day", words, word2Vec);


        sim = word2Vec.similarity("two", "four");
        System.out.println("two/four similarity: " + sim);

        words = word2Vec.wordsNearest("two", 10);
        printWords("two", words, word2Vec);

        sc.stop();
    }

    @Test
    public void testRandom() throws Exception {
        INDArray arr1 = Nd4j.rand(123L, 1, 20);
        INDArray arr2 = Nd4j.rand(123L, 1, 20);
        INDArray arr3 = Nd4j.rand(124L, 1, 20);

        assertEquals(arr1, arr2);
        assertNotEquals(arr3, arr2);
    }

    @Test
    public void testAddi1() throws Exception {
        INDArray arr1 = Nd4j.create(new double[] {0.0, 0.0, 0.0, 0.0});
        INDArray arr2 = Nd4j.create(new double[] {0.5, 0.5, 0.5, 0.5});

        arr1.addi(arr2);

        assertEquals(arr1, arr2);

        assertEquals(0.5, arr1.getDouble(0), 0.001);
        assertEquals(0.5, arr1.getDouble(1), 0.001);
        assertEquals(0.5, arr1.getDouble(2), 0.001);
        assertEquals(0.5, arr1.getDouble(3), 0.001);
    }

    @Test
    public void testAddi2() throws Exception {
        INDArray arr1 = Nd4j.zeros(1,4);
        INDArray arr2 = Nd4j.create(new double[] {0.5, 0.5, 0.5, 0.5});

        arr1.addi(arr2);

        assertEquals(arr1, arr2);

        assertEquals(0.5, arr1.getDouble(0), 0.001);
        assertEquals(0.5, arr1.getDouble(1), 0.001);
        assertEquals(0.5, arr1.getDouble(2), 0.001);
        assertEquals(0.5, arr1.getDouble(3), 0.001);
    }

    private static void printWords(String target, Collection<String> list, Word2Vec vec) {
        System.out.println("Words close to ["+target+"]:");
        for (String word: list) {
            double sim = vec.similarity(target, word);
            System.out.print("'"+ word+"': ["+ sim+"], ");
        }
        System.out.print("\n");
    }
}