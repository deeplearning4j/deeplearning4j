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
import org.junit.Ignore;
import org.junit.Test;


import java.util.Collection;

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
       SparkConf sparkConf = new SparkConf().setMaster("local[*]").setAppName("sparktest");

        // Set SparkContext
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // Path of data
        String dataPath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();
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
                .setVectorLength(100)
                .setWindow(5)
                .setAlpha(0.055).setMinAlpha(0.001)
                .setIterations(3)
                .setNumWords(5);

        word2Vec.train(corpus);
        Collection<String> words = word2Vec.wordsNearest("day", 10);
        System.out.println(words);


        sc.stop();
    }


}