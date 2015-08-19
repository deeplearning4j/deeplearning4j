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
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

import java.util.Arrays;
import java.util.Collection;

import static org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecVariables.*;

/**
 * @author jeffreytang
 */
public class Word2VecTest {

    @Test
    public void testConcepts() throws Exception {
        // These are all default values for word2vec
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[4]")
                .setAppName("sparktest")
                .set(VECTOR_LENGTH, String.valueOf(100))
                .set(ADAGRAD, String.valueOf(false))
                .set(NEGATIVE, String.valueOf(0))
                .set(NUM_WORDS, String.valueOf(1))
                .set(WINDOW, String.valueOf(5))
                .set(ALPHA, String.valueOf(0.025))
                .set(MIN_ALPHA, String.valueOf(1e-2))
                .set(ITERATIONS, String.valueOf(1))
                .set(N_GRAMS, String.valueOf(1))
                .set(TOKENIZER, "org.deeplearning4j.text.tokenization.tokenizerfactory.DefaultTokenizerFactory")
                .set(TOKEN_PREPROCESSOR, "org.deeplearning4j.text.tokenization.tokenizer.preprocessor.CommonPreprocessor")
                .set(REMOVE_STOPWORDS, String.valueOf(false));

        // Set SparkContext
        JavaSparkContext sc = new JavaSparkContext(sparkConf);

        // Path of data
        String dataPath = new ClassPathResource("raw_sentences.txt").getFile().getAbsolutePath();

        // Read in data
        JavaRDD<String> corpus = sc.textFile(dataPath);

        Word2Vec word2Vec = new Word2Vec();
        word2Vec.train(corpus);
        Collection<String> words = word2Vec.wordsNearest("day", 10);
        System.out.println(Arrays.toString(words.toArray()));

//        assertTrue(words.contains("week"));
    }


}