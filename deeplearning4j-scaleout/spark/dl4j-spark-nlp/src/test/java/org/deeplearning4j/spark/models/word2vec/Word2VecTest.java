/*
 * Copyright 2015 Skymind,Inc.
 *
 *    Licensed under the Apache License, Version 2.0 (the "License");
 *    you may not use this file except in compliance with the License.
 *    You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0
 *
 *    Unless required by applicable law or agreed to in writing, software
 *    distributed under the License is distributed on an "AS IS" BASIS,
 *    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *    See the License for the specific language governing permissions and
 *    limitations under the License.
 */

package org.deeplearning4j.spark.models.word2vec;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.text.BaseSparkTest;
import org.junit.Test;
import org.springframework.core.io.ClassPathResource;

/**
 * Created by agibsonccc on 1/31/15.
 */
public class Word2VecTest extends BaseSparkTest {

    @Test
    public void testSparkWord2Vec() throws Exception {
        JavaRDD<String> corpus = sc.textFile(new ClassPathResource("basic/word2vec.txt").getFile().getAbsolutePath());
        Word2Vec word2Vec = new Word2Vec();
        sc.getConf().set(Word2VecPerformer.NEGATIVE,String.valueOf(0));
        word2Vec.train(corpus);

    }


    /**
     *
     * @return
     */
    @Override
    public JavaSparkContext getContext() {
        if(sc != null)
            return sc;
        // set to test mode
        SparkConf sparkConf = new SparkConf()
                .setMaster("local[8]").set(Word2VecPerformer.NEGATIVE, String.valueOf(0))
                .setAppName("sparktest");


        sc = new JavaSparkContext(sparkConf);
        return sc;

    }


}
