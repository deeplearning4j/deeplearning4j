/*-
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
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecVariables;
import org.junit.After;
import org.junit.Before;

import java.io.Serializable;

/**
 * Created by agibsonccc on 1/23/15.
 */
public abstract class BaseSparkTest implements Serializable {
    protected transient JavaSparkContext sc;

    @Before
    public void before() throws Exception {
        sc = getContext();
    }

    @After
    public void after() {
        sc.close();
        sc = null;
    }

    /**
     *
     * @return
     */
    public JavaSparkContext getContext() {
        if (sc != null)
            return sc;
        // set to test mode
        SparkConf sparkConf = new SparkConf().setMaster("local[4]").set("spark.driver.host", "localhost")
            .setAppName("sparktest")
            .set(Word2VecVariables.NUM_WORDS, String.valueOf(1));


        sc = new JavaSparkContext(sparkConf);
        return sc;

    }

}
