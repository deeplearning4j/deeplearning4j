/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
 *
 * This program and the accompanying materials are made available under the
 * terms of the Apache License, Version 2.0 which is available at
 * https://www.apache.org/licenses/LICENSE-2.0.
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 * License for the specific language governing permissions and limitations
 * under the License.
 *
 * SPDX-License-Identifier: Apache-2.0
 ******************************************************************************/

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
