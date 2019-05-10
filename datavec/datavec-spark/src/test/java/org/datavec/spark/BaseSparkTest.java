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

package org.datavec.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.After;
import org.junit.Before;

import java.io.Serializable;

public abstract class BaseSparkTest implements Serializable {
    protected static JavaSparkContext sc;

    @Before
    public void before() {
        sc = getContext();
    }

    @After
    public synchronized void after() {
        sc.close();
        //Wait until it's stopped, to avoid race conditions during tests
        for (int i = 0; i < 100; i++) {
            if (!sc.sc().stopped().get()) {
                try {
                    Thread.sleep(100L);
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            } else {
                break;
            }
        }
        if (!sc.sc().stopped().get()) {
            throw new RuntimeException("Spark context is not stopped after 10s");
        }


        sc = null;
    }

    public synchronized JavaSparkContext getContext() {
        if (sc != null)
            return sc;

        SparkConf sparkConf = new SparkConf().setMaster("local[*]").set("spark.driver.host", "localhost")
                        .set("spark.driverEnv.SPARK_LOCAL_IP", "127.0.0.1")
                        .set("spark.executorEnv.SPARK_LOCAL_IP", "127.0.0.1").setAppName("sparktest");
        if (useKryo()) {
            sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        }


        sc = new JavaSparkContext(sparkConf);

        return sc;
    }

    public boolean useKryo(){
        return false;
    }
}
