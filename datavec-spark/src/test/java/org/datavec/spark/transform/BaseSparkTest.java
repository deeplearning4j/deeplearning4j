/*
 *  * Copyright 2016 Skymind, Inc.
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
 */

package org.datavec.spark.transform;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.AfterClass;
import org.junit.BeforeClass;

/**
 * Created by Alex on 1/06/2016.
 */
public class BaseSparkTest {

    public static JavaSparkContext sc;

    @BeforeClass
    public static void beforeClass(){
        SparkConf conf = new SparkConf();
        conf.setAppName("Test");
        conf.set("spark.driverEnv.SPARK_LOCAL_IP","127.0.0.1");
        conf.set("spark.executorEnv.SPARK_LOCAL_IP","127.0.0.1");
        conf.setMaster("local[*]");
        sc = new JavaSparkContext(conf);
    }

    @AfterClass
    public static void afterClass() {
        if(sc != null)
            sc.stop();
    }

}
