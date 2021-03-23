/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */
package org.datavec.spark;

import com.sun.jna.Platform;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeEach;

import java.io.File;
import java.io.Serializable;
import java.net.URI;

import org.junit.jupiter.api.DisplayName;
import org.junit.jupiter.api.extension.ExtendWith;
import org.nd4j.common.resources.Downloader;

@Slf4j
@DisplayName("Base Spark Test")
public abstract class BaseSparkTest implements Serializable {

    protected static JavaSparkContext sc;

    @SneakyThrows
    @BeforeEach
    void before() {
        if(Platform.isWindows()) {
            File hadoopHome = new File(System.getProperty("java.io.tmpdir"),"hadoop-tmp");
            File binDir = new File(hadoopHome,"bin");
            if(!binDir.exists())
                binDir.mkdirs();
            File outputFile = new File(binDir,"winutils.exe");
            if(!outputFile.exists()) {
                log.info("Fixing spark for windows");
                Downloader.download("winutils.exe",
                        URI.create("https://github.com/cdarlint/winutils/blob/master/hadoop-2.6.5/bin/winutils.exe?raw=true").toURL(),
                        outputFile,"db24b404d2331a1bec7443336a5171f1",3);
            }

            System.setProperty("hadoop.home.dir", hadoopHome.getAbsolutePath());
        }
        sc = getContext();
    }

    @AfterEach
    synchronized void after() {
        sc.close();
        // Wait until it's stopped, to avoid race conditions during tests
        for (int i = 0; i < 100; i++) {
            if (!sc.sc().stopped().get()) {
                try {
                    Thread.sleep(100L);
                } catch (InterruptedException e) {
                    log.error("", e);
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
        SparkConf sparkConf = new SparkConf().setMaster("local[*]").set("spark.driver.host", "localhost").set("spark.driverEnv.SPARK_LOCAL_IP", "127.0.0.1").set("spark.executorEnv.SPARK_LOCAL_IP", "127.0.0.1").setAppName("sparktest");
        if (useKryo()) {
            sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        }
        sc = new JavaSparkContext(sparkConf);
        return sc;
    }

    public boolean useKryo() {
        return false;
    }
}
