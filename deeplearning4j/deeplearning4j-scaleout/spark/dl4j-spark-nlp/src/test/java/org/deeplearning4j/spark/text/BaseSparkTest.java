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

package org.deeplearning4j.spark.text;

import com.sun.jna.Platform;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecVariables;
import org.junit.jupiter.api.AfterEach;
import org.junit.jupiter.api.BeforeAll;
import org.junit.jupiter.api.BeforeEach;
import org.nd4j.common.resources.Downloader;

import java.io.File;
import java.io.Serializable;
import java.lang.reflect.Field;
import java.net.URI;
import java.util.Collections;
import java.util.Map;
@Slf4j
public abstract class BaseSparkTest extends BaseDL4JTest implements Serializable {
    protected transient JavaSparkContext sc;

    @BeforeAll
    @SneakyThrows
    public static void beforeAll() {
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
    }



    @Override
    public long getTimeoutMilliseconds() {
        return 120000L;
    }

    @BeforeEach
    public void before() throws Exception {
        sc = getContext();
    }

    @AfterEach
    public void after() {
        if(sc != null) {
            sc.close();
        }
        sc = null;
    }

    /**
     *
     * @return
     */
    public JavaSparkContext getContext() {
        if (sc != null)
            return sc;

        //Ensure SPARK_USER environment variable is set for Spark tests
        String u = System.getenv("SPARK_USER");
        Map<String, String> env = System.getenv();
        if(u == null || u.isEmpty()) {
            try {
                Class[] classes = Collections.class.getDeclaredClasses();
                for (Class cl : classes) {
                    if ("java.util.Collections$UnmodifiableMap".equals(cl.getName())) {
                        Field field = cl.getDeclaredField("m");
                        field.setAccessible(true);
                        Object obj = field.get(env);
                        Map<String, String> map = (Map<String, String>) obj;
                        String user = System.getProperty("user.name");
                        if (user == null || user.isEmpty())
                            user = "user";
                        map.put("SPARK_USER", user);
                    }
                }
            } catch (Exception e) {
                throw new RuntimeException(e);
            }
        }

        // set to test mode
        SparkConf sparkConf = new SparkConf().setMaster("local[4]").set("spark.driver.host", "localhost")
            .setAppName("sparktest")
            .set(Word2VecVariables.NUM_WORDS, String.valueOf(1));


        sc = new JavaSparkContext(sparkConf);
        return sc;

    }

}
