/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.deeplearning4j.BaseDL4JTest;
import org.deeplearning4j.spark.models.embeddings.word2vec.Word2VecVariables;
import org.junit.After;
import org.junit.Before;

import java.io.Serializable;
import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Map;

/**
 * Created by agibsonccc on 1/23/15.
 */
public abstract class BaseSparkTest extends BaseDL4JTest implements Serializable {
    protected transient JavaSparkContext sc;

    @Override
    public long getTimeoutMilliseconds() {
        return 120000L;
    }

    @Before
    public void before() throws Exception {
        sc = getContext();
    }

    @After
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
