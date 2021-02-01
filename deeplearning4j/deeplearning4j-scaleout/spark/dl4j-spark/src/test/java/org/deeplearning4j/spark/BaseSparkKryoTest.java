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

package org.deeplearning4j.spark;

import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;

import java.lang.reflect.Field;
import java.util.Collections;
import java.util.Map;

public class BaseSparkKryoTest extends BaseSparkTest {

    @Override
    public long getTimeoutMilliseconds() {
        return 120000L;
    }

    @Override
    public JavaSparkContext getContext() {
        if (sc != null) {
            return sc;
        }

        //Ensure SPARK_USER environment variable is set for Spark Kryo tests
        String u = System.getenv("SPARK_USER");
        if(u == null || u.isEmpty()){
            try {
                Class[] classes = Collections.class.getDeclaredClasses();
                Map<String, String> env = System.getenv();
                for (Class cl : classes) {
                    if ("java.util.Collections$UnmodifiableMap".equals(cl.getName())) {
                        Field field = cl.getDeclaredField("m");
                        field.setAccessible(true);
                        Object obj = field.get(env);
                        Map<String, String> map = (Map<String, String>) obj;
                        String user = System.getProperty("user.name");
                        if(user == null || user.isEmpty())
                            user = "user";
                        map.put("SPARK_USER", user);
                    }
                }
            } catch (Exception e){
                throw new RuntimeException(e);
            }
        }



        SparkConf sparkConf = new SparkConf().setMaster("local[" + numExecutors() + "]")
                .setAppName("sparktest")
                .set("spark.driver.host", "localhost");

        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.kryo.Nd4jRegistrator");

        sc = new JavaSparkContext(sparkConf);

        return sc;
    }

}

