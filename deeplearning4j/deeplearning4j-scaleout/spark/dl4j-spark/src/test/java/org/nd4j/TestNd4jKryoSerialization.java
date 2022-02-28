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

package org.nd4j;

import com.sun.jna.Platform;
import lombok.AllArgsConstructor;
import lombok.SneakyThrows;
import lombok.extern.slf4j.Slf4j;
import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.api.java.function.VoidFunction;
import org.apache.spark.broadcast.Broadcast;
import org.apache.spark.serializer.SerializerInstance;
import org.junit.jupiter.api.*;
import org.nd4j.common.primitives.*;
import org.nd4j.common.resources.Downloader;
import org.nd4j.common.tests.BaseND4JTest;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import scala.Tuple2;

import java.io.File;
import java.net.URI;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertTrue;
@Slf4j
@Tag(TagNames.SPARK)
@Tag(TagNames.DIST_SYSTEMS)
public class TestNd4jKryoSerialization extends BaseND4JTest {

    private JavaSparkContext sc;

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

    @BeforeEach
    public void before() {
        SparkConf sparkConf = new SparkConf();
        sparkConf.setMaster("local[*]");
        sparkConf.set("spark.driver.host", "localhost");
        sparkConf.setAppName("Iris");

        sparkConf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer");
        sparkConf.set("spark.kryo.registrator", "org.nd4j.kryo.Nd4jRegistrator");

        sc = new JavaSparkContext(sparkConf);
    }

    @Test
    public void testSerialization() {

        Tuple2<INDArray, INDArray> t2 = new Tuple2<>(Nd4j.linspace(1, 10, 10, DataType.FLOAT), Nd4j.linspace(10, 20, 10, DataType.FLOAT));

        Broadcast<Tuple2<INDArray, INDArray>> b = sc.broadcast(t2);

        List<INDArray> list = new ArrayList<>();
        for (int i = 0; i < 100; i++) {
            list.add(Nd4j.ones(5));
        }

        JavaRDD<INDArray> rdd = sc.parallelize(list);

        rdd.foreach(new AssertFn(b));
    }

    @Test
    public void testSerializationPrimitives(){

        Counter<Integer> c = new Counter<>();
        c.incrementCount(5, 3.0);

        CounterMap<Integer,Double> cm = new CounterMap<>();
        cm.setCount(7, 3.0, 4.5);

        Object[] objs = new Object[]{
                new AtomicBoolean(true),
                new AtomicBoolean(false),
                new AtomicDouble(5.0),
                c,
                cm,
                new ImmutablePair<>(5,3.0),
                new ImmutableQuad<>(1,2.0,3.0f,4L),
                new ImmutableTriple<>(1,2.0,3.0f),
                new Pair<>(5, 3.0),
                new Quad<>(1,2.0,3.0f,4L),
                new Triple<>(1,2.0,3.0f)};


        SerializerInstance si = sc.env().serializer().newInstance();

        for (Object o : objs) {
            System.out.println(o.getClass());
            //System.out.println(ie.getClass());
            testSerialization(o, si);
        }
    }

    private <T> void testSerialization(T in, SerializerInstance si) {
        ByteBuffer bb = si.serialize(in, null);
        T deserialized = (T)si.deserialize(bb, null);

//        assertEquals(in, deserialized);
        boolean equals = in.equals(deserialized);
        assertTrue(equals,in.getClass() + "\t" + in.toString());
    }


    @AfterEach
    public void after() {
        if (sc != null)
            sc.close();
    }

    @AllArgsConstructor
    public static class AssertFn implements VoidFunction<INDArray> {

        private Broadcast<Tuple2<INDArray, INDArray>> b;

        @Override
        public void call(INDArray arr) throws Exception {
            Tuple2<INDArray, INDArray> t2 = b.getValue();
            assertEquals(Nd4j.linspace(1, 10, 10, DataType.FLOAT), t2._1());
            assertEquals(Nd4j.linspace(10, 20, 10, DataType.FLOAT), t2._2());

            assertEquals(Nd4j.ones(5), arr);
        }
    }
}
