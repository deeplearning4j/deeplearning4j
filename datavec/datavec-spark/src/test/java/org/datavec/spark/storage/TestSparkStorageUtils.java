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

package org.datavec.spark.storage;

import com.google.common.io.Files;
import org.apache.spark.api.java.JavaPairRDD;
import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.writable.*;
import org.datavec.spark.BaseSparkTest;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.io.File;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 30/05/2017.
 */
public class TestSparkStorageUtils extends BaseSparkTest {

    @Test
    public void testSaveRestoreMapFile() {
        List<List<Writable>> l = new ArrayList<>();
        l.add(Arrays.<org.datavec.api.writable.Writable>asList(new Text("zero"), new IntWritable(0),
                        new DoubleWritable(0), new NDArrayWritable(Nd4j.valueArrayOf(10, 0.0))));
        l.add(Arrays.<org.datavec.api.writable.Writable>asList(new Text("one"), new IntWritable(11),
                        new DoubleWritable(11.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 11.0))));
        l.add(Arrays.<org.datavec.api.writable.Writable>asList(new Text("two"), new IntWritable(22),
                        new DoubleWritable(22.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 22.0))));

        JavaRDD<List<Writable>> rdd = sc.parallelize(l);

        File f = Files.createTempDir();
        f.delete();
        f.deleteOnExit();
        String path = "file:///" + f.getAbsolutePath();

        SparkStorageUtils.saveMapFile(path, rdd);
        JavaPairRDD<Long, List<Writable>> restored = SparkStorageUtils.restoreMapFile(path, sc);

        Map<Long, List<Writable>> m = restored.collectAsMap();

        assertEquals(3, m.size());
        for (int i = 0; i < 3; i++) {
            assertEquals(l.get(i), m.get((long) i));
        }


        //Also test sequence file:
        f = Files.createTempDir();
        f.delete();
        f.deleteOnExit();
        path = "file:///" + f.getAbsolutePath();

        SparkStorageUtils.saveSequenceFile(path, rdd);
        List<List<Writable>> restored2 = SparkStorageUtils.restoreSequenceFile(path, sc).collect();

        //Sequence file loading + collect iteration order is not guaranteed (depends on number of partitions, etc)
        assertEquals(3, restored2.size());
        assertTrue(l.containsAll(restored2) && restored2.containsAll(l));
    }

    @Test
    public void testSaveRestoreMapFileSequences() {
        List<List<List<Writable>>> l = new ArrayList<>();
        l.add(Arrays.asList(
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("zero"), new IntWritable(0),
                                        new DoubleWritable(0), new NDArrayWritable(Nd4j.valueArrayOf(10, 0.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("one"), new IntWritable(1),
                                        new DoubleWritable(1.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 1.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("two"), new IntWritable(2),
                                        new DoubleWritable(2.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 2.0)))));

        l.add(Arrays.asList(
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Bzero"), new IntWritable(10),
                                        new DoubleWritable(10), new NDArrayWritable(Nd4j.valueArrayOf(10, 10.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Bone"), new IntWritable(11),
                                        new DoubleWritable(11.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 11.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Btwo"), new IntWritable(12),
                                        new DoubleWritable(12.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 12.0)))));

        l.add(Arrays.asList(
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Czero"), new IntWritable(20),
                                        new DoubleWritable(20), new NDArrayWritable(Nd4j.valueArrayOf(10, 20.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Cone"), new IntWritable(21),
                                        new DoubleWritable(21.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 21.0))),
                        Arrays.<org.datavec.api.writable.Writable>asList(new Text("Ctwo"), new IntWritable(22),
                                        new DoubleWritable(22.0), new NDArrayWritable(Nd4j.valueArrayOf(10, 22.0)))));

        JavaRDD<List<List<Writable>>> rdd = sc.parallelize(l);

        File f = Files.createTempDir();
        f.delete();
        f.deleteOnExit();
        String path = "file:///" + f.getAbsolutePath();

        SparkStorageUtils.saveMapFileSequences(path, rdd);
        JavaPairRDD<Long, List<List<Writable>>> restored = SparkStorageUtils.restoreMapFileSequences(path, sc);

        Map<Long, List<List<Writable>>> m = restored.collectAsMap();

        assertEquals(3, m.size());
        for (int i = 0; i < 3; i++) {
            assertEquals(l.get(i), m.get((long) i));
        }

        //Also test sequence file:
        f = Files.createTempDir();
        f.delete();
        f.deleteOnExit();
        path = "file:///" + f.getAbsolutePath();

        SparkStorageUtils.saveSequenceFileSequences(path, rdd);
        List<List<List<Writable>>> restored2 = SparkStorageUtils.restoreSequenceFileSequences(path, sc).collect();

        //Sequence file loading + collect iteration order is not guaranteed (depends on number of partitions, etc)
        assertEquals(3, restored2.size());
        assertTrue(l.containsAll(restored2) && restored2.containsAll(l));
    }



}
