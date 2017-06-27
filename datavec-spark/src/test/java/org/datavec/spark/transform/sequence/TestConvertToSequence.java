/*-
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

package org.datavec.spark.transform.sequence;

import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.sequence.comparator.NumericalColumnComparator;
import org.datavec.api.writable.LongWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.spark.BaseSparkTest;
import org.datavec.spark.transform.SparkTransformExecutor;
import org.junit.Test;

import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 19/05/2017.
 */
public class TestConvertToSequence extends BaseSparkTest {

    @Test
    public void testConvertToSequenceCompoundKey() {

        Schema s = new Schema.Builder().addColumnsString("key1", "key2").addColumnLong("time").build();

        List<List<Writable>> allExamples =
                        Arrays.asList(Arrays.<Writable>asList(new Text("k1a"), new Text("k2a"), new LongWritable(10)),
                                        Arrays.<Writable>asList(new Text("k1b"), new Text("k2b"), new LongWritable(10)),
                                        Arrays.<Writable>asList(new Text("k1a"), new Text("k2a"),
                                                        new LongWritable(-10)),
                                        Arrays.<Writable>asList(new Text("k1b"), new Text("k2b"), new LongWritable(5)),
                                        Arrays.<Writable>asList(new Text("k1a"), new Text("k2a"), new LongWritable(0)));

        TransformProcess tp = new TransformProcess.Builder(s)
                        .convertToSequence(Arrays.asList("key1", "key2"), new NumericalColumnComparator("time"))
                        .build();

        JavaRDD<List<Writable>> rdd = sc.parallelize(allExamples);

        List<List<List<Writable>>> out = SparkTransformExecutor.executeToSequence(rdd, tp).collect();

        assertEquals(2, out.size());
        List<List<Writable>> seq0;
        List<List<Writable>> seq1;
        if (out.get(0).size() == 3) {
            seq0 = out.get(0);
            seq1 = out.get(1);
        } else {
            seq0 = out.get(1);
            seq1 = out.get(0);
        }

        List<List<Writable>> expSeq0 = Arrays.asList(
                        Arrays.<Writable>asList(new Text("k1a"), new Text("k2a"), new LongWritable(-10)),
                        Arrays.<Writable>asList(new Text("k1a"), new Text("k2a"), new LongWritable(0)),
                        Arrays.<Writable>asList(new Text("k1a"), new Text("k2a"), new LongWritable(10)));

        List<List<Writable>> expSeq1 = Arrays.asList(
                        Arrays.<Writable>asList(new Text("k1b"), new Text("k2b"), new LongWritable(5)),
                        Arrays.<Writable>asList(new Text("k1b"), new Text("k2b"), new LongWritable(10)));

        assertEquals(expSeq0, seq0);
        assertEquals(expSeq1, seq1);
    }

    @Test
    public void testConvertToSequenceLength1(){

        Schema s = new Schema.Builder()
                .addColumnsString("string")
                .addColumnLong("long")
                .build();

        List<List<Writable>> allExamples = Arrays.asList(
                Arrays.<Writable>asList(new Text("a"), new LongWritable(0)),
                Arrays.<Writable>asList(new Text("b"), new LongWritable(1)),
                Arrays.<Writable>asList(new Text("c"), new LongWritable(2)));

        TransformProcess tp = new TransformProcess.Builder(s)
                .convertToSequence()
                .build();

        JavaRDD<List<Writable>> rdd = sc.parallelize(allExamples);

        JavaRDD<List<List<Writable>>> out = SparkTransformExecutor.executeToSequence(rdd, tp);

        List<List<List<Writable>>> out2 = out.collect();

        assertEquals(3, out2.size());

        for( int i=0; i<3; i++ ){
            assertTrue(out2.contains(Collections.singletonList(allExamples.get(i))));
        }
    }
}
