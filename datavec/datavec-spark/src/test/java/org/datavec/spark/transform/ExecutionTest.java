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

package org.datavec.spark.transform;

import org.apache.spark.api.java.JavaRDD;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.transform.transform.categorical.FirstDigitTransform;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.IntWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.api.writable.NDArrayWritable;
import org.datavec.spark.BaseSparkTest;
import org.datavec.python.PythonTransform;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.*;

import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertTrue;

/**
 * Created by Alex on 25/11/2016.
 */
public class ExecutionTest extends BaseSparkTest {

    @Test
    public void testExecutionSimple() {
        Schema schema = new Schema.Builder().addColumnInteger("col0")
                        .addColumnCategorical("col1", "state0", "state1", "state2").addColumnDouble("col2").build();

        TransformProcess tp = new TransformProcess.Builder(schema).categoricalToInteger("col1")
                        .doubleMathOp("col2", MathOp.Add, 10.0).build();

        List<List<Writable>> inputData = new ArrayList<>();
        inputData.add(Arrays.<Writable>asList(new IntWritable(0), new Text("state2"), new DoubleWritable(0.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(1), new Text("state1"), new DoubleWritable(1.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(2), new Text("state0"), new DoubleWritable(2.1)));

        JavaRDD<List<Writable>> rdd = sc.parallelize(inputData);

        List<List<Writable>> out = new ArrayList<>(SparkTransformExecutor.execute(rdd, tp).collect());

        Collections.sort(out, new Comparator<List<Writable>>() {
            @Override
            public int compare(List<Writable> o1, List<Writable> o2) {
                return Integer.compare(o1.get(0).toInt(), o2.get(0).toInt());
            }
        });

        List<List<Writable>> expected = new ArrayList<>();
        expected.add(Arrays.<Writable>asList(new IntWritable(0), new IntWritable(2), new DoubleWritable(10.1)));
        expected.add(Arrays.<Writable>asList(new IntWritable(1), new IntWritable(1), new DoubleWritable(11.1)));
        expected.add(Arrays.<Writable>asList(new IntWritable(2), new IntWritable(0), new DoubleWritable(12.1)));

        assertEquals(expected, out);
    }

    @Test
    public void testExecutionSequence() {

        Schema schema = new SequenceSchema.Builder().addColumnInteger("col0")
                        .addColumnCategorical("col1", "state0", "state1", "state2").addColumnDouble("col2").build();

        TransformProcess tp = new TransformProcess.Builder(schema).categoricalToInteger("col1")
                        .doubleMathOp("col2", MathOp.Add, 10.0).build();

        List<List<List<Writable>>> inputSequences = new ArrayList<>();
        List<List<Writable>> seq1 = new ArrayList<>();
        seq1.add(Arrays.<Writable>asList(new IntWritable(0), new Text("state2"), new DoubleWritable(0.1)));
        seq1.add(Arrays.<Writable>asList(new IntWritable(1), new Text("state1"), new DoubleWritable(1.1)));
        seq1.add(Arrays.<Writable>asList(new IntWritable(2), new Text("state0"), new DoubleWritable(2.1)));
        List<List<Writable>> seq2 = new ArrayList<>();
        seq2.add(Arrays.<Writable>asList(new IntWritable(3), new Text("state0"), new DoubleWritable(3.1)));
        seq2.add(Arrays.<Writable>asList(new IntWritable(4), new Text("state1"), new DoubleWritable(4.1)));

        inputSequences.add(seq1);
        inputSequences.add(seq2);

        JavaRDD<List<List<Writable>>> rdd = sc.parallelize(inputSequences);

        List<List<List<Writable>>> out =
                        new ArrayList<>(SparkTransformExecutor.executeSequenceToSequence(rdd, tp).collect());

        Collections.sort(out, new Comparator<List<List<Writable>>>() {
            @Override
            public int compare(List<List<Writable>> o1, List<List<Writable>> o2) {
                return -Integer.compare(o1.size(), o2.size());
            }
        });

        List<List<List<Writable>>> expectedSequence = new ArrayList<>();
        List<List<Writable>> seq1e = new ArrayList<>();
        seq1e.add(Arrays.<Writable>asList(new IntWritable(0), new IntWritable(2), new DoubleWritable(10.1)));
        seq1e.add(Arrays.<Writable>asList(new IntWritable(1), new IntWritable(1), new DoubleWritable(11.1)));
        seq1e.add(Arrays.<Writable>asList(new IntWritable(2), new IntWritable(0), new DoubleWritable(12.1)));
        List<List<Writable>> seq2e = new ArrayList<>();
        seq2e.add(Arrays.<Writable>asList(new IntWritable(3), new IntWritable(0), new DoubleWritable(13.1)));
        seq2e.add(Arrays.<Writable>asList(new IntWritable(4), new IntWritable(1), new DoubleWritable(14.1)));

        expectedSequence.add(seq1e);
        expectedSequence.add(seq2e);

        assertEquals(expectedSequence, out);
    }


    @Test
    public void testReductionGlobal() {

        List<List<Writable>> in = Arrays.asList(
                Arrays.<Writable>asList(new Text("first"), new DoubleWritable(3.0)),
                Arrays.<Writable>asList(new Text("second"), new DoubleWritable(5.0))
        );

        JavaRDD<List<Writable>> inData = sc.parallelize(in);

        Schema s = new Schema.Builder()
                .addColumnString("textCol")
                .addColumnDouble("doubleCol")
                .build();

        TransformProcess tp = new TransformProcess.Builder(s)
                .reduce(new Reducer.Builder(ReduceOp.TakeFirst)
                        .takeFirstColumns("textCol")
                        .meanColumns("doubleCol").build())
                .build();

        JavaRDD<List<Writable>> outRdd = SparkTransformExecutor.execute(inData, tp);

        List<List<Writable>> out = outRdd.collect();

        List<List<Writable>> expOut = Collections.singletonList(Arrays.<Writable>asList(new Text("first"), new DoubleWritable(4.0)));

        assertEquals(expOut, out);
    }

    @Test
    public void testReductionByKey(){

        List<List<Writable>> in = Arrays.asList(
                Arrays.<Writable>asList(new IntWritable(0), new Text("first"), new DoubleWritable(3.0)),
                Arrays.<Writable>asList(new IntWritable(0), new Text("second"), new DoubleWritable(5.0)),
                Arrays.<Writable>asList(new IntWritable(1), new Text("f"), new DoubleWritable(30.0)),
                Arrays.<Writable>asList(new IntWritable(1), new Text("s"), new DoubleWritable(50.0))
        );

        JavaRDD<List<Writable>> inData = sc.parallelize(in);

        Schema s = new Schema.Builder()
                .addColumnInteger("intCol")
                .addColumnString("textCol")
                .addColumnDouble("doubleCol")
                .build();

        TransformProcess tp = new TransformProcess.Builder(s)
                .reduce(new Reducer.Builder(ReduceOp.TakeFirst)
                        .keyColumns("intCol")
                        .takeFirstColumns("textCol")
                        .meanColumns("doubleCol").build())
                .build();

        JavaRDD<List<Writable>> outRdd = SparkTransformExecutor.execute(inData, tp);

        List<List<Writable>> out = outRdd.collect();

        List<List<Writable>> expOut = Arrays.asList(
                Arrays.<Writable>asList(new IntWritable(0), new Text("first"), new DoubleWritable(4.0)),
                Arrays.<Writable>asList(new IntWritable(1), new Text("f"), new DoubleWritable(40.0)));

        out = new ArrayList<>(out);
        Collections.sort(
                out, new Comparator<List<Writable>>() {
                    @Override
                    public int compare(List<Writable> o1, List<Writable> o2) {
                        return Integer.compare(o1.get(0).toInt(), o2.get(0).toInt());
                    }
                }
        );

        assertEquals(expOut, out);
    }


    @Test
    public void testUniqueMultiCol(){

        Schema schema = new Schema.Builder()
                .addColumnInteger("col0")
                .addColumnCategorical("col1", "state0", "state1", "state2")
                .addColumnDouble("col2").build();

        List<List<Writable>> inputData = new ArrayList<>();
        inputData.add(Arrays.<Writable>asList(new IntWritable(0), new Text("state2"), new DoubleWritable(0.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(1), new Text("state1"), new DoubleWritable(1.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(2), new Text("state0"), new DoubleWritable(2.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(0), new Text("state2"), new DoubleWritable(0.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(1), new Text("state1"), new DoubleWritable(1.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(2), new Text("state0"), new DoubleWritable(2.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(0), new Text("state2"), new DoubleWritable(0.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(1), new Text("state1"), new DoubleWritable(1.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(2), new Text("state0"), new DoubleWritable(2.1)));

        JavaRDD<List<Writable>> rdd = sc.parallelize(inputData);

        Map<String,List<Writable>> l = AnalyzeSpark.getUnique(Arrays.asList("col0", "col1"), schema, rdd);

        assertEquals(2, l.size());
        List<Writable> c0 = l.get("col0");
        assertEquals(3, c0.size());
        assertTrue(c0.contains(new IntWritable(0)) && c0.contains(new IntWritable(1)) && c0.contains(new IntWritable(2)));

        List<Writable> c1 = l.get("col1");
        assertEquals(3, c1.size());
        assertTrue(c1.contains(new Text("state0")) && c1.contains(new Text("state1")) && c1.contains(new Text("state2")));
    }

    @Test(timeout = 60000L)
    @Ignore("AB 2019/05/21 - Fine locally, timeouts on CI - Issue #7657 and #7771")
    public void testPythonExecution() throws Exception {
        Schema schema = new Schema.Builder().addColumnInteger("col0")
                .addColumnString("col1").addColumnDouble("col2").build();

        Schema finalSchema = new Schema.Builder().addColumnInteger("col0")
                .addColumnInteger("col1").addColumnDouble("col2").build();
        String pythonCode = "col1 = ['state0', 'state1', 'state2'].index(col1)\ncol2 += 10.0";
        TransformProcess tp = new TransformProcess.Builder(schema).transform(
          new PythonTransform(
                pythonCode,
                  finalSchema
          )
        ).build();
        List<List<Writable>> inputData = new ArrayList<>();
        inputData.add(Arrays.<Writable>asList(new IntWritable(0), new Text("state2"), new DoubleWritable(0.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(1), new Text("state1"), new DoubleWritable(1.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(2), new Text("state0"), new DoubleWritable(2.1)));

        JavaRDD<List<Writable>> rdd = sc.parallelize(inputData);

        List<List<Writable>> out = new ArrayList<>(SparkTransformExecutor.execute(rdd, tp).collect());

        Collections.sort(out, new Comparator<List<Writable>>() {
            @Override
            public int compare(List<Writable> o1, List<Writable> o2) {
                return Integer.compare(o1.get(0).toInt(), o2.get(0).toInt());
            }
        });

        List<List<Writable>> expected = new ArrayList<>();
        expected.add(Arrays.<Writable>asList(new IntWritable(0), new IntWritable(2), new DoubleWritable(10.1)));
        expected.add(Arrays.<Writable>asList(new IntWritable(1), new IntWritable(1), new DoubleWritable(11.1)));
        expected.add(Arrays.<Writable>asList(new IntWritable(2), new IntWritable(0), new DoubleWritable(12.1)));

        assertEquals(expected, out);
    }

    @Test(timeout = 60000L)
    @Ignore("AB 2019/05/21 - Fine locally, timeouts on CI - Issue #7657 and #7771")
    public void testPythonExecutionWithNDArrays() throws Exception {
        long[] shape = new long[]{3, 2};
        Schema schema = new Schema.Builder().addColumnInteger("id").addColumnNDArray("col1", shape)
                .addColumnNDArray("col2", shape).build();

        Schema finalSchema = new Schema.Builder().addColumnInteger("id").addColumnNDArray("col1", shape)
                .addColumnNDArray("col2", shape).addColumnNDArray("col3", shape).build();

        String pythonCode = "col3 = col1 + col2";
        TransformProcess tp = new TransformProcess.Builder(schema).transform(
                new PythonTransform(
                        pythonCode,
                        finalSchema
                )
        ).build();

        INDArray zeros = Nd4j.zeros(shape);
        INDArray ones = Nd4j.ones(shape);
        INDArray twos = ones.add(ones);

        List<List<Writable>> inputData = new ArrayList<>();
        inputData.add(Arrays.<Writable>asList(new IntWritable(0), new NDArrayWritable(zeros), new NDArrayWritable(zeros)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(1), new NDArrayWritable(zeros), new NDArrayWritable(ones)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(2), new NDArrayWritable(ones), new NDArrayWritable(ones)));

        JavaRDD<List<Writable>> rdd = sc.parallelize(inputData);

        List<List<Writable>> out = new ArrayList<>(SparkTransformExecutor.execute(rdd, tp).collect());

        Collections.sort(out, new Comparator<List<Writable>>() {
            @Override
            public int compare(List<Writable> o1, List<Writable> o2) {
                return Integer.compare(o1.get(0).toInt(), o2.get(0).toInt());
            }
        });

        List<List<Writable>> expected = new ArrayList<>();
        expected.add(Arrays.<Writable>asList(new IntWritable(0), new NDArrayWritable(zeros), new NDArrayWritable(zeros), new NDArrayWritable(zeros)));
        expected.add(Arrays.<Writable>asList(new IntWritable(1), new NDArrayWritable(zeros), new NDArrayWritable(ones), new NDArrayWritable(ones)));
        expected.add(Arrays.<Writable>asList(new IntWritable(2), new NDArrayWritable(ones), new NDArrayWritable(ones), new NDArrayWritable(twos)));
    }

    @Test
    public void testFirstDigitTransformBenfordsLaw(){
        Schema s = new Schema.Builder()
                .addColumnString("data")
                .addColumnDouble("double")
                .addColumnString("stringNumber")
                .build();

        List<List<Writable>> in = Arrays.asList(
                Arrays.<Writable>asList(new Text("a"), new DoubleWritable(3.14159), new Text("8e-4")),
                Arrays.<Writable>asList(new Text("a2"), new DoubleWritable(3.14159), new Text("7e-4")),
                Arrays.<Writable>asList(new Text("b"), new DoubleWritable(2.71828), new Text("7e2")),
                Arrays.<Writable>asList(new Text("c"), new DoubleWritable(1.61803), new Text("6e8")),
                Arrays.<Writable>asList(new Text("c"), new DoubleWritable(1.61803), new Text("2.0")),
                Arrays.<Writable>asList(new Text("c"), new DoubleWritable(1.61803), new Text("2.1")),
                Arrays.<Writable>asList(new Text("c"), new DoubleWritable(1.61803), new Text("2.2")),
                Arrays.<Writable>asList(new Text("c"), new DoubleWritable(-2), new Text("non numerical")));

        //Test Benfords law use case:
        TransformProcess tp = new TransformProcess.Builder(s)
                .firstDigitTransform("double", "fdDouble", FirstDigitTransform.Mode.EXCEPTION_ON_INVALID)
                .firstDigitTransform("stringNumber", "stringNumber", FirstDigitTransform.Mode.INCLUDE_OTHER_CATEGORY)
                .removeAllColumnsExceptFor("stringNumber")
                .categoricalToOneHot("stringNumber")
                .reduce(new Reducer.Builder(ReduceOp.Sum).build())
                .build();

        JavaRDD<List<Writable>> rdd = sc.parallelize(in);


        List<List<Writable>> out = SparkTransformExecutor.execute(rdd, tp).collect();
        assertEquals(1, out.size());

        List<Writable> l = out.get(0);
        List<Writable> exp = Arrays.<Writable>asList(
                new IntWritable(0),  //0
                new IntWritable(0),  //1
                new IntWritable(3),  //2
                new IntWritable(0),  //3
                new IntWritable(0),  //4
                new IntWritable(0),  //5
                new IntWritable(1),  //6
                new IntWritable(2),  //7
                new IntWritable(1),  //8
                new IntWritable(0),  //9
                new IntWritable(1)); //Other
        assertEquals(exp, l);
    }

}
