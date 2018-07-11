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

package org.datavec.local.transforms.transform;


import org.datavec.api.transform.MathFunction;
import org.datavec.api.transform.MathOp;
import org.datavec.api.transform.ReduceOp;
import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.ConditionOp;
import org.datavec.api.transform.condition.column.DoubleColumnCondition;
import org.datavec.api.transform.reduce.Reducer;
import org.datavec.api.transform.schema.Schema;
import org.datavec.api.transform.schema.SequenceSchema;
import org.datavec.api.writable.*;

import org.datavec.arrow.recordreader.ArrowWritableRecordTimeSeriesBatch;
import org.datavec.local.transforms.LocalTransformExecutor;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cpu.nativecpu.NDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.transforms.Transforms;

import java.util.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 25/11/2016.
 */
public class ExecutionTest  {

    @Test
    public void testExecutionNdarray() {
        Schema schema = new Schema.Builder()
                .addColumnNDArray("first",new long[]{1,32577})
                .addColumnNDArray("second",new long[]{1,32577}).build();

        TransformProcess transformProcess = new TransformProcess.Builder(schema)
                .ndArrayMathFunctionTransform("first", MathFunction.SIN)
                .ndArrayMathFunctionTransform("second",MathFunction.COS)
                .build();

        List<List<Writable>> functions = new ArrayList<>();
        List<Writable> firstRow = new ArrayList<>();
        INDArray firstArr = Nd4j.linspace(1,4,4);
        INDArray secondArr = Nd4j.linspace(1,4,4);
        firstRow.add(new NDArrayWritable(firstArr));
        firstRow.add(new NDArrayWritable(secondArr));
        functions.add(firstRow);

        List<List<Writable>> execute = LocalTransformExecutor.execute(functions, transformProcess);
        INDArray firstResult = ((NDArrayWritable) execute.get(0).get(0)).get();
        INDArray secondResult = ((NDArrayWritable) execute.get(0).get(1)).get();

        INDArray expected = Transforms.sin(firstArr);
        INDArray secondExpected = Transforms.cos(secondArr);
        assertEquals(expected,firstResult);
        assertEquals(secondExpected,secondResult);

    }

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

        List<List<Writable>> rdd = (inputData);

        List<List<Writable>> out = new ArrayList<>(LocalTransformExecutor.execute(rdd, tp));

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
    public void testFilter() {
        Schema filterSchema = new Schema.Builder()
                .addColumnDouble("col1").addColumnDouble("col2")
                .addColumnDouble("col3").build();
        List<List<Writable>> inputData = new ArrayList<>();
        inputData.add(Arrays.<Writable>asList(new IntWritable(0), new DoubleWritable(1), new DoubleWritable(0.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(1), new DoubleWritable(3), new DoubleWritable(1.1)));
        inputData.add(Arrays.<Writable>asList(new IntWritable(2), new DoubleWritable(3), new DoubleWritable(2.1)));
        TransformProcess transformProcess = new TransformProcess.Builder(filterSchema)
                .filter(new DoubleColumnCondition("col1",ConditionOp.LessThan,1)).build();
        List<List<Writable>> execute = LocalTransformExecutor.execute(inputData, transformProcess);
        assertEquals(2,execute.size());
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

        List<List<List<Writable>>> rdd =  (inputSequences);

        List<List<List<Writable>>> out = LocalTransformExecutor.executeSequenceToSequence(rdd, tp);

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

        List<List<Writable>> inData = in;

        Schema s = new Schema.Builder()
                .addColumnString("textCol")
                .addColumnDouble("doubleCol")
                .build();

        TransformProcess tp = new TransformProcess.Builder(s)
                .reduce(new Reducer.Builder(ReduceOp.TakeFirst)
                        .takeFirstColumns("textCol")
                        .meanColumns("doubleCol").build())
                .build();

        List<List<Writable>> outRdd = LocalTransformExecutor.execute(inData, tp);

        List<List<Writable>> out = outRdd;

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

        List<List<Writable>> inData = in;

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

        List<List<Writable>> outRdd = LocalTransformExecutor.execute(inData, tp);

        List<List<Writable>> out = outRdd;

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

}
