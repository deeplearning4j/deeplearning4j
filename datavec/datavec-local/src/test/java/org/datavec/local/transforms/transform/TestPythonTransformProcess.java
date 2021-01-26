/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
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

package org.datavec.local.transforms.transform;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.transform.schema.Schema;
import org.datavec.local.transforms.LocalTransformExecutor;

import org.datavec.api.writable.*;
import org.datavec.python.PythonCondition;
import org.datavec.python.PythonTransform;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.concurrent.NotThreadSafe;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;

import static junit.framework.TestCase.assertTrue;
import static org.datavec.api.transform.schema.Schema.Builder;
import static org.junit.Assert.*;

@NotThreadSafe
public class TestPythonTransformProcess {


    @Test()
    public void testStringConcat() throws Exception{
        Builder schemaBuilder = new Builder();
        schemaBuilder
                .addColumnString("col1")
                .addColumnString("col2");

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnString("col3");
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";

        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                PythonTransform.builder().code(pythonCode)
                        .outputSchema(finalSchema)
                        .build()
        ).build();

        List<Writable> inputs = Arrays.asList((Writable)new Text("Hello "), new Text("World!"));

        List<Writable> outputs = tp.execute(inputs);
        assertEquals((outputs.get(0)).toString(), "Hello ");
        assertEquals((outputs.get(1)).toString(), "World!");
        assertEquals((outputs.get(2)).toString(), "Hello World!");

    }

    @Test(timeout = 60000L)
    public void testMixedTypes() throws Exception{
        Builder schemaBuilder = new Builder();
        schemaBuilder
                .addColumnInteger("col1")
                .addColumnFloat("col2")
                .addColumnString("col3")
                .addColumnDouble("col4");


        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnInteger("col5");
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col5  = (int(col3) + col1 + int(col2)) * int(col4)";

        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                PythonTransform.builder().code(pythonCode)
                        .outputSchema(finalSchema)
                        .inputSchema(initialSchema)
                        .build()        ).build();

        List<Writable> inputs = Arrays.asList((Writable)new IntWritable(10),
                new FloatWritable(3.5f),
                new Text("5"),
                new DoubleWritable(2.0)
        );

        List<Writable> outputs = tp.execute(inputs);
        assertEquals(((LongWritable)outputs.get(4)).get(), 36);
    }

    @Test(timeout = 60000L)
    public void testNDArray() throws Exception{
        long[] shape = new long[]{3, 2};
        INDArray arr1 = Nd4j.rand(shape);
        INDArray arr2 = Nd4j.rand(shape);

        INDArray expectedOutput = arr1.add(arr2);

        Builder schemaBuilder = new Builder();
        schemaBuilder
                .addColumnNDArray("col1", shape)
                .addColumnNDArray("col2", shape);

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnNDArray("col3", shape);
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";
        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                PythonTransform.builder().code(pythonCode)
                        .outputSchema(finalSchema)
                        .build()        ).build();

        List<Writable> inputs = Arrays.asList(
                (Writable)
                new NDArrayWritable(arr1),
                new NDArrayWritable(arr2)
        );

        List<Writable> outputs = tp.execute(inputs);
        assertEquals(arr1, ((NDArrayWritable)outputs.get(0)).get());
        assertEquals(arr2, ((NDArrayWritable)outputs.get(1)).get());
        assertEquals(expectedOutput,((NDArrayWritable)outputs.get(2)).get());

    }

    @Test(timeout = 60000L)
    public void testNDArray2() throws Exception{
        long[] shape = new long[]{3, 2};
        INDArray arr1 = Nd4j.rand(shape);
        INDArray arr2 = Nd4j.rand(shape);

        INDArray expectedOutput = arr1.add(arr2);

        Builder schemaBuilder = new Builder();
        schemaBuilder
                .addColumnNDArray("col1", shape)
                .addColumnNDArray("col2", shape);

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnNDArray("col3", shape);
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";
        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                PythonTransform.builder().code(pythonCode)
                        .outputSchema(finalSchema)
                        .build()        ).build();

        List<Writable> inputs = Arrays.asList(
                (Writable)
                new NDArrayWritable(arr1),
                new NDArrayWritable(arr2)
        );

        List<Writable> outputs = tp.execute(inputs);
        assertEquals(arr1, ((NDArrayWritable)outputs.get(0)).get());
        assertEquals(arr2, ((NDArrayWritable)outputs.get(1)).get());
        assertEquals(expectedOutput,((NDArrayWritable)outputs.get(2)).get());

    }

    @Test(timeout = 60000L)
    public void testNDArrayMixed() throws Exception{
        long[] shape = new long[]{3, 2};
        INDArray arr1 = Nd4j.rand(DataType.DOUBLE, shape);
        INDArray arr2 = Nd4j.rand(DataType.DOUBLE, shape);
        INDArray expectedOutput = arr1.add(arr2.castTo(DataType.DOUBLE));

        Builder schemaBuilder = new Builder();
        schemaBuilder
                .addColumnNDArray("col1", shape)
                .addColumnNDArray("col2", shape);

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnNDArray("col3", shape);
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";
        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                PythonTransform.builder().code(pythonCode)
                        .outputSchema(finalSchema)
                        .build()
        ).build();

        List<Writable> inputs = Arrays.asList(
                (Writable)
                new NDArrayWritable(arr1),
                new NDArrayWritable(arr2)
        );

        List<Writable> outputs = tp.execute(inputs);
        assertEquals(arr1, ((NDArrayWritable)outputs.get(0)).get());
        assertEquals(arr2, ((NDArrayWritable)outputs.get(1)).get());
        assertEquals(expectedOutput,((NDArrayWritable)outputs.get(2)).get());

    }

    @Test(timeout = 60000L)
    public void testPythonFilter() {
        Schema schema = new Builder().addColumnInteger("column").build();

        Condition condition = new PythonCondition(
                "f = lambda: column < 0"
        );

        condition.setInputSchema(schema);

        Filter filter = new ConditionFilter(condition);

        assertFalse(filter.removeExample(Collections.singletonList(new IntWritable(10))));
        assertFalse(filter.removeExample(Collections.singletonList(new IntWritable(1))));
        assertFalse(filter.removeExample(Collections.singletonList(new IntWritable(0))));
        assertTrue(filter.removeExample(Collections.singletonList(new IntWritable(-1))));
        assertTrue(filter.removeExample(Collections.singletonList(new IntWritable(-10))));

    }

    @Test(timeout = 60000L)
    public void testPythonFilterAndTransform() throws Exception{
        Builder schemaBuilder = new Builder();
        schemaBuilder
                .addColumnInteger("col1")
                .addColumnFloat("col2")
                .addColumnString("col3")
                .addColumnDouble("col4");

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnString("col6");
        Schema finalSchema = schemaBuilder.build();

        Condition condition = new PythonCondition(
                "f = lambda: col1 < 0 and col2 > 10.0"
        );

        condition.setInputSchema(initialSchema);

        Filter filter = new ConditionFilter(condition);

        String pythonCode = "col6 = str(col1 + col2)";
        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                PythonTransform.builder().code(pythonCode)
                        .outputSchema(finalSchema)
                        .build()
        ).filter(
                filter
        ).build();

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(
                Arrays.asList(
                        (Writable)
                        new IntWritable(5),
                        new FloatWritable(3.0f),
                        new Text("abcd"),
                        new DoubleWritable(2.1))
        );
        inputs.add(
                Arrays.asList(
                        (Writable)
                        new IntWritable(-3),
                        new FloatWritable(3.0f),
                        new Text("abcd"),
                        new DoubleWritable(2.1))
        );
        inputs.add(
                Arrays.asList(
                        (Writable)
                        new IntWritable(5),
                        new FloatWritable(11.2f),
                        new Text("abcd"),
                        new DoubleWritable(2.1))
        );

        LocalTransformExecutor.execute(inputs,tp);
    }


    @Test
    public void testPythonTransformNoOutputSpecified() throws Exception {
        PythonTransform pythonTransform = PythonTransform.builder()
                .code("a += 2; b = 'hello world'")
                .returnAllInputs(true)
                .build();
        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable)new IntWritable(1)));
        Schema inputSchema = new Builder()
                .addColumnInteger("a")
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputSchema)
                .transform(pythonTransform)
                .build();
        List<List<Writable>> execute = LocalTransformExecutor.execute(inputs, tp);
        assertEquals(3,execute.get(0).get(0).toInt());
        assertEquals("hello world",execute.get(0).get(1).toString());

    }

    @Test
    public void testNumpyTransform() {
        PythonTransform pythonTransform = PythonTransform.builder()
                .code("a += 2; b = 'hello world'")
                .returnAllInputs(true)
                .build();

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new NDArrayWritable(Nd4j.scalar(1).reshape(1,1))));
        Schema inputSchema = new Builder()
                .addColumnNDArray("a",new long[]{1,1})
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputSchema)
                .transform(pythonTransform)
                .build();
        List<List<Writable>> execute = LocalTransformExecutor.execute(inputs, tp);
        assertFalse(execute.isEmpty());
        assertNotNull(execute.get(0));
        assertNotNull(execute.get(0).get(0));
        assertNotNull(execute.get(0).get(1));
        assertEquals(Nd4j.scalar(3).reshape(1, 1),((NDArrayWritable)execute.get(0).get(0)).get());
        assertEquals("hello world",execute.get(0).get(1).toString());
    }

    @Test
    public void testWithSetupRun() throws Exception {

        PythonTransform pythonTransform = PythonTransform.builder()
                .code("five=None\n" +
                        "def setup():\n" +
                        "    global five\n"+
                        "    five = 5\n\n" +
                        "def run(a, b):\n" +
                        "    c = a + b + five\n"+
                        "    return {'c':c}\n\n")
                .returnAllInputs(true)
                .setupAndRun(true)
                .build();

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(Arrays.asList((Writable) new NDArrayWritable(Nd4j.scalar(1).reshape(1,1)),
                new NDArrayWritable(Nd4j.scalar(2).reshape(1,1))));
        Schema inputSchema = new Builder()
                .addColumnNDArray("a",new long[]{1,1})
                .addColumnNDArray("b", new long[]{1, 1})
                .build();

        TransformProcess tp = new TransformProcess.Builder(inputSchema)
                .transform(pythonTransform)
                .build();
        List<List<Writable>> execute = LocalTransformExecutor.execute(inputs, tp);
        assertFalse(execute.isEmpty());
        assertNotNull(execute.get(0));
        assertNotNull(execute.get(0).get(0));
        assertEquals(Nd4j.scalar(8).reshape(1, 1),((NDArrayWritable)execute.get(0).get(3)).get());
    }

}