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

package org.datavec.python;

import org.datavec.api.transform.TransformProcess;
import org.datavec.api.transform.condition.Condition;
import org.datavec.api.transform.filter.ConditionFilter;
import org.datavec.api.transform.filter.Filter;
import org.datavec.api.writable.*;
import org.datavec.api.transform.schema.Schema;
import org.junit.Test;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;


import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import static org.junit.Assert.assertEquals;
import static org.junit.Assert.assertFalse;
import static org.junit.Assert.assertTrue;


public class TestPythonTransformProcess {

    @Test(timeout = 60000L)
    public void testStringConcat() throws Exception{
        Schema.Builder schemaBuilder = new Schema.Builder();
        schemaBuilder
                .addColumnString("col1")
                .addColumnString("col2");

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnString("col3");
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";

        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                new PythonTransform(pythonCode, finalSchema)
        ).build();

        List<Writable> inputs = Arrays.asList((Writable) new Text("Hello "), new Text("World!"));

        List<Writable> outputs = tp.execute(inputs);
        assertEquals((outputs.get(0)).toString(), "Hello ");
        assertEquals((outputs.get(1)).toString(), "World!");
        assertEquals((outputs.get(2)).toString(), "Hello World!");

    }

    @Test(timeout = 60000L)
    public void testMixedTypes() throws Exception{
        Schema.Builder schemaBuilder = new Schema.Builder();
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
                new PythonTransform(pythonCode, finalSchema)
        ).build();

        List<Writable> inputs = Arrays.asList((Writable)
        new IntWritable(10),
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

        Schema.Builder schemaBuilder = new Schema.Builder();
        schemaBuilder
                .addColumnNDArray("col1", shape)
                .addColumnNDArray("col2", shape);

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnNDArray("col3", shape);
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";
        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                new PythonTransform(pythonCode, finalSchema)
        ).build();

        List<Writable> inputs = Arrays.asList(
                (Writable) new NDArrayWritable(arr1),
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

        Schema.Builder schemaBuilder = new Schema.Builder();
        schemaBuilder
                .addColumnNDArray("col1", shape)
                .addColumnNDArray("col2", shape);

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnNDArray("col3", shape);
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";
        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                new PythonTransform(pythonCode, finalSchema)
        ).build();

        List<Writable> inputs = Arrays.asList(
                (Writable) new NDArrayWritable(arr1),
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

        Schema.Builder schemaBuilder = new Schema.Builder();
        schemaBuilder
                .addColumnNDArray("col1", shape)
                .addColumnNDArray("col2", shape);

        Schema initialSchema = schemaBuilder.build();
        schemaBuilder.addColumnNDArray("col3", shape);
        Schema finalSchema = schemaBuilder.build();

        String pythonCode = "col3 = col1 + col2";
        TransformProcess tp = new TransformProcess.Builder(initialSchema).transform(
                new PythonTransform(pythonCode, finalSchema)
        ).build();

        List<Writable> inputs = Arrays.asList(
                (Writable) new NDArrayWritable(arr1),
                new NDArrayWritable(arr2)
        );

        List<Writable> outputs = tp.execute(inputs);
        assertEquals(arr1, ((NDArrayWritable)outputs.get(0)).get());
        assertEquals(arr2, ((NDArrayWritable)outputs.get(1)).get());
        assertEquals(expectedOutput,((NDArrayWritable)outputs.get(2)).get());

    }

    @Test(timeout = 60000L)
    public void testPythonFilter(){
        Schema schema = new Schema.Builder().addColumnInteger("column").build();

        Condition condition = new PythonCondition(
                "f = lambda: column < 0"
        );

        condition.setInputSchema(schema);

        Filter filter = new ConditionFilter(condition);

        assertFalse(filter.removeExample(Collections.singletonList((Writable) new IntWritable(10))));
        assertFalse(filter.removeExample(Collections.singletonList((Writable) new IntWritable(1))));
        assertFalse(filter.removeExample(Collections.singletonList((Writable) new IntWritable(0))));
        assertTrue(filter.removeExample(Collections.singletonList((Writable) new IntWritable(-1))));
        assertTrue(filter.removeExample(Collections.singletonList((Writable) new IntWritable(-10))));

    }

    @Test(timeout = 60000L)
    public void testPythonFilterAndTransform() throws Exception{
        Schema.Builder schemaBuilder = new Schema.Builder();
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
                new PythonTransform(
                        pythonCode,
                        finalSchema
                )
        ).filter(
                filter
        ).build();

        List<List<Writable>> inputs = new ArrayList<>();
        inputs.add(
                Arrays.asList((Writable) new IntWritable(5),
                        new FloatWritable(3.0f),
                        new Text("abcd"),
                        new DoubleWritable(2.1))
        );
        inputs.add(
                Arrays.asList((Writable) new IntWritable(-3),
                        new FloatWritable(3.0f),
                        new Text("abcd"),
                        new DoubleWritable(2.1))
        );
        inputs.add(
                Arrays.asList((Writable) new IntWritable(5),
                        new FloatWritable(11.2f),
                        new Text("abcd"),
                        new DoubleWritable(2.1))
        );

    }
}
