/*
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

package org.datavec.api.transform;

import org.datavec.api.transform.schema.Schema;
import org.datavec.api.writable.DoubleWritable;
import org.datavec.api.writable.Text;
import org.datavec.api.writable.Writable;
import org.datavec.common.data.NDArrayWritable;
import org.junit.Test;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 02/06/2017.
 */
public class TestNDArrayWritableTransforms {

    @Test
    public void testNDArrayWritableBasic(){

        Schema s = new Schema.Builder()

                .addColumnDouble("col0")
                .addColumnNDArray("col1", new int[]{1,10})
                .addColumnString("col2")
                .build();


        TransformProcess tp = new TransformProcess.Builder(s)
                .ndArrayMathOpTransform("col1",MathOp.Add, 100)
                .build();

        List<Writable> in = Arrays.<Writable>asList(new DoubleWritable(0), new NDArrayWritable(Nd4j.linspace(0,9,10)), new Text("str0"));
        List<Writable> out = tp.execute(in);

        List<Writable> exp = Arrays.<Writable>asList(new DoubleWritable(0), new NDArrayWritable(Nd4j.linspace(0,9,10).addi(100)), new Text("str0"));

        assertEquals(exp, out);

    }

    @Test
    public void testNDArrayColumnsMathOpTransform(){

        Schema s = new Schema.Builder()

                .addColumnDouble("col0")
                .addColumnNDArray("col1", new int[]{1,10})
                .addColumnNDArray("col2", new int[]{1,10})
                .build();


        TransformProcess tp = new TransformProcess.Builder(s)
                .ndArrayColumnsMathOpTransform("myCol", MathOp.Add, "col1", "col2")
                .build();

        List<String> expColNames = Arrays.asList("col0", "col1", "col2", "myCol");
        assertEquals(expColNames, tp.getFinalSchema().getColumnNames());


        List<Writable> in = Arrays.<Writable>asList(new DoubleWritable(0), new NDArrayWritable(Nd4j.linspace(0,9,10)),
                new NDArrayWritable(Nd4j.valueArrayOf(1,10,2.0)));
        List<Writable> out = tp.execute(in);

        List<Writable> exp = Arrays.<Writable>asList(new DoubleWritable(0), new NDArrayWritable(Nd4j.linspace(0,9,10)),
                new NDArrayWritable(Nd4j.valueArrayOf(1,10,2.0)), new NDArrayWritable(Nd4j.linspace(0,9,10).addi(2.0)));

        assertEquals(exp, out);

    }

}
