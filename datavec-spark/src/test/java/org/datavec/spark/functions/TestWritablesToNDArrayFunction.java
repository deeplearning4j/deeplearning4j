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

package org.datavec.spark.functions;

import org.datavec.api.writable.*;
import org.datavec.spark.transform.misc.SequenceWritablesToStringFunction;
import org.datavec.spark.transform.misc.WritablesToNDArrayFunction;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import static org.junit.Assert.assertEquals;

public class TestWritablesToNDArrayFunction {

    @Test
    public void testWritablesToNDArrayAllScalars() throws Exception {
        List<Writable> l = new ArrayList<>();
        for (int i = 0; i < 5; i++)
            l.add(new IntWritable(i));
        INDArray expected = Nd4j.arange(5);
        assertEquals(expected, new WritablesToNDArrayFunction().call(l));
    }

    @Test
    public void testWritablesToNDArrayMixed() throws Exception {
        List<Writable> l = new ArrayList<>();
        l.add(new IntWritable(0));
        l.add(new IntWritable(1));
        INDArray arr = Nd4j.arange(2, 5);
        l.add(new NDArrayWritable(arr));
        l.add(new IntWritable(5));
        arr = Nd4j.arange(6, 9);
        l.add(new NDArrayWritable(arr));
        l.add(new IntWritable(9));

        INDArray expected = Nd4j.arange(10);
        assertEquals(expected, new WritablesToNDArrayFunction().call(l));
    }
}
