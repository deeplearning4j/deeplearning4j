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

package org.datavec.common;

import org.datavec.common.data.NDArrayWritable;
import org.junit.Test;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import java.io.*;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 02/06/2017.
 */
public class TestWritableSerialization {

    @Test
    public void testWritableSerialization() throws Exception {

        INDArray arrC = Nd4j.rand(new int[]{10,20},'c');
        INDArray arrF = Nd4j.rand(new int[]{10,20},'f');


        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        DataOutput da = new DataOutputStream(baos);

        NDArrayWritable wC = new NDArrayWritable(arrC);
        NDArrayWritable wF = new NDArrayWritable(arrF);

        wC.write(da);
        wF.write(da);

        byte[] b = baos.toByteArray();

        NDArrayWritable w2C = new NDArrayWritable();
        NDArrayWritable w2F = new NDArrayWritable();

        ByteArrayInputStream bais = new ByteArrayInputStream(b);
        DataInput din = new DataInputStream(bais);

        w2C.readFields(din);
        w2F.readFields(din);


        assertEquals(arrC, w2C.get());
        assertEquals(arrF, w2F.get());
    }

}
