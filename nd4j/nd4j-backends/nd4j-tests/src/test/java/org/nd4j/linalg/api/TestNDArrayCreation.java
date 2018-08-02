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

package org.nd4j.linalg.api;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.Pointer;
import org.junit.Ignore;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.io.ClassPathResource;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.io.File;

import static org.junit.Assert.assertEquals;

/**
 * Created by Alex on 30/04/2016.
 */
@Slf4j
public class TestNDArrayCreation extends BaseNd4jTest {


    public TestNDArrayCreation(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testBufferCreation() {
        DataBuffer dataBuffer = Nd4j.createBuffer(new double[] {1, 2});
        Pointer pointer = dataBuffer.pointer();
        FloatPointer floatPointer = new FloatPointer(pointer);
        DataBuffer dataBuffer1 = Nd4j.createBuffer(floatPointer, 2);
        assertEquals(2, dataBuffer1.length());
        assertEquals(1.0, dataBuffer1.getDouble(0), 1e-1);
        assertEquals(2.0, dataBuffer1.getDouble(1), 1e-1);
        INDArray arr = Nd4j.create(dataBuffer1);
        System.out.println(arr);
    }


    @Test
    @Ignore
    public void testCreateNpy() throws Exception {
        INDArray arrCreate = Nd4j.createFromNpyFile(new ClassPathResource("nd4j-tests/test.npy").getFile());
        assertEquals(2, arrCreate.size(0));
        assertEquals(2, arrCreate.size(1));
        assertEquals(1.0, arrCreate.getDouble(0, 0), 1e-1);
        assertEquals(2.0, arrCreate.getDouble(0, 1), 1e-1);
        assertEquals(3.0, arrCreate.getDouble(1, 0), 1e-1);
        assertEquals(4.0, arrCreate.getDouble(1, 1), 1e-1);

    }

    @Test
    public void testCreateNpy3() throws Exception {
        INDArray arrCreate = Nd4j.createFromNpyFile(new ClassPathResource("nd4j-tests/rank3.npy").getFile());
        assertEquals(8, arrCreate.length());
        assertEquals(3, arrCreate.rank());

        Pointer pointer = NativeOpsHolder.getInstance().getDeviceNativeOps()
                        .pointerForAddress(arrCreate.data().address());
        assertEquals(arrCreate.data().address(), pointer.address());
    }


    @Override
    public char ordering() {
        return 'c';
    }
}
