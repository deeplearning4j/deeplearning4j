/*
 *  ******************************************************************************
 *  *
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
 *  *  See the NOTICE file distributed with this work for additional
 *  *  information regarding copyright ownership.
 *  * Unless required by applicable law or agreed to in writing, software
 *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  * License for the specific language governing permissions and limitations
 *  * under the License.
 *  *
 *  * SPDX-License-Identifier: Apache-2.0
 *  *****************************************************************************
 */

package org.nd4j.linalg.memory;

import lombok.extern.slf4j.Slf4j;
import lombok.val;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;
import org.nd4j.linalg.util.DeviceLocalNDArray;

import java.util.Arrays;
import java.util.concurrent.atomic.AtomicInteger;

import static org.junit.jupiter.api.Assertions.assertArrayEquals;
import static org.junit.jupiter.api.Assertions.assertEquals;

@Slf4j
@RunWith(Parameterized.class)
public class DeviceLocalNDArrayTests extends BaseNd4jTest {

    public DeviceLocalNDArrayTests(Nd4jBackend backend) {
        super(backend);
    }

    @Test
    public void testDeviceLocalStringArray(){
        val arr = Nd4j.create(Arrays.asList("first", "second"), 2);
        assertEquals(DataType.UTF8, arr.dataType());
        assertArrayEquals(new long[]{2}, arr.shape());

        val dl = new DeviceLocalNDArray(arr);

        for (int e = 0; e < Nd4j.getAffinityManager().getNumberOfDevices(); e++) {
            val arr2 = dl.get(e);
            assertEquals(arr, arr2);
        }
    }

    @Test
    public void testDtypes(){
        for(DataType globalDType : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
            Nd4j.setDefaultDataTypes(globalDType, globalDType);
            for(DataType arrayDtype : new DataType[]{DataType.DOUBLE, DataType.FLOAT, DataType.HALF}){
                INDArray arr = Nd4j.linspace(arrayDtype, 1, 10, 1);
                DeviceLocalNDArray dl = new DeviceLocalNDArray(arr);
                INDArray get = dl.get();
                assertEquals(arr, get);
            }
        }
    }

    @Test
    public void testDeviceLocalUpdate_1() throws Exception {
        val numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        if (numDevices < 2)
            return;

        val array = Nd4j.createFromArray(1.f, 2.f, 3.f, 4.f);

        val deviceLocal = new DeviceLocalNDArray(array);
        for (int e = 0; e < numDevices; e++) {
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    deviceLocal.get().add(1.f);
                    Nd4j.getExecutioner().commit();;
                }
            });

            t.start();
            t.join();
        }

        val counter = new AtomicInteger(0);

        val update = Nd4j.createFromArray(5.f, 5.f, 5.f, 5.f);
        deviceLocal.update(update);

        for (int e = 0; e < numDevices; e++) {
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    assertEquals(5.f, deviceLocal.get().meanNumber().floatValue(), 1e-5f);
                    counter.incrementAndGet();
                }
            });

            t.start();
            t.join();
        }

        assertEquals(numDevices, counter.get());
    }


    @Test
    public void testDelayedDeviceLocalUpdate_1() throws Exception {
        val numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        if (numDevices < 2)
            return;

        val array = Nd4j.createFromArray(5.f, 5.f, 5.f, 5.f);

        val deviceLocal = new DeviceLocalNDArray(array, true);
        val counter = new AtomicInteger(0);

        for (int e = 0; e < numDevices; e++) {
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    assertEquals(5.f, deviceLocal.get().meanNumber().floatValue(), 1e-5f);
                    counter.incrementAndGet();
                }
            });

            t.start();
            t.join();
        }

        assertEquals(numDevices, counter.get());
    }

    @Test
    public void testDelayedDeviceLocalUpdate_2() throws Exception {
        val numDevices = Nd4j.getAffinityManager().getNumberOfDevices();
        if (numDevices < 2)
            return;

        val array = Nd4j.createFromArray(5.f, 5.f, 5.f, 5.f);

        val deviceLocal = new DeviceLocalNDArray(array, true);
        val counter = new AtomicInteger(0);

        deviceLocal.update(Nd4j.createFromArray(4.f, 4.f, 4.f, 4.f));

        for (int e = 0; e < numDevices; e++) {
            val t = new Thread(new Runnable() {
                @Override
                public void run() {
                    assertEquals(4.f, deviceLocal.get().meanNumber().floatValue(), 1e-5f);
                    counter.incrementAndGet();
                }
            });

            t.start();
            t.join();
        }

        assertEquals(numDevices, counter.get());
    }

    @Override
    public char ordering() {
        return 'c';
    }
}
