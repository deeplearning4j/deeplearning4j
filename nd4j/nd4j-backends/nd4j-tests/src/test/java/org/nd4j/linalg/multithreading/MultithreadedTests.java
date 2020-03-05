/*******************************************************************************
 * Copyright (c) 2019 Konduit K.K.
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

package org.nd4j.linalg.multithreading;

import lombok.val;
import org.junit.Test;
import org.nd4j.linalg.BaseNd4jTest;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.factory.Nd4jBackend;

import java.util.ArrayList;
import java.util.HashSet;

import static org.junit.Assert.assertEquals;

/**
 * @author raver119@gmail.com
 */
public class MultithreadedTests extends BaseNd4jTest {

    public MultithreadedTests(Nd4jBackend backend) {
        super(backend);
    }

    @Override
    public char ordering() {
        return 'c';
    }

    @Test
    public void basicMigrationTest_1() throws Exception {
        if (Nd4j.getAffinityManager().getNumberOfDevices() < 2)
            return;

        val exp = Nd4j.create(DataType.INT32, 5, 5).assign(2);

        val hash = new HashSet<Integer>();

        // we're creating bunch of arrays on different devices
        val list = new ArrayList<INDArray>();
        for (int e = 0; e < Nd4j.getAffinityManager().getNumberOfDevices(); e++) {
            val t = e;
            val thread = new Thread(new Runnable() {
                @Override
                public void run() {
                    for (int f = 0; f < 10; f++) {
                        val array = Nd4j.create(DataType.INT32, 5, 5).assign(1);

                        // store current deviceId for further validation
                        hash.add(Nd4j.getAffinityManager().getDeviceForCurrentThread());

                        // make sure INDArray has proper affinity set
                        assertEquals(Nd4j.getAffinityManager().getDeviceForCurrentThread(), Nd4j.getAffinityManager().getDeviceForArray(array));

                        list.add(array);
                    }
                };
            });

            thread.start();
            thread.join();
        }

        // lets make sure all devices covered
        assertEquals(Nd4j.getAffinityManager().getNumberOfDevices(), hash.size());

        // make sure nothing failed in threads
        assertEquals(10 * Nd4j.getAffinityManager().getNumberOfDevices(), list.size());

        // now we're going to use arrays on current device, so data will be migrated
        for (val arr:list) {
            arr.addi(1);

            assertEquals(exp, arr);
        }
    }
}
