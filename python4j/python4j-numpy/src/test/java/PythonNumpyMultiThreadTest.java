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

import org.nd4j.python4j.*;
import org.junit.Assert;
import org.junit.jupiter.api.Test;
import org.junit.runner.RunWith;
import org.junit.runners.Parameterized;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.concurrent.NotThreadSafe;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;


@NotThreadSafe
@RunWith(Parameterized.class)
public class PythonNumpyMultiThreadTest {
    private DataType dataType;

    public PythonNumpyMultiThreadTest(DataType dataType) {
        this.dataType = dataType;
    }

    @Parameterized.Parameters(name = "{index}: Testing with DataType={0}")
    public static DataType[] params() {
        return new DataType[]{
//                DataType.BOOL,
//                DataType.FLOAT16,
//                DataType.BFLOAT16,
                DataType.FLOAT,
                DataType.DOUBLE,
//                DataType.INT8,
//                DataType.INT16,
                DataType.INT32,
                DataType.INT64,
//                DataType.UINT8,
//                DataType.UINT16,
//                DataType.UINT32,
//                DataType.UINT64
        };
    }


    @Test
    public void testMultiThreading1() throws Throwable {
        final List<Throwable> exceptions = Collections.synchronizedList(new ArrayList<Throwable>());
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                try (PythonGIL gil = PythonGIL.lock()) {
                    try (PythonGC gc = PythonGC.watch()) {
                        List<PythonVariable> inputs = new ArrayList<>();
                        inputs.add(new PythonVariable<>("x", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(3)));
                        inputs.add(new PythonVariable<>("y", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(4)));
                        PythonVariable out = new PythonVariable<>("z", NumpyArray.INSTANCE);
                        String code = "z = x + y";
                        PythonExecutioner.exec(code, inputs, Collections.singletonList(out));
                        Assert.assertEquals(Nd4j.ones(dataType, 2, 3).mul(7), out.getValue());
                    }
                } catch (Throwable e) {
                    exceptions.add(e);
                }
            }
        };

        int numThreads = 10;
        Thread[] threads = new Thread[numThreads];
        for (int i = 0; i < threads.length; i++) {
            threads[i] = new Thread(runnable);
        }
        for (int i = 0; i < threads.length; i++) {
            threads[i].start();
        }
        Thread.sleep(100);
        for (int i = 0; i < threads.length; i++) {
            threads[i].join();
        }
        if (!exceptions.isEmpty()) {
            throw (exceptions.get(0));
        }

    }

    @Test
    public void testMultiThreading2() throws Throwable {
        final List<Throwable> exceptions = Collections.synchronizedList(new ArrayList<Throwable>());
        Runnable runnable = new Runnable() {
            @Override
            public void run() {
                try (PythonGIL gil = PythonGIL.lock()) {
                    try (PythonGC gc = PythonGC.watch()) {
                        PythonContextManager.reset();
                        List<PythonVariable> inputs = new ArrayList<>();
                        inputs.add(new PythonVariable<>("x", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(3)));
                        inputs.add(new PythonVariable<>("y", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(4)));
                        String code = "z = x + y";
                        List<PythonVariable> outputs = PythonExecutioner.execAndReturnAllVariables(code, inputs);
                        Assert.assertEquals(Nd4j.ones(dataType, 2, 3).mul(3), outputs.get(0).getValue());
                        Assert.assertEquals(Nd4j.ones(dataType, 2, 3).mul(4), outputs.get(1).getValue());
                        Assert.assertEquals(Nd4j.ones(dataType, 2, 3).mul(7), outputs.get(2).getValue());
                    }
                } catch (Throwable e) {
                    exceptions.add(e);
                }
            }
        };

        int numThreads = 10;
        Thread[] threads = new Thread[numThreads];
        for (int i = 0; i < threads.length; i++) {
            threads[i] = new Thread(runnable);
        }
        for (int i = 0; i < threads.length; i++) {
            threads[i].start();
        }
        Thread.sleep(100);
        for (int i = 0; i < threads.length; i++) {
            threads[i].join();
        }
        if (!exceptions.isEmpty()) {
            throw (exceptions.get(0));
        }
    }


}
