/*
 *
 *  *  ******************************************************************************
 *  *  *
 *  *  *
 *  *  * This program and the accompanying materials are made available under the
 *  *  * terms of the Apache License, Version 2.0 which is available at
 *  *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *  *
 *  *  *  See the NOTICE file distributed with this work for additional
 *  *  *  information regarding copyright ownership.
 *  *  * Unless required by applicable law or agreed to in writing, software
 *  *  * distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
 *  *  * WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
 *  *  * License for the specific language governing permissions and limitations
 *  *  * under the License.
 *  *  *
 *  *  * SPDX-License-Identifier: Apache-2.0
 *  *  *****************************************************************************
 *
 *
 */

package org.nd4j.python4j;/*
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

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.Arguments;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.python4j.*;

import org.junit.jupiter.api.Test;

import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.concurrent.NotThreadSafe;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;
import java.util.stream.Stream;

import static org.junit.jupiter.api.Assertions.assertEquals;


@NotThreadSafe
@Tag(TagNames.FILE_IO)
@NativeTag
@Tag(TagNames.PYTHON)
public class PythonNumpyMultiThreadTest {

    public static Stream<Arguments> params() {
        return Arrays.asList(new DataType[]{
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
        }).stream().map(Arguments::of);
    }


    @MethodSource("org.nd4j.python4j.PythonNumpyMultiThreadTest#params")
    @ParameterizedTest
    public void testMultiThreading1(DataType dataType) throws Throwable {
        final List<Throwable> exceptions = Collections.synchronizedList(new ArrayList<Throwable>());
        Runnable runnable = () -> {
            try (PythonGIL gil = PythonGIL.lock()) {
                try (PythonGC gc = PythonGC.watch()) {
                    List<PythonVariable> inputs = new ArrayList<>();
                    inputs.add(new PythonVariable<>("x", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(3)));
                    inputs.add(new PythonVariable<>("y", NumpyArray.INSTANCE, Nd4j.ones(dataType, 2, 3).mul(4)));
                    PythonVariable out = new PythonVariable<>("z", NumpyArray.INSTANCE);
                    String code = "z = x + y";
                    PythonExecutioner.exec(code, inputs, Collections.singletonList(out));
                    assertEquals(Nd4j.ones(dataType, 2, 3).mul(7), out.getValue());
                }
            } catch (Throwable e) {
                exceptions.add(e);
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

    @MethodSource("org.nd4j.python4j.PythonNumpyMultiThreadTest#params")
    @ParameterizedTest
    public void testMultiThreading2(DataType dataType) throws Throwable {
        final List<Throwable> exceptions = Collections.synchronizedList(new ArrayList<>());
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
                        assertEquals(Nd4j.ones(dataType, 2, 3).mul(3), outputs.get(0).getValue());
                        assertEquals(Nd4j.ones(dataType, 2, 3).mul(4), outputs.get(1).getValue());
                        assertEquals(Nd4j.ones(dataType, 2, 3).mul(7), outputs.get(2).getValue());
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
