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
package org.nd4j.python4j;

import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.params.ParameterizedTest;
import org.junit.jupiter.params.provider.MethodSource;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

import javax.annotation.concurrent.NotThreadSafe;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.Callable;
import java.util.concurrent.ExecutorService;
import java.util.concurrent.Executors;
import java.util.concurrent.TimeUnit;

import static org.junit.jupiter.api.Assertions.assertEquals;
import static org.junit.jupiter.api.Assertions.assertNull;

@NotThreadSafe
@Tag(TagNames.FILE_IO)
@NativeTag
@Tag(TagNames.PYTHON)
public class PythonInterpreterTests {

    @Test
    public void testBasicExecution() {
        PythonInterpreter pythonInterpreter = InitializingPythonInterpreter.getInstance();
        testBasicExecution(pythonInterpreter);
    }

    @Test
    public void testMultiThreadedExecution() throws Exception {
        ExecutorService executorService = Executors.newFixedThreadPool(2);
        PythonInterpreter initializingPythonInterpreter =  InitializingPythonInterpreter.getInstance();
        List<Callable<Void>> tasks = new ArrayList<>();
        for(int i = 0; i < 2; i++) {
            tasks.add(() -> {
                testBasicExecution(initializingPythonInterpreter);
                return null;
            });
        }
        executorService.invokeAll(tasks);

    }

    private void testBasicExecution(PythonInterpreter interpreter) {
        testNull(interpreter);
        testAddInt(interpreter);
        testAddDouble(interpreter);
        testNumpyAdd(interpreter);
    }


    private void testNumpyAdd(PythonInterpreter pythonInterpreter) {
        pythonInterpreter.set("a", Nd4j.ones(2).castTo(DataType.INT64));
        pythonInterpreter.exec("a += 2");
        assertEquals(Nd4j.createFromArray(new long[]{3,3}),(INDArray) pythonInterpreter.get("a"));
    }

    private void testAddDouble(PythonInterpreter interpreter) {
        interpreter.set("a",2.0);
        interpreter.exec("a += 2");
        assertEquals(interpreter.get("a"),4.0);
    }
    private void testAddInt(PythonInterpreter interpreter) {
        interpreter.set("a",2);
        interpreter.exec("a += 2");
        assertEquals(interpreter.get("a"),(long) 4);
    }

    private void testNull(PythonInterpreter interpreter) {
        interpreter.set("some_none",null);
        Object some_none = interpreter.get("some_none");
        assertNull(some_none);

    }

}
