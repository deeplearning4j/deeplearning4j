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

import org.junit.jupiter.api.Disabled;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.nd4j.common.tests.tags.NativeTag;
import org.nd4j.common.tests.tags.TagNames;
import org.nd4j.linalg.api.buffer.DataType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.shade.netty.util.concurrent.DefaultThreadFactory;

import javax.annotation.concurrent.NotThreadSafe;

import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.*;

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
    @Disabled("Inconsistent across machines.")
    public void testMultiThreadedExecution() throws Exception {
        ExecutorService executorService = Executors.newFixedThreadPool(Runtime.getRuntime().availableProcessors(),new DefaultThreadFactory("test-thread"));
        List<Callable<Integer>> tasks = new ArrayList<>();
        int count = 10;
        for(int i = 0; i < count; i++) {
            tasks.add(() -> {
                PythonInterpreter initializingPythonInterpreter =  InitializingPythonInterpreter.getInstance();
                testBasicExecution(initializingPythonInterpreter);
                return 1;
            });
        }

        List<Future<Integer>> futures = executorService.invokeAll(tasks);
        int done = 0;
        for(Future<Integer> f : futures) {
            done += f.get(20,TimeUnit.MILLISECONDS);
        }

        assertEquals(count,done);

    }

    private void testBasicExecution(PythonInterpreter interpreter) {
        interpreter.gilLock().lock();
        testAddInt(interpreter);
        testAddDouble(interpreter);
        testNumpyAdd(interpreter);
        interpreter.gilLock().unlock();
        assertEquals(Nd4j.createFromArray(new long[]{3,3}), interpreter.getCachedJava("a_arr"));
        assertEquals(interpreter.getCachedJava("a_double"),4.0);
        assertEquals(interpreter.getCachedJava("a_int"),(long) 4);

    }


    private void testNumpyAdd(PythonInterpreter pythonInterpreter) {
        pythonInterpreter.set("a_arr", Nd4j.ones(2).castTo(DataType.INT64));
        pythonInterpreter.exec("a_arr += 2");
        assertEquals(Nd4j.createFromArray(new long[]{3,3}), pythonInterpreter.get("a_arr", false));
    }

    private void testAddDouble(PythonInterpreter interpreter) {
        interpreter.set("a_double",2.0);
        interpreter.exec("a_double += 2");
        assertEquals(interpreter.get("a_double", false),4.0);
    }
    private void testAddInt(PythonInterpreter interpreter) {
        interpreter.set("a_int",2);
        interpreter.exec("a_int += 2");
        assertEquals(interpreter.get("a_int", false),(long) 4);
    }



}
