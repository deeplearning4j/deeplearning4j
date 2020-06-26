/*******************************************************************************
 * Copyright (c) 2020 Konduit K.K.
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

package org.nd4j.python4j;


import org.bytedeco.cpython.PyThreadState;

import java.util.concurrent.atomic.AtomicBoolean;

import static org.bytedeco.cpython.global.python.*;


public class PythonGIL implements AutoCloseable {
    private static PyThreadState mainThreadState;
    private static final AtomicBoolean acquired = new AtomicBoolean();
    private boolean acquiredByMe = false;
    private static long defaultThreadId = -1;

    public static void assertThreadSafe() {
        if (acquired.get()) {
            return;
        }
        if (defaultThreadId == -1) {
            defaultThreadId = Thread.currentThread().getId();
        } else if (defaultThreadId != Thread.currentThread().getId()) {
            throw new RuntimeException("Attempt to use Python4j from multiple threads without " +
                    "acquiring GIL. Enclose your code in a try(PythonGIL gil = PythonGIL.lock()){...}" +
                    " block to ensure that GIL is acquired in multi-threaded environments.");
        }


    }

    static {
        new PythonExecutioner();
    }

    private PythonGIL() {
        while (acquired.get()) {
            try {
                Thread.sleep(10);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        }
        acquire();
        acquired.set(true);
        acquiredByMe = true;

    }

    @Override
    public void close() {
        if (acquiredByMe) {
            release();
            acquired.set(false);
            acquiredByMe = false;
        }

    }

    public static synchronized PythonGIL lock() {
        return new PythonGIL();
    }

    private static synchronized void acquire() {
        mainThreadState = PyEval_SaveThread();
        PyThreadState ts = PyThreadState_New(mainThreadState.interp());
        PyEval_RestoreThread(ts);
        PyThreadState_Swap(ts);
    }

    private static void release() { // do not synchronize!
        PyEval_SaveThread();
        PyEval_RestoreThread(mainThreadState);
    }

    public static boolean locked(){
        return acquired.get();
    }
}
