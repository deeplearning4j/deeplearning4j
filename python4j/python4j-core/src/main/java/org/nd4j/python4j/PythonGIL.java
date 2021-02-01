/*
 *  ******************************************************************************
 *  * Copyright (c) 2021 Deeplearning4j Contributors
 *  *
 *  * This program and the accompanying materials are made available under the
 *  * terms of the Apache License, Version 2.0 which is available at
 *  * https://www.apache.org/licenses/LICENSE-2.0.
 *  *
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


import lombok.extern.slf4j.Slf4j;
import org.bytedeco.cpython.PyThreadState;


import java.util.concurrent.atomic.AtomicBoolean;

import static org.bytedeco.cpython.global.python.*;

@Slf4j
public class PythonGIL implements AutoCloseable {
    private static final AtomicBoolean acquired = new AtomicBoolean();
    private boolean acquiredByMe = false;
    private static long defaultThreadId = -1;
    private int gilState;
    private static PyThreadState mainThreadState;
    private static long mainThreadId = -1;
    static {
        new PythonExecutioner();
    }

    /**
     * Set the main thread state
     * based on the current thread calling this method.
     * This method should not be called by the user.
     * It is already invoked automatically in {@link PythonExecutioner}
     */
    public static synchronized void setMainThreadState() {
        if(mainThreadId < 0 && mainThreadState != null) {
            mainThreadState = PyThreadState_Get();
            mainThreadId = Thread.currentThread().getId();
        }

    }

    /**
     * Asserts that the lock has been acquired.
     */
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

        if(!acquired.get()) {
            throw new IllegalStateException("Execution happening outside of GIL. Please use PythonExecutioner within a GIL block by wrapping it in a call via: try(PythonGIL gil = PythonGIL.lock()) { .. }");
        }
    }


    private PythonGIL() {
        while (acquired.get()) {
            try {
                log.debug("Blocking for GIL on thread " + Thread.currentThread().getId());
                Thread.sleep(100);
            } catch (Exception e) {
                throw new RuntimeException(e);
            }

        }

        log.debug("Acquiring GIL on " + Thread.currentThread().getId());
        acquired.set(true);
        acquiredByMe = true;
        acquire();

    }

    @Override
    public synchronized  void close() {
        if (acquiredByMe) {
            release();
            log.info("Releasing GIL on thread " + Thread.currentThread().getId());
            acquired.set(false);
            acquiredByMe = false;
        }
        else {
            log.info("Attempted to release GIL without having acquired GIL on thread " + Thread.currentThread().getId());
        }

    }


    /**
     * Lock the GIL for running python scripts.
     * This method should be used to create a new
     * {@link PythonGIL} object in the form of:
     * try(PythonGIL gil = PythonGIL.lock()) {
     *     //your python code here
     * }
     * @return the gil for this instance
     */
    public static  synchronized  PythonGIL  lock() {
        return new PythonGIL();
    }

    private  synchronized void acquire() {
        if(Thread.currentThread().getId() != mainThreadId) {
            log.info("Pre Gil State ensure for thread " + Thread.currentThread().getId());
            gilState = PyGILState_Ensure();
            log.info("Thread " + Thread.currentThread().getId() + " acquired GIL");
        } else {
            PyEval_RestoreThread(mainThreadState);
        }
    }

    private  void release() { // do not synchronize!
        if(Thread.currentThread().getId() != mainThreadId) {
            log.debug("Pre gil state release for thread " + Thread.currentThread().getId());
            PyGILState_Release(gilState);
        }
        else {
            PyEval_RestoreThread(mainThreadState);
        }
    }

    /**
     * Returns true if the GIL is currently in use.
     * This is typically true when {@link #lock()}
     * @return
     */
    public static boolean locked() {
        return acquired.get();
    }
}