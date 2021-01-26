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


package org.datavec.python;

import lombok.extern.slf4j.Slf4j;
import org.bytedeco.cpython.PyThreadState;

import static org.bytedeco.cpython.global.python.*;


@Slf4j
public class PythonGIL implements AutoCloseable {
    private static PyThreadState mainThreadState;

    static {
        log.debug("CPython: PyThreadState_Get()");
        mainThreadState = PyThreadState_Get();
    }

    private PythonGIL() {
        acquire();
    }

    @Override
    public void close() {
        release();
    }

    public static PythonGIL lock() {
        return new PythonGIL();
    }

    private static synchronized void acquire() {
        log.debug("acquireGIL()");
        log.debug("CPython: PyEval_SaveThread()");
        mainThreadState = PyEval_SaveThread();
        log.debug("CPython: PyThreadState_New()");
        PyThreadState ts = PyThreadState_New(mainThreadState.interp());
        log.debug("CPython: PyEval_RestoreThread()");
        PyEval_RestoreThread(ts);
        log.debug("CPython: PyThreadState_Swap()");
        PyThreadState_Swap(ts);
    }

    private static synchronized void release() {
        log.debug("CPython: PyEval_SaveThread()");
        PyEval_SaveThread();
        log.debug("CPython: PyEval_RestoreThread()");
        PyEval_RestoreThread(mainThreadState);
    }
}
