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

import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Python GIL holder based on:
 * https://github.com/invesdwin/invesdwin-context-python/blob/master/invesdwin-context-python-parent/invesdwin-context-python-runtime-python4j/src/main/java/de/invesdwin/context/python/runtime/python4j/internal/GilLock.java
 * Permission under apache license granted here: https://github.com/eclipse/deeplearning4j/issues/9595
 * @author Adam Gibson
 */
public class GILLock {

    private ReentrantLock reentrantLock = new ReentrantLock();
    private PythonGIL pythonGIL;
    private AtomicInteger lockCount = new AtomicInteger(0);

    public void lock() {
        reentrantLock.lock();
        pythonGIL = PythonGIL.lock();

    }

    public void unlock() {
        if(pythonGIL != null)
            pythonGIL.close();
        pythonGIL = null;
        reentrantLock.unlock();

    }



}
