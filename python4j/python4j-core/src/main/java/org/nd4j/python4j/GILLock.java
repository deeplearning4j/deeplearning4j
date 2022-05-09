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

import org.nd4j.shade.guava.util.concurrent.CycleDetectingLockFactory;

import java.util.concurrent.TimeUnit;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.locks.Condition;
import java.util.concurrent.locks.Lock;
import java.util.concurrent.locks.ReentrantLock;

/**
 * Python GIL holder based on:
 * https://github.com/invesdwin/invesdwin-context-python/blob/master/invesdwin-context-python-parent/invesdwin-context-python-runtime-python4j/src/main/java/de/invesdwin/context/python/runtime/python4j/internal/GilLock.java
 * Permission under apache license granted here: https://github.com/eclipse/deeplearning4j/issues/9595
 * @author Adam Gibson
 */
public class GILLock implements Lock {

    private static CycleDetectingLockFactory cycleDetectingLockFactory = CycleDetectingLockFactory.newInstance(CycleDetectingLockFactory.Policies.DISABLED);;

    private static final ReentrantLock reentrantLock = cycleDetectingLockFactory.newReentrantLock("python4j_lock");
    private PythonGIL pythonGIL;
    private final AtomicInteger lockedCount = new AtomicInteger();

    public void lock() {
        if(lockedCount.incrementAndGet() == 1) {
            reentrantLock.lock();
            pythonGIL = PythonGIL.lock();
        }

    }

    @Override
    public void lockInterruptibly() throws InterruptedException {
        lock();
    }

    @Override
    public boolean tryLock() {
        lock();
        return true;
    }

    @Override
    public boolean tryLock(long l, TimeUnit timeUnit) throws InterruptedException {
        return false;
    }

    public void unlock() {
        if(lockedCount.decrementAndGet() == 0) {
            if (pythonGIL != null)
                pythonGIL.close();
            pythonGIL = null;
            reentrantLock.unlock();
        }

    }

    @Override
    public Condition newCondition() {
        throw new UnsupportedOperationException();
    }


}
