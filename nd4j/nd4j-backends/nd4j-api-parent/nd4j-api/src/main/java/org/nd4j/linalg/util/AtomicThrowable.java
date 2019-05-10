/*******************************************************************************
 * Copyright (c) 2015-2018 Skymind, Inc.
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

package org.nd4j.linalg.util;

import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * This class provides thread-safe holder for Throwable
 *
 * @author raver119@gmail.com
 */
public class AtomicThrowable {
    protected volatile Throwable t = null;
    protected ReentrantReadWriteLock lock;

    /**
     * This method creates new instance
     */
    public AtomicThrowable() {
        // unfortunately we want fair queue here, to avoid situations when we return null after exception
        lock = new ReentrantReadWriteLock(true);
    }

    /**
     * This method creates new instance with given initial state
     * @param e
     */
    public AtomicThrowable(Exception e) {
        this();
        t = e;
    }

    /**
     * This method returns current state
     * @return
     */
    public Throwable get() {
        try {
            lock.readLock().lock();

            return t;
        } finally {
            lock.readLock().unlock();
        }
    }

    /**
     * This method updates state with given Throwable
     * @param t
     */
    public void set(Throwable t) {
        try {
            lock.writeLock().lock();

            this.t = t;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * This method updates state only if it wasn't set before
     *
     * @param t
     */
    public void setIfFirst(Throwable t) {
        try {
            lock.writeLock().lock();

            if (this.t == null)
                this.t = t;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * This method returns TRUE if internal state holds error, FALSE otherwise
     *
     * @return
     */
    public boolean isTriggered() {
        try {
            lock.readLock().lock();

            return t != null;
        } finally {
            lock.readLock().unlock();
        }
    }
}
