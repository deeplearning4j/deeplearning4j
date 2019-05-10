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

package org.nd4j.linalg.primitives;

import lombok.NoArgsConstructor;

import java.io.IOException;
import java.io.Serializable;
import java.util.Objects;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 *
 * @param <T>
 */
@NoArgsConstructor
public class Atomic<T extends Serializable> implements Serializable {
    private volatile T value;
    private transient ReentrantReadWriteLock lock = new ReentrantReadWriteLock();

    public Atomic(T initialValue) {
        this.value = initialValue;
    }

    /**
     * This method assigns new value
     * @param value
     */
    public void set(T value) {
        try {
            lock.writeLock().lock();

            this.value = value;
        } finally {
            lock.writeLock().unlock();
        }
    }

    /**
     * This method returns current value
     * @return
     */
    public T get() {
        try {
            lock.readLock().lock();

            return this.value;
        } finally {
            lock.readLock().unlock();
        }
    }



    /**
     * This method implements compare-and-swap
     *
     * @param expected
     * @param newValue
     * @return true if value was swapped, false otherwise
     */
    public boolean cas(T expected, T newValue) {
        try {
            lock.writeLock().lock();

            if (Objects.equals(value, expected)) {
                this.value = newValue;
                return true;
            } else
                return false;
        } finally {
            lock.writeLock().unlock();
        }
    }

    private void readObject(java.io.ObjectInputStream in) throws IOException, ClassNotFoundException {
        in.defaultReadObject();
        lock = new ReentrantReadWriteLock();
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;
        Atomic<?> atomic = (Atomic<?>) o;
        try {
            this.lock.readLock().lock();
            atomic.lock.readLock().lock();

            return Objects.equals(this.value, atomic.value);
        } catch (Exception e) {
            throw new RuntimeException(e);
        } finally {
            atomic.lock.readLock().unlock();
            this.lock.readLock().unlock();
        }
    }

    @Override
    public int hashCode() {
        try {
            this.lock.readLock().lock();

            return Objects.hash(value);
        } finally {
            this.lock.readLock().unlock();
        }
    }
}
