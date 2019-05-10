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

package org.nd4j.jita.allocator.concurrency;

import org.nd4j.jita.allocator.impl.AllocationShape;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * Lock implementation based on ReentrantReadWriteLock
 *
 * @author raver119@gmail.com
 */
public class RRWLock implements Lock {
    private ReentrantReadWriteLock globalLock = new ReentrantReadWriteLock();
    private ReentrantReadWriteLock externalsLock = new ReentrantReadWriteLock();

    private Map<Object, ReentrantReadWriteLock> objectLocks = new ConcurrentHashMap<>();


    /**
     * This method notifies locker, that specific object was added to tracking list
     *
     * @param object
     */
    @Override
    public void attachObject(Object object) {
        if (!objectLocks.containsKey(object))
            objectLocks.put(object, new ReentrantReadWriteLock());
    }

    /**
     * This method notifies locker that specific object was removed from tracking list
     *
     * @param object
     */
    @Override
    public void detachObject(Object object) {
        objectLocks.remove(object);
    }

    /**
     * This method acquires global-level read lock
     */
    @Override
    public void globalReadLock() {
        globalLock.readLock().lock();
    }

    /**
     * This method releases global-level read lock
     */
    @Override
    public void globalReadUnlock() {
        globalLock.readLock().unlock();
    }

    /**
     * This method acquires global-level write lock
     */
    @Override
    public void globalWriteLock() {
        globalLock.writeLock().lock();
    }

    /**
     * This method releases global-level write lock
     */
    @Override
    public void globalWriteUnlock() {
        globalLock.writeLock().unlock();
    }

    /**
     * This method acquires object-level read lock, and global-level read lock
     *
     * @param object
     */
    @Override
    public void objectReadLock(Object object) {
        //     globalReadLock();

        objectLocks.get(object).readLock().lock();
    }

    /**
     * This method releases object-level read lock, and global-level read lock
     *
     * @param object
     */
    @Override
    public void objectReadUnlock(Object object) {
        objectLocks.get(object).readLock().unlock();

        //     globalReadUnlock();
    }

    /**
     * This method acquires object-level write lock, and global-level read lock
     *
     * @param object
     */
    @Override
    public void objectWriteLock(Object object) {
        //     globalReadLock();

        objectLocks.get(object).writeLock().lock();
    }

    /**
     * This method releases object-level read lock, and global-level read lock
     *
     * @param object
     */
    @Override
    public void objectWriteUnlock(Object object) {
        objectLocks.get(object).writeLock().unlock();

        //      globalReadUnlock();
    }

    /**
     * This method acquires shape-level read lock, and read locks for object and global
     *
     * @param object
     * @param shape
     */
    @Override
    public void shapeReadLock(Object object, AllocationShape shape) {
        objectReadLock(object);
    }

    /**
     * This method releases shape-level read lock, and read locks for object and global
     *
     * @param object
     * @param shape
     */
    @Override
    public void shapeReadUnlock(Object object, AllocationShape shape) {

        objectReadUnlock(object);
    }

    /**
     * This method acquires shape-level write lock, and read locks for object and global
     *
     * @param object
     * @param shape
     */
    @Override
    public void shapeWriteLock(Object object, AllocationShape shape) {
        objectReadLock(object);
    }

    /**
     * This method releases shape-level write lock, and read locks for object and global
     *
     * @param object
     * @param shape
     */
    @Override
    public void shapeWriteUnlock(Object object, AllocationShape shape) {
        objectReadUnlock(object);
    }

    /**
     * This methods acquires read-lock on externals, and read-lock on global
     */
    @Override
    public void externalsReadLock() {
        externalsLock.readLock().lock();
    }

    /**
     * This methods releases read-lock on externals, and read-lock on global
     */
    @Override
    public void externalsReadUnlock() {
        externalsLock.readLock().unlock();
    }

    /**
     * This methods acquires write-lock on externals, and read-lock on global
     */
    @Override
    public void externalsWriteLock() {
        externalsLock.writeLock().lock();
    }

    /**
     * This methods releases write-lock on externals, and read-lock on global
     */
    @Override
    public void externalsWriteUnlock() {
        externalsLock.writeLock().unlock();
    }
}
