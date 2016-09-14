package org.nd4j.jita.allocator.concurrency;

import org.nd4j.jita.allocator.impl.AllocationShape;

/**
 * Collection of multilevel locks for JITA
 *
 * @author raver119@gmail.com
 */
public interface Lock {

    /**
     * This method notifies locker, that specific object was added to tracking list
     * @param object
     */
    void attachObject(Object object);

    /**
     * This method notifies locker that specific object was removed from tracking list
     * @param object
     */
    void detachObject(Object object);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////  Global-level locks

    /**
     * This method acquires global-level read lock
     */
    void globalReadLock();

    /**
     * This method releases global-level read lock
     */
    void globalReadUnlock();

    /**
     * This method acquires global-level write lock
     */
    void globalWriteLock();

    /**
     * This method releases global-level write lock
     */
    void globalWriteUnlock();

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////  Object-level locks

    /**
     * This method acquires object-level read lock, and global-level read lock
     * @param object
     */
    void objectReadLock(Object object);

    /**
     * This method releases object-level read lock, and global-level read lock
     * @param object
     */
    void objectReadUnlock(Object object);

    /**
     * This method acquires object-level write lock, and global-level read lock
     * @param object
     */
    void objectWriteLock(Object object);

    /**
     * This method releases object-level read lock, and global-level read lock
     * @param object
     */
    void objectWriteUnlock(Object object);

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
    ////  Shape-level locks

    /**
     * This method acquires shape-level read lock, and read locks for object and global
     * @param object
     * @param shape
     */
    void shapeReadLock(Object object, AllocationShape shape);

    /**
     * This method releases shape-level read lock, and read locks for object and global
     * @param object
     * @param shape
     */
    void shapeReadUnlock(Object object, AllocationShape shape);

    /**
     * This method acquires shape-level write lock, and read locks for object and global
     * @param object
     * @param shape
     */
    void shapeWriteLock(Object object, AllocationShape shape);

    /**
     * This method releases shape-level write lock, and read locks for object and global
     * @param object
     * @param shape
     */
    void shapeWriteUnlock(Object object, AllocationShape shape);

    /**
     * This methods acquires read-lock on externals, and read-lock on global
     */
    void externalsReadLock();

    /**
     * This methods releases read-lock on externals, and read-lock on global
     */
    void externalsReadUnlock();

    /**
     * This methods acquires write-lock on externals, and read-lock on global
     */
    void externalsWriteLock();

    /**
     * This methods releases write-lock on externals, and read-lock on global
     */
    void externalsWriteUnlock();
}
