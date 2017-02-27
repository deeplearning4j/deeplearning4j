package org.nd4j.storage;

import lombok.NonNull;
import lombok.extern.slf4j.Slf4j;
import org.nd4j.compression.impl.NoOp;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.compression.AbstractStorage;
import org.nd4j.linalg.compression.NDArrayCompressor;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.locks.ReentrantReadWriteLock;

/**
 * AbstractStorage implementation, with Integer as key.
 * Primary goal is storage of individual rows/slices in system ram, even if working in GPU environment
 *
 * This implementation IS thread-safe, so it can be easily used together with ParallelWrapper
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CompressedRamStorage<T extends Object> implements AbstractStorage<T> {

    private NDArrayCompressor compressor = new NoOp();
    private Map<T, INDArray> compressedEntries = new ConcurrentHashMap<>();
    private boolean useInplaceCompression = false;
    private ReentrantReadWriteLock lock = new ReentrantReadWriteLock();
    private boolean emulateIsAbsent = false;

    private CompressedRamStorage() {
        //
    }

    /**
     * Store object into storage
     *
     * @param key
     * @param object
     */
    @Override
    public void store(T key, INDArray object) {
        INDArray toStore;
        if (useInplaceCompression) {
            compressor.compressi(object);
            toStore = object;
        } else {
            toStore = compressor.compress(object);
        }

        if (emulateIsAbsent)
            lock.writeLock().lock();

        compressedEntries.put(key, toStore);

        if (emulateIsAbsent)
            lock.writeLock().unlock();
    }

    /**
     * Store object into storage
     *
     * @param key
     * @param array
     */
    @Override
    public void store(T key, float[] array) {
        INDArray toStore = compressor.compress(array);

        if (emulateIsAbsent)
            lock.writeLock().lock();

        compressedEntries.put(key, toStore);

        if (emulateIsAbsent)
            lock.writeLock().unlock();
    }

    /**
     * Store object into storage
     *
     * @param key
     * @param array
     */
    @Override
    public void store(T key, double[] array) {
        INDArray toStore = compressor.compress(array);

        if (emulateIsAbsent)
            lock.writeLock().lock();

        compressedEntries.put(key, toStore);

        if (emulateIsAbsent)
            lock.writeLock().unlock();
    }

    /**
     * Store object into storage, if it doesn't exist
     *
     * @param key
     * @param object
     * @return Returns TRUE if store operation was applied, FALSE otherwise
     */
    @Override
    public boolean storeIfAbsent(T key, INDArray object) {
        try {
            if (emulateIsAbsent)
                lock.writeLock().lock();

            if (compressedEntries.containsKey(key)) {
                return false;
            } else {
                store(key, object);
                return true;
            }
        } finally {
            if (emulateIsAbsent)
                lock.writeLock().unlock();
        }
    }

    /**
     * Get object from the storage, by key
     *
     * @param key
     */
    @Override
    public INDArray get(T key) {
        try {
            if (emulateIsAbsent)
                lock.readLock().lock();

            if (containsKey(key)) {
                INDArray result = compressedEntries.get(key);

                // TODO: we don't save decompressed entries here, but something like LRU might be good idea
                return compressor.decompress(result);
            } else {
                return null;
            }
        } finally {
            if (emulateIsAbsent)
                lock.readLock().unlock();
        }
    }

    /**
     * This method checks, if storage contains specified key
     *
     * @param key
     * @return
     */
    @Override
    public boolean containsKey(T key) {
        try {
            if (emulateIsAbsent)
                lock.readLock().lock();

            return compressedEntries.containsKey(key);
        } finally {
            if (emulateIsAbsent)
                lock.readLock().unlock();
        }
    }

    /**
     * This method purges everything from storage
     */
    @Override
    public void clear() {
        if (emulateIsAbsent)
            lock.writeLock().lock();

        compressedEntries.clear();

        if (emulateIsAbsent)
            lock.writeLock().unlock();
    }

    /**
     * This method removes value by specified key
     *
     * @param key
     */
    @Override
    public void drop(T key) {
        if (emulateIsAbsent)
            lock.writeLock().lock();

        compressedEntries.remove(key);

        if (emulateIsAbsent)
            lock.writeLock().unlock();
    }

    /**
     * This method returns number of entries available in storage
     */
    @Override
    public long size() {
        try {
            if (emulateIsAbsent)
                lock.readLock().lock();

            return compressedEntries.size();
        } finally {
            if (emulateIsAbsent)
                lock.readLock().unlock();
        }
    }

    public static class Builder<T> {
        // we use NoOp as default compressor
        private NDArrayCompressor compressor = new NoOp();
        private boolean useInplaceCompression = false;
        private boolean emulateIsAbsent = false;

        public Builder() {

        }

        /**
         * This method defines, which compression algorithm will be used during storage
         * Default value: NoOp();
         *
         * @param compressor
         * @return
         */
        public Builder<T> setCompressor(@NonNull NDArrayCompressor compressor) {
            this.compressor = compressor;
            return this;
        }

        /**
         * If set to TRUE, all store/update calls will use inplace compression.
         * If set to FALSE, original array won't be modified, and copy will be used.
         *
         * Default value: FALSE;
         *
         * @param reallyUse
         * @return
         */
        public Builder<T> useInplaceCompression(boolean reallyUse) {
            this.useInplaceCompression = reallyUse;
            return this;
        }

        /**
         * If set to TRUE, all Read/Write locks will be used to emulate storeIfAbsent behaviour
         * If set to FALSE, concurrency will be provided by ConcurrentHashMap at Java7 level
         *
         * Default value: FALSE;
         *
         * @param reallyEmulate
         * @return
         */
        public Builder<T> emulateIsAbsent(boolean reallyEmulate) {
            this.emulateIsAbsent = reallyEmulate;
            return this;
        }


        public CompressedRamStorage<T> build() {
            CompressedRamStorage<T> storage = new CompressedRamStorage<>();
            storage.compressor = this.compressor;
            storage.useInplaceCompression = this.useInplaceCompression;
            storage.emulateIsAbsent = this.emulateIsAbsent;

            return storage;
        }
    }
}
