package org.nd4j.linalg.cpu.nativecpu.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.factory.Nd4j;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class ConstantBuffersCache extends BasicConstantHandler {
    protected Map<ArrayDescriptor, DataBuffer> buffersCache = new ConcurrentHashMap<>();
    private AtomicInteger counter = new AtomicInteger(0);
    private AtomicLong bytes = new AtomicLong(0);
    private static final int MAX_ENTRIES = 1000;

    @Override
    public DataBuffer getConstantBuffer(int[] array) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createBufferDetached(array);

            // we always allow int arrays with length < 3. 99.9% it's just dimension array. we don't want to recreate them over and over
            if (counter.get() < MAX_ENTRIES || array.length < 4) {
                counter.incrementAndGet();
                buffersCache.put(descriptor, buffer);
                bytes.addAndGet(array.length * 4);
            }
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    /**
     * This method removes all cached constants
     */
    @Override
    public void purgeConstants() {
        buffersCache = new ConcurrentHashMap<>();
    }

    @Override
    public DataBuffer getConstantBuffer(float[] array) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createBufferDetached(array);

            if (counter.get() < MAX_ENTRIES) {
                counter.incrementAndGet();
                buffersCache.put(descriptor, buffer);

                bytes.addAndGet(array.length * Nd4j.sizeOfDataType());
            }
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    @Override
    public DataBuffer getConstantBuffer(double[] array) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createBufferDetached(array);

            if (counter.get() < MAX_ENTRIES) {
                counter.incrementAndGet();
                buffersCache.put(descriptor, buffer);

                bytes.addAndGet(array.length * Nd4j.sizeOfDataType());
            }
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    @Override
    public DataBuffer getConstantBuffer(long[] array) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createBufferDetached(array);

            if (counter.get() < MAX_ENTRIES) {
                counter.incrementAndGet();
                buffersCache.put(descriptor, buffer);

                bytes.addAndGet(array.length * Nd4j.sizeOfDataType());
            }
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    @Override
    public long getCachedBytes() {
        return bytes.get();
    }
}
