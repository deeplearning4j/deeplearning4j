package org.nd4j.linalg.cpu.nativecpu.cache;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.cache.ArrayDescriptor;
import org.nd4j.linalg.cache.BasicConstantHandler;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.factory.Nd4j;

import java.util.HashMap;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.Semaphore;

/**
 * @author raver119@gmail.com
 */
public class ConstantBuffersCache extends BasicConstantHandler {
    protected Map<ArrayDescriptor, DataBuffer> buffersCache = new ConcurrentHashMap<>();

    @Override
    public DataBuffer getConstantBuffer(int[] array) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createBuffer(array);

            buffersCache.put(descriptor, buffer);
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
            DataBuffer buffer = Nd4j.createBuffer(array);

            buffersCache.put(descriptor, buffer);
            return buffer;
        }

        return buffersCache.get(descriptor);
    }

    @Override
    public DataBuffer getConstantBuffer(double[] array) {
        ArrayDescriptor descriptor = new ArrayDescriptor(array);

        if (!buffersCache.containsKey(descriptor)) {
            DataBuffer buffer = Nd4j.createBuffer(array);

            buffersCache.put(descriptor, buffer);
            return buffer;
        }

        return buffersCache.get(descriptor);
    }
}
