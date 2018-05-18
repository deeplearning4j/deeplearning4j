package org.nd4j.linalg.cpu.nativecpu;

import lombok.NonNull;
import lombok.val;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.LongPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.nd4j.linalg.api.buffer.LongBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.cache.TadDescriptor;
import org.nd4j.linalg.primitives.Pair;
import org.nd4j.nativeblas.LongPointerWrapper;
import org.nd4j.nativeblas.NativeOps;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.concurrent.atomic.AtomicInteger;
import java.util.concurrent.atomic.AtomicLong;

/**
 * @author raver119@gmail.com
 */
public class CpuTADManager implements TADManager {
    private Map<TadDescriptor, Pair<DataBuffer, DataBuffer>> cache = new ConcurrentHashMap<>();
    private NativeOps nativeOps;
    private ConstantHandler constantHandler;
    private AtomicLong bytes = new AtomicLong(0);
    private AtomicInteger counter = new AtomicInteger(0);
    private static final int MAX_ENTRIES = 100;

    public CpuTADManager() {
        //
    }

    public void init(@NonNull NativeOps nativeOps, @NonNull ConstantHandler constantHandler) {
        this.nativeOps = nativeOps;
        this.constantHandler = constantHandler;
    }

    /**
     * This method removes all cached shape buffers
     */
    @Override
    public void purgeBuffers() {
        cache = new ConcurrentHashMap<>();
    }

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        if (dimension != null && dimension.length > 1)
            Arrays.sort(dimension);

        if (dimension == null || dimension.length >= 1 && dimension[0] == Integer.MAX_VALUE) {
            return new Pair<>(array.shapeInfoDataBuffer(), null);
        } else {
            TadDescriptor descriptor = new TadDescriptor(array, dimension);

            if (!cache.containsKey(descriptor)) {
                int dimensionLength = dimension.length;

                // FIXME: this is fast triage, remove it later
                int targetRank = array.rank(); //dimensionLength <= 1 ? 2 : dimensionLength;
                long offsetLength;
                long tadLength = 1;
                for (int i = 0; i < dimensionLength; i++) {
                    tadLength *= array.shape()[dimension[i]];
                }

                offsetLength = array.lengthLong() / tadLength;

                DataBuffer outputBuffer = new LongBuffer(targetRank * 2 + 4);
                DataBuffer offsetsBuffer = new LongBuffer(offsetLength);

                DataBuffer dimensionBuffer = constantHandler.getConstantBuffer(dimension);
                Pointer dimensionPointer = dimensionBuffer.addressPointer();

                Pointer xShapeInfo = array.shapeInfoDataBuffer().addressPointer();
                Pointer targetPointer = outputBuffer.addressPointer();
                Pointer offsetsPointer = offsetsBuffer.addressPointer();

                nativeOps.tadOnlyShapeInfo((LongPointer) xShapeInfo, (IntPointer) dimensionPointer, dimension.length,
                                (LongPointer) targetPointer, new LongPointerWrapper(offsetsPointer));


                // If the line below will be uncommented, shapes from JVM will be used on native side
                //outputBuffer = array.tensorAlongDimension(0, dimension).shapeInfoDataBuffer();
                Pair<DataBuffer, DataBuffer> pair = new Pair<>(outputBuffer, offsetsBuffer);
                if (counter.get() < MAX_ENTRIES) {
                    counter.incrementAndGet();
                    cache.put(descriptor, pair);

                    bytes.addAndGet((outputBuffer.length() * 4) + (offsetsBuffer.length() * 8));
                }
                return pair;
            }

            return cache.get(descriptor);
        }
    }

    @Override
    public long getCachedBytes() {
        return bytes.get();
    }
}
