package org.nd4j.linalg.cpu.nativecpu;

import lombok.NonNull;
import org.apache.commons.math3.util.Pair;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.ConstantHandler;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.cache.TadDescriptor;
import org.nd4j.nativeblas.NativeOps;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author raver119@gmail.com
 */
public class CpuTADManager implements TADManager {
    private Map<TadDescriptor, Pair<DataBuffer, DataBuffer>> cache = new ConcurrentHashMap<>();
    private NativeOps nativeOps;
    private ConstantHandler constantHandler;
    private static Logger logger = LoggerFactory.getLogger(CpuTADManager.class);

    public CpuTADManager() {
        //
    }

    public void init(@NonNull NativeOps nativeOps, @NonNull ConstantHandler constantHandler) {
        this.nativeOps = nativeOps;
        this.constantHandler = constantHandler;
    }

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        if (dimension == null || dimension[0] == Integer.MAX_VALUE) {
            return new Pair<DataBuffer, DataBuffer>(array.shapeInfoDataBuffer(), null);
        } else {
            TadDescriptor descriptor = new TadDescriptor(array, dimension);

            if (!cache.containsKey(descriptor)) {
                int dimensionLength = dimension.length;

                int targetRank = array.rank(); ///Math.max(array.rank() - dimensionLength, 2);
                int offsetLength = 0;
                int tadLength = 1;
                for (int i = 0; i < dimensionLength; i++) {
                    tadLength *= array.shape()[dimension[i]];
                }

                offsetLength = array.length() / tadLength;

                DataBuffer outputBuffer = new IntBuffer(targetRank * 2 + 4);
                DataBuffer offsetsBuffer = new IntBuffer(offsetLength);

                DataBuffer dimensionBuffer = constantHandler.getConstantBuffer(dimension);
                Pointer dimensionPointer = dimensionBuffer.addressPointer();

                Pointer xShapeInfo = array.shapeInfoDataBuffer().addressPointer();
                Pointer targetPointer = outputBuffer.addressPointer();
                Pointer offsetsPointer = offsetsBuffer.addressPointer();

                nativeOps.tadOnlyShapeInfo(xShapeInfo, dimensionPointer, dimension.length, targetPointer, offsetsPointer);

                Pair<DataBuffer, DataBuffer> pair = new Pair<DataBuffer, DataBuffer>(outputBuffer, offsetsBuffer);
                cache.put(descriptor, pair);
                return pair;
            }

            return cache.get(descriptor);
        }
    }
}
