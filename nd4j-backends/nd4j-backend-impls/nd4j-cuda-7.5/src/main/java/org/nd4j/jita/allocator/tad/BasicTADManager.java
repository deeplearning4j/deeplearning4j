package org.nd4j.jita.allocator.tad;

import org.apache.commons.math3.util.Pair;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.cache.TADManager;
import org.nd4j.linalg.jcublas.buffer.AddressRetriever;
import org.nd4j.linalg.jcublas.buffer.CudaIntDataBuffer;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.Arrays;

/**
 * @author raver119@gmail.com
 */
public class BasicTADManager implements TADManager {
    protected NativeOps nativeOps = NativeOpsHolder.getInstance().getDeviceNativeOps();
    private static Logger logger = LoggerFactory.getLogger(BasicTADManager.class);

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        if (dimension == null || dimension.length == 0 || dimension[0] == Integer.MAX_VALUE) {
            return new Pair<DataBuffer, DataBuffer>(array.shapeInfoDataBuffer(), null);
        } else {
            Arrays.sort(dimension);

            int dimensionLength = dimension.length;


            int targetRank = array.rank(); ///Math.max(array.rank() - dimensionLength, 2);
            int offsetLength = 0;
            int tadLength = 1;
            for (int i = 0; i < dimensionLength; i++) {
                tadLength *= array.shape()[dimension[i]];
            }

            offsetLength = array.length() / tadLength;

       //     logger.info("Original shape info before TAD: {}", array.shapeInfoDataBuffer());
        //    logger.info("dimension: {}, tadLength: {}, offsetLength for TAD: {}", Arrays.toString(dimension),tadLength, offsetLength);

            DataBuffer outputBuffer = new CudaIntDataBuffer(targetRank * 2 + 4);
            DataBuffer offsetsBuffer = new CudaIntDataBuffer(offsetLength);

            DataBuffer dimensionBuffer = AtomicAllocator.getInstance().getConstantBuffer(dimension);
            Pointer dimensionPointer = AtomicAllocator.getInstance().getHostPointer(dimensionBuffer);

            Pointer xShapeInfo = AddressRetriever.retrieveHostPointer(array.shapeInfoDataBuffer());
            Pointer targetPointer = AddressRetriever.retrieveHostPointer(outputBuffer);
            Pointer offsetsPointer = AddressRetriever.retrieveHostPointer(offsetsBuffer);

            nativeOps.tadOnlyShapeInfo(xShapeInfo, dimensionPointer, dimensionLength, targetPointer, offsetsPointer);

            AtomicAllocator.getInstance().getAllocationPoint(outputBuffer).tickHostWrite();
            AtomicAllocator.getInstance().getAllocationPoint(offsetsBuffer).tickHostWrite();

        //   logger.info("TAD shapeInfo after construction: {}", Arrays.toString(TadDescriptor.dataBufferToArray(outputBuffer)));
            // now we need to copy this buffer to either device global memory or device cache

            return new Pair<DataBuffer, DataBuffer>(outputBuffer, offsetsBuffer);
        }
    }
}
