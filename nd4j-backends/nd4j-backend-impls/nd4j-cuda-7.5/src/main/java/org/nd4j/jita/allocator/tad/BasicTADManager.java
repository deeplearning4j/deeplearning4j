package org.nd4j.jita.allocator.tad;

import org.apache.commons.math3.util.Pair;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.shape.Shape;
import org.nd4j.linalg.factory.Nd4j;
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
    protected AtomicAllocator allocator = AtomicAllocator.getInstance();
    private static Logger logger = LoggerFactory.getLogger(BasicTADManager.class);

    @Override
    public Pair<DataBuffer, DataBuffer> getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        if (dimension == null || dimension[0] == Integer.MAX_VALUE) {
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

            //logger.info("tadLength: {}, offsetLength for TAD: {}", tadLength, offsetLength);

            DataBuffer outputBuffer = new CudaIntDataBuffer(targetRank * 2 + 4);
            DataBuffer offsetsBuffer = new CudaIntDataBuffer(offsetLength);

            long xShapeInfo = AddressRetriever.retrieveHostAddress(array.shapeInfoDataBuffer());
            long dimensionPointer = AddressRetriever.retrieveHostAddress(Nd4j.createBuffer(dimension));
            long targetPointer = AddressRetriever.retrieveHostAddress(outputBuffer);
            long offsetsPointer = AddressRetriever.retrieveHostAddress(offsetsBuffer);

            nativeOps.tadOnlyShapeInfo(xShapeInfo, dimensionPointer, dimensionLength, targetPointer, offsetsPointer);

            AtomicAllocator.getInstance().getAllocationPoint(outputBuffer).tickHostWrite();
            AtomicAllocator.getInstance().getAllocationPoint(offsetsBuffer).tickHostWrite();

            //logger.info("TAD shapeInfo after construction: {}", Arrays.toString(TadDescriptor.dataBufferToArray(outputBuffer)));
            // now we need to copy this buffer to either device global memory or device cache

            return new Pair<DataBuffer, DataBuffer>(outputBuffer, offsetsBuffer);
        }
    }
}
