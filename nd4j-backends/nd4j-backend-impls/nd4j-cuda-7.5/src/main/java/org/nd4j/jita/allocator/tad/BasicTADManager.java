package org.nd4j.jita.allocator.tad;

import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.jita.conf.Configuration;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
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
    private static Logger logger = LoggerFactory.getLogger(Configuration.class);

    @Override
    public DataBuffer getTADOnlyShapeInfo(INDArray array, int[] dimension) {
        if (dimension == null || dimension[0] == Integer.MAX_VALUE) {
            return array.shapeInfoDataBuffer();
        } else {
            Arrays.sort(dimension);

            int dimensionLength = dimension.length;


            int targetRank = array.rank(); ///Math.max(array.rank() - dimensionLength, 2);


            DataBuffer outputBuffer = new CudaIntDataBuffer(targetRank * 2 + 4);

            long xShapeInfo = AddressRetriever.retrieveHostAddress(array.shapeInfoDataBuffer());
            long dimensionPointer = AddressRetriever.retrieveHostAddress(Nd4j.createBuffer(dimension));
            long targetPointer = AddressRetriever.retrieveHostAddress(outputBuffer);

            nativeOps.tadOnlyShapeInfo(xShapeInfo, dimensionPointer, dimensionLength, targetPointer);

            AtomicAllocator.getInstance().getAllocationPoint(outputBuffer).tickHostWrite();

            //logger.info("TAD shapeInfo after construction: {}", Arrays.toString(TadDescriptor.dataBufferToArray(outputBuffer)));
            // now we need to copy this buffer to either device global memory or device cache

            return outputBuffer;
        }
    }
}
