package org.nd4j.linalg.jcublas.compression;

import lombok.Getter;
import lombok.Setter;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.math3.util.FastMath;
import org.bytedeco.javacpp.FloatPointer;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.bytedeco.javacpp.PointerPointer;
import org.nd4j.compression.impl.AbstractCompressor;
import org.nd4j.jita.allocator.impl.AtomicAllocator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.exception.ND4JIllegalStateException;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.AbsValueGreaterThan;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.ops.transforms.Transforms;
import org.nd4j.nativeblas.NativeOps;
import org.nd4j.nativeblas.NativeOpsHolder;

import java.util.Arrays;

/**
 * This compression is very special case, and shouldn't be ever used outside of ParallelWrapper/ParameterServer implementation.
 * It encodes data as delta between zero and abs threshold.
 *
 * PLEASE NOTE: DO NOT USE THIS COMPRESSOR UNLESS YOU'RE 100% SURE WHAT YOU DO!
 *
 * @author raver119@gmail.com
 */
@Slf4j
public class CudaThreshold extends AbstractCompressor {
    @Getter @Setter protected float threshold = 1e-3f;

    /**
     * This method returns compression descriptor. It should be unique for any compressor implementation
     *
     * @return
     */
    @Override
    public String getDescriptor() {
        return "THRESHOLD";
    }

    /**
     * This method allows you to configure threshold for delta extraction. Pass it as float/double value
     *
     * Default value: 1e-3
     * @param vars
     */
    @Override
    public void configure(Object... vars) {
        if (vars[0] instanceof Number) {
            Number t = (Number) vars[0];
            threshold = FastMath.abs(t.floatValue());
            log.info("Setting threshold to [{}]", threshold);
        } else {
            throw new ND4JIllegalStateException("Threshold value should be Number");
        }
    }

    @Override
    public INDArray compress(INDArray array) {
        //logger.info("Threshold [{}] compression", threshold);

        Nd4j.getExecutioner().commit();
        Nd4j.getAffinityManager().ensureLocation(array, AffinityManager.Location.HOST);

        DataBuffer buffer = compress(array.data());
        if (buffer == null)
            return null;

        INDArray dup = Nd4j.createArrayFromShapeBuffer(buffer, array.shapeInfoDataBuffer());
        dup.markAsCompressed(true);

        return dup;
    }

    @Override
    public CompressionType getCompressionType() {
        return CompressionType.LOSSLESS;
    }

    @Override
    public DataBuffer decompress(DataBuffer buffer) {
        if (buffer.dataType() != DataBuffer.Type.INT)
            throw new UnsupportedOperationException();

        long compressedLength = buffer.getInt(0);
        long originalLength = buffer.getInt(1);

        DataBuffer result = Nd4j.createBuffer(originalLength);

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        PointerPointer extras = new PointerPointer(32).put(1, context.getOldStream());

        log.info("Source: {}", Arrays.toString(buffer.asInt()));

        NativeOpsHolder.getInstance().getDeviceNativeOps().decodeThresholdFloat(extras, AtomicAllocator.getInstance().getPointer(buffer), compressedLength, (FloatPointer) AtomicAllocator.getInstance().getPointer(result));
        AtomicAllocator.getInstance().getAllocationPoint(result).tickDeviceWrite();

        //DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(DataBuffer.TypeEx.THRESHOLD, buffer, getGlobalTypeEx());

        return result;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {

        int numThreads = 1024;
        int numBlocks = (int) (buffer.length() / numThreads + (buffer.length() % numThreads == 0 ? 0 : 1));

        CudaContext context = (CudaContext) AtomicAllocator.getInstance().getDeviceContext().getContext();

        DataBuffer offsetsBuffer = Nd4j.getDataBufferFactory().createInt(numBlocks+1);
        PointerPointer extras = new PointerPointer(32).put(1, context.getOldStream());


        NativeOpsHolder.getInstance().getDeviceNativeOps().encodeThresholdP1Float(extras, (FloatPointer) AtomicAllocator.getInstance().getPointer(buffer), buffer.length(), (IntPointer) AtomicAllocator.getInstance().getPointer(offsetsBuffer), threshold);
        AtomicAllocator.getInstance().getAllocationPoint(offsetsBuffer).tickDeviceWrite();

        int numMatches = offsetsBuffer.getInt(0);
        log.info("Totals: {}", numMatches);

        DataBuffer encodedBuffer = Nd4j.getDataBufferFactory().createInt(3+numMatches);
        encodedBuffer.put(0, numMatches);
        encodedBuffer.put(1, (int) buffer.length());
        encodedBuffer.put(2, Float.floatToIntBits(threshold));
        AtomicAllocator.getInstance().getAllocationPoint(encodedBuffer).tickHostWrite();

        // FIXME: make it parallel via some kernel, because it can be pretty big array here, i.e. for 150m original array, offsets can
        int prevSum = 0;
        for (int e = 0; e < numBlocks; e++) {
            int prevVal = offsetsBuffer.getInt(e + 1);
            offsetsBuffer.put(e + 1, prevSum);
            prevSum += prevVal;
        }

        AtomicAllocator.getInstance().getAllocationPoint(offsetsBuffer).tickHostWrite();

        log.info("Offsets: {}", Arrays.toString(offsetsBuffer.asInt()));
        log.info("Target: {}", Arrays.toString(encodedBuffer.asInt()));



        NativeOpsHolder.getInstance().getDeviceNativeOps().encodeThresholdP2Float(extras, (FloatPointer) AtomicAllocator.getInstance().getPointer(buffer), (IntPointer) AtomicAllocator.getInstance().getPointer(offsetsBuffer), buffer.length(), (IntPointer) AtomicAllocator.getInstance().getPointer(encodedBuffer));
        AtomicAllocator.getInstance().getAllocationPoint(encodedBuffer).tickDeviceWrite();
        AtomicAllocator.getInstance().getAllocationPoint(buffer).tickDeviceWrite();

        log.info("Encoded: {}", Arrays.toString(encodedBuffer.asInt()));

        return encodedBuffer;

        /*
        INDArray temp = Nd4j.createArrayFromShapeBuffer(buffer, Nd4j.getShapeInfoProvider().createShapeInformation(new int[]{1, (int) buffer.length()}));
        MatchCondition condition = new MatchCondition(temp, Conditions.absGreaterThanOrEqual(threshold));
        int cntAbs = Nd4j.getExecutioner().exec(condition, Integer.MAX_VALUE).getInt(0);


        //log.info("density ratio: {}", String.format("%.2f", cntAbs * 100.0f / buffer.length()));

        if (cntAbs == 0)
            return null;

        long originalLength = buffer.length() * Nd4j.sizeOfDataType(buffer.dataType());
        int compressedLength = cntAbs + 3;
        // first 3 elements contain header
        IntPointer pointer = new IntPointer(compressedLength);
        pointer.put(0, cntAbs);
        pointer.put(1, (int) buffer.length());
        pointer.put(2, Float.floatToIntBits(threshold));

        CompressionDescriptor descriptor = new CompressionDescriptor();
        descriptor.setCompressedLength(compressedLength * 4); // sizeOf(INT)
        descriptor.setOriginalLength(originalLength);
        descriptor.setOriginalElementSize(Nd4j.sizeOfDataType(buffer.dataType()));
        descriptor.setNumberOfElements(buffer.length());

        descriptor.setCompressionAlgorithm(getDescriptor());
        descriptor.setCompressionType(getCompressionType());



        CompressedDataBuffer cbuff = new CompressedDataBuffer(pointer, descriptor);

        Nd4j.getNDArrayFactory().convertDataEx(getBufferTypeEx(buffer), buffer.addressPointer(), DataBuffer.TypeEx.THRESHOLD, pointer, buffer.length());

        Nd4j.getAffinityManager().tagLocation(buffer, AffinityManager.Location.HOST);

        return cbuff;
        */
    }

    @Override
    protected CompressedDataBuffer compressPointer(DataBuffer.TypeEx srcType, Pointer srcPointer, int length, int elementSize) {
        throw new UnsupportedOperationException();
    }
}
