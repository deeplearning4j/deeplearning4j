package org.nd4j.compression.impl;

import lombok.Getter;
import lombok.Setter;
import org.bytedeco.javacpp.IntPointer;
import org.bytedeco.javacpp.Pointer;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.concurrency.AffinityManager;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.accum.MatchCondition;
import org.nd4j.linalg.compression.CompressedDataBuffer;
import org.nd4j.linalg.compression.CompressionDescriptor;
import org.nd4j.linalg.compression.CompressionType;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.indexing.conditions.Conditions;
import org.nd4j.linalg.ops.transforms.Transforms;

/**
 * This compression is very special case, and shouldn't be ever used outside of ParallelWrapper/ParameterServer implementation.
 * It encodes data as delta between zero and abs threshold.
 *
 * PLEASE NOTE: DO NOT USE THIS COMPRESSOR UNLESS YOU'RE 100% SURE WHAT YOU DO!
 *
 * @author raver119@gmail.com
 */
public class Threshold extends AbstractCompressor {
    @Getter @Setter private float threshold = 1e-3f;

    /**
     * This method returns compression descriptor. It should be unique for any compressor implementation
     *
     * @return
     */
    @Override
    public String getDescriptor() {
        return "THRESHOLD";
    }

    @Override
    public INDArray compress(INDArray array) {
        //logger.info("Threshold [{}] compression", threshold);

        Nd4j.getAffinityManager().ensureLocation(array, AffinityManager.Location.HOST);

        DataBuffer buffer = compress(array.data());
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


        DataBuffer result = Nd4j.getNDArrayFactory().convertDataEx(DataBuffer.TypeEx.THRESHOLD, buffer, getGlobalTypeEx());

        return result;
    }

    @Override
    public DataBuffer compress(DataBuffer buffer) {
        INDArray temp = Nd4j.createArrayFromShapeBuffer(buffer, Nd4j.getShapeInfoProvider().createShapeInformation(new int[]{1, (int) buffer.length()}));
        MatchCondition condition = new MatchCondition(Transforms.abs(temp, true), Conditions.greaterThanOrEqual(threshold));
        int cntAbs = Nd4j.getExecutioner().exec(condition, Integer.MAX_VALUE).getInt(0);

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

        return cbuff;
    }

    @Override
    protected CompressedDataBuffer compressPointer(DataBuffer.TypeEx srcType, Pointer srcPointer, int length, int elementSize) {
        return null;
    }
}
