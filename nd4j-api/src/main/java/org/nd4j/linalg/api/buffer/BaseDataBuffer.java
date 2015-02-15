package org.nd4j.linalg.api.buffer;

import org.nd4j.linalg.api.complex.IComplexDouble;
import org.nd4j.linalg.api.complex.IComplexFloat;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.complex.IComplexNumber;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.ops.ElementWiseOp;

import java.io.RandomAccessFile;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.List;

/**
 * Base class for a data buffer handling basic byte operations among other things.
 * @author Adam Gibson
 */
public abstract  class BaseDataBuffer implements DataBuffer {

    protected int length;
    //memory mapped file
    protected String path;
    protected RandomAccessFile memoryMappedBuffer;
    public static final int MAPPING_SIZE = 1 << 30;
    protected final List<ByteBuffer> mappings = new ArrayList<>();


    protected BaseDataBuffer(int length) {
        this.length = length;
    }

    @Override
    public   int length() {
        return length;
    }


    public static byte[] toByteArray(double value) {
        byte[] bytes = new byte[8];
        ByteBuffer.wrap(bytes).putDouble(value);
        return bytes;
    }

    public static byte[] toByteArray(float value) {
        byte[] bytes = new byte[4];
        ByteBuffer.wrap(bytes).putFloat(value);
        return bytes;
    }


    public static byte[] toByteArray(int value) {
        byte[] bytes = new byte[4];
        ByteBuffer.wrap(bytes).putFloat(value);
        return bytes;
    }

    public static double toDouble(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getDouble();
    }

    public static int toInt(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getInt();
    }

    public static float toFloat(byte[] bytes) {
        return ByteBuffer.wrap(bytes).getFloat();
    }

    @Override
    public <E> E getElement(int i) {
        throw new UnsupportedOperationException();

    }


    @Override
    public IComplexFloat getComplexFloat(int i) {
        return Nd4j.createFloat(getFloat(i),getFloat(i) + 1);
    }

    @Override
    public IComplexDouble getComplexDouble(int i) {
        return Nd4j.createDouble(getDouble(i),getDouble(i + 1));
    }

    @Override
    public IComplexNumber getComplex(int i) {
        return dataType() == DataBuffer.FLOAT ? getComplexFloat(i) : getComplexDouble(i);
    }

    @Override
    public void apply(ElementWiseOp op, int offset) {
        INDArray from = op.from();
        if(from instanceof IComplexNDArray) {
            for(int i = offset; i < length(); i++) {
                put(i,op.apply(from,getComplex(i),i));
            }
        }
        else {
            for(int i = offset; i < length(); i++) {
                put(i,op.apply(from,getDouble(i),i));
            }
        }

    }




    @Override
    public <E> void put(int i, E element) {
        throw new UnsupportedOperationException();
    }
    @Override
    public <E> E[] asType() {
        throw new UnsupportedOperationException();
    }
}
