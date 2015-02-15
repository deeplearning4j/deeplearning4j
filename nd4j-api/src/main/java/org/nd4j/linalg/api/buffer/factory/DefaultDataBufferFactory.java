package org.nd4j.linalg.api.buffer.factory;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.DoubleBuffer;
import org.nd4j.linalg.api.buffer.FloatBuffer;
import org.nd4j.linalg.api.buffer.IntBuffer;
import org.nd4j.linalg.util.ArrayUtil;

/**
 * Normal data buffer creation
 * @author Adam Gibson
 */
public class DefaultDataBufferFactory implements DataBufferFactory {
    @Override
    public DataBuffer createDouble(int length) {
        return new DoubleBuffer(length);
    }

    @Override
    public DataBuffer createFloat(int length) {
        return new FloatBuffer(length);
    }

    @Override
    public DataBuffer createInt(int length) {
        return new IntBuffer(length);
    }

    @Override
    public DataBuffer createDouble(int[] data) {
        return createDouble(data,true);
    }

    @Override
    public DataBuffer createFloat(int[] data) {
        return createFloat(data,true);
    }

    @Override
    public DataBuffer createInt(int[] data) {
        return createInt(data,true);
    }

    @Override
    public DataBuffer createDouble(double[] data) {
        return createDouble(data,true);
    }

    @Override
    public DataBuffer createFloat(double[] data) {
        return createFloat(data,true);
    }

    @Override
    public DataBuffer createInt(double[] data) {
        return createInt(data,true);
    }

    @Override
    public DataBuffer createDouble(float[] data) {
        return createDouble(data,true);
    }

    @Override
    public DataBuffer createFloat(float[] data) {
        return createFloat(data,true);
    }

    @Override
    public DataBuffer createInt(float[] data) {
        return createInt(data,true);
    }

    @Override
    public DataBuffer createDouble(int[] data, boolean copy) {
        return new DoubleBuffer(ArrayUtil.toDoubles(data),copy);
    }

    @Override
    public DataBuffer createFloat(int[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data),copy);
    }

    @Override
    public DataBuffer createInt(int[] data, boolean copy) {
        return new IntBuffer(data,copy);
    }

    @Override
    public DataBuffer createDouble(double[] data, boolean copy) {
        return new DoubleBuffer(data,copy);
    }

    @Override
    public DataBuffer createFloat(double[] data, boolean copy) {
        return new FloatBuffer(ArrayUtil.toFloats(data),copy);
    }

    @Override
    public DataBuffer createInt(double[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data),copy);
    }

    @Override
    public DataBuffer createDouble(float[] data, boolean copy) {
        return new FloatBuffer(data,copy);
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy) {
        return new FloatBuffer(data,copy);
    }

    @Override
    public DataBuffer createInt(float[] data, boolean copy) {
        return new IntBuffer(ArrayUtil.toInts(data),copy);
    }
}
