package org.nd4j.linalg.jcublas.buffer.factory;

import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.buffer.factory.DataBufferFactory;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;

/**
 * Created by agibsonccc on 2/14/15.
 */
public class CudaDataBufferFactory implements DataBufferFactory {
    @Override
    public DataBuffer createDouble(int length) {
        return null;
    }

    @Override
    public DataBuffer createFloat(int length) {
        return null;
    }

    @Override
    public DataBuffer createInt(int length) {
        return null;
    }

    @Override
    public DataBuffer createDouble(int[] data) {
        return null;
    }

    @Override
    public DataBuffer createFloat(int[] data) {
        return null;
    }

    @Override
    public DataBuffer createInt(int[] data) {
        return null;
    }

    @Override
    public DataBuffer createDouble(double[] data) {
        return new CudaDoubleDataBuffer(data);
    }

    @Override
    public DataBuffer createFloat(double[] data) {
        return null;
    }

    @Override
    public DataBuffer createInt(double[] data) {
        return null;
    }

    @Override
    public DataBuffer createDouble(float[] data) {
        return null;
    }

    @Override
    public DataBuffer createFloat(float[] data) {
        return null;
    }

    @Override
    public DataBuffer createInt(float[] data) {
        return null;
    }

    @Override
    public DataBuffer createDouble(int[] data, boolean copy) {
        return null;
    }

    @Override
    public DataBuffer createFloat(int[] data, boolean copy) {
        return null;
    }

    @Override
    public DataBuffer createInt(int[] data, boolean copy) {
        return null;
    }

    @Override
    public DataBuffer createDouble(double[] data, boolean copy) {
        return null;
    }

    @Override
    public DataBuffer createFloat(double[] data, boolean copy) {
        return null;
    }

    @Override
    public DataBuffer createInt(double[] data, boolean copy) {
        return null;
    }

    @Override
    public DataBuffer createDouble(float[] data, boolean copy) {
        return null;
    }

    @Override
    public DataBuffer createFloat(float[] data, boolean copy) {
        return null;
    }

    @Override
    public DataBuffer createInt(float[] data, boolean copy) {
        return null;
    }
}
