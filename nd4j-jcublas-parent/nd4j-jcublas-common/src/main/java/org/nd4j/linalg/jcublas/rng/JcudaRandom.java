package org.nd4j.linalg.jcublas.rng;

import jcuda.jcurand.JCurand;
import jcuda.jcurand.curandGenerator;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.SetRange;
import org.nd4j.linalg.api.rng.Random;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;

import static jcuda.jcurand.JCurand.*;
import static jcuda.jcurand.curandRngType.CURAND_RNG_PSEUDO_DEFAULT;

/**
 * Jcuda random number generator
 *
 * @author Adam Gibson
 */
public class JcudaRandom implements Random {
    private curandGenerator generator = new curandGenerator();

    /**
     * Initialize the random generator on the gpu
     */
    public JcudaRandom() {
        curandCreateGenerator(generator, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(generator, 1234);
        JCurand.setExceptionsEnabled(true);

    }

    public curandGenerator generator() {
        return generator;
    }


    @Override
    public void setSeed(int seed) {
        curandSetPseudoRandomGeneratorSeed(generator, seed);
    }

    @Override
    public void setSeed(int[] seed) {
    }

    @Override
    public void setSeed(long seed) {
        curandSetPseudoRandomGeneratorSeed(generator, seed);

    }

    @Override
    public void nextBytes(byte[] bytes) {
        throw new UnsupportedOperationException();
    }

    @Override
    public int nextInt() {
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerate(generator, buffer.pointer(), 2);
        double[] data = buffer.asDouble();
        int ret = (int) data[0];
        buffer.destroy();
        return ret;
    }

    @Override
    public int nextInt(int n) {
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerateUniformDouble(generator, buffer.pointer(), 2);
        double[] data = buffer.asDouble();
        int ret = (int) data[0];
        buffer.destroy();
        return ret;
    }

    @Override
    public long nextLong() {
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerate(generator, buffer.pointer(), 2);
        double[] data = buffer.asDouble();
        long ret = (long) data[0];
        buffer.destroy();
        return ret;
    }

    @Override
    public boolean nextBoolean() {
        return nextGaussian() > 0.5;
    }

    @Override
    public float nextFloat() {
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerate(generator, buffer.pointer(), 2);
        double[] data = buffer.asDouble();
        float ret = (float) data[0];
        buffer.destroy();
        return ret;
    }

    @Override
    public double nextDouble() {
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerate(generator, buffer.pointer(), 2);
        double[] data = buffer.asDouble();
        buffer.destroy();
        return data[0];
    }

    @Override
    public double nextGaussian() {
        JCudaBuffer buffer = new CudaDoubleDataBuffer(2);
        curandGenerateUniformDouble(generator, buffer.pointer(), 2);
        double[] data = buffer.asDouble();
        buffer.destroy();
        return data[0];
    }

    @Override
    public INDArray nextGaussian(int[] shape) {
        INDArray create = Nd4j.create(shape);
        JCudaBuffer buffer = (JCudaBuffer) create.data();
        if (buffer.dataType() == DataBuffer.FLOAT)
            curandGenerateUniform(generator, buffer.pointer(), create.length());
        else if (buffer.dataType() == DataBuffer.DOUBLE)
            curandGenerateUniformDouble(generator, buffer.pointer(), create.length());
        else
            throw new IllegalStateException("Illegal data type discovered");
        return create;
    }

    @Override
    public INDArray nextDouble(int[] shape) {
        INDArray create = Nd4j.create(shape);
        JCudaBuffer buffer = (JCudaBuffer) create.data();
        if (buffer.dataType() == DataBuffer.FLOAT)
            curandGenerateUniform(generator, buffer.pointer(), create.length());
        else if (buffer.dataType() == DataBuffer.DOUBLE)
            curandGenerateUniformDouble(generator, buffer.pointer(), create.length());
        else
            throw new IllegalStateException("Illegal data type discovered");
        return create;
    }

    @Override
    public INDArray nextFloat(int[] shape) {
        INDArray create = Nd4j.create(shape);
        JCudaBuffer buffer = (JCudaBuffer) create.data();
        if (buffer.dataType() == DataBuffer.FLOAT)
            curandGenerateUniform(generator, buffer.pointer(), create.length());
        else if (buffer.dataType() == DataBuffer.DOUBLE)
            curandGenerateUniformDouble(generator, buffer.pointer(), create.length());
        else
            throw new IllegalStateException("Illegal data type discovered");
        return create;
    }

    @Override
    public INDArray nextInt(int[] shape) {
        INDArray create = Nd4j.create(shape);
        JCudaBuffer buffer = (JCudaBuffer) create.data();
        if (buffer.dataType() == DataBuffer.FLOAT)
            curandGenerateUniform(generator, buffer.pointer(), create.length());
        else if (buffer.dataType() == DataBuffer.DOUBLE)
            curandGenerateUniformDouble(generator, buffer.pointer(), create.length());
        else
            throw new IllegalStateException("Illegal data type discovered");

        Nd4j.getExecutioner().exec(new SetRange(create, 0, 1));

        return create;
    }

    @Override
    public INDArray nextInt(int n, int[] shape) {
        INDArray create = Nd4j.create(shape);
        JCudaBuffer buffer = (JCudaBuffer) create.data();
        if (buffer.dataType() == DataBuffer.FLOAT)
            curandGenerateUniform(generator, buffer.pointer(), create.length());
        else if (buffer.dataType() == DataBuffer.DOUBLE)
            curandGenerateUniformDouble(generator, buffer.pointer(), create.length());
        else
            throw new IllegalStateException("Illegal data type discovered");

        Nd4j.getExecutioner().exec(new SetRange(create, 0, n));

        return create;
    }


}
