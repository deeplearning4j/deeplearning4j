package org.nd4j.linalg.jcublas.rng.distribution;


import jcuda.Pointer;
import jcuda.Sizeof;
import jcuda.driver.CUfunction;
import jcuda.jcublas.JCublas;
import jcuda.jcurand.JCurand;
import jcuda.runtime.JCuda;
import jcuda.runtime.cudaMemcpyKind;
import jcuda.utils.KernelLauncher;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.rng.distribution.Distribution;
import org.nd4j.linalg.jcublas.buffer.CudaDoubleDataBuffer;
import org.nd4j.linalg.jcublas.buffer.CudaFloatDataBuffer;
import org.nd4j.linalg.jcublas.buffer.JCudaBuffer;
import org.nd4j.linalg.jcublas.kernel.KernelFunctionLoader;
import org.nd4j.linalg.jcublas.kernel.KernelFunctions;
import org.nd4j.linalg.jcublas.rng.JcudaRandom;
import org.nd4j.linalg.jcublas.util.PointerUtil;

/**
 * Base JCuda distribution
 * Mainly handles calls to cuda
 *
 * @author Adam Gibson
 */
public abstract class BaseJCudaDistribution implements Distribution {


    protected JcudaRandom random;


    public BaseJCudaDistribution(JcudaRandom random) {
        this.random = random;
    }

    @Override
    public void reseedRandomGenerator(long seed) {
        random.setSeed(seed);
    }


    protected void doBinomial(INDArray p, Pointer out, int n, int len) {
        String functionName = "binomial";
        int blocks = PointerUtil.getNumBlocks(len, 128, 64);
        int threads = PointerUtil.getNumThreads(len, 64);
        JCudaBuffer randomNumbers = new CudaFloatDataBuffer(len * n);
        JCudaBuffer probBuffer = (JCudaBuffer) p.data();
        KernelLauncher.syncContext();

        Object[] kernelParams = new Object[]{
                Pointer.to(new int[]{len})
                , Pointer.to(new int[]{n})
                , Pointer.to(probBuffer.pointer())
                , Pointer.to(randomNumbers.pointer())
                , Pointer.to(out)
                , random.generator()

        };

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                functionName,
                "float"
                , kernelParams);
        //we don't need this buffer anymore this was purely for storing the output
        randomNumbers.destroy();
    }


    protected void doBinomialDouble(INDArray p, Pointer out, int n, int len) {
        String functionName = "binomial";
        int blocks = PointerUtil.getNumBlocks(len, 128, 64);
        int threads = PointerUtil.getNumThreads(len, 64);
        JCudaBuffer randomNumbers = new CudaDoubleDataBuffer(len);
        JCudaBuffer probBuffer = (JCudaBuffer) p.data();
        KernelLauncher.syncContext();

        Object[] kernelParams = new Object[]{
                Pointer.to(new int[]{len})
                , Pointer.to(new int[]{n})
                , Pointer.to(probBuffer.pointer()),
                Pointer.to(randomNumbers.pointer())
                , Pointer.to(out)
                , random.generator()

        };


        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                functionName,"double"
                , kernelParams);
        //we don't need this buffer anymore this was purely for storing the output
        randomNumbers.destroy();

    }

    protected void doBinomial(float p, Pointer out, int n, int len) {
        String functionName = "binomial_scalar";
        int blocks = PointerUtil.getNumBlocks(len, KernelFunctions.BLOCKS, KernelFunctions.THREADS);
        int threads = PointerUtil.getNumThreads(len, KernelFunctions.THREADS);
        JCudaBuffer randomNumbers = new CudaFloatDataBuffer(len * n);
        KernelLauncher.syncContext();

        Object[] kernelParams = new Object[]{
                Pointer.to(new int[]{len})
                , Pointer.to(new int[]{n})
                , Pointer.to(new float[]{p}),
                Pointer.to(randomNumbers.pointer())
                , Pointer.to(out)
                , random.generator()

        };

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                functionName,"float"
                , kernelParams);
        //we don't need this buffer anymore this was purely for storing the output
        randomNumbers.destroy();
    }


    protected void doBinomialDouble(double p, Pointer out, int n, int len) {
        String functionName = "binomial_scalar";
        int blocks = PointerUtil.getNumBlocks(len, KernelFunctions.BLOCKS, KernelFunctions.THREADS);
        int threads = PointerUtil.getNumThreads(len, KernelFunctions.THREADS);
        JCudaBuffer randomNumbers = new CudaDoubleDataBuffer(len);
        KernelLauncher.syncContext();

        Object[] kernelParams = new Object[]{
                Pointer.to(new int[]{len})
                , Pointer.to(new int[]{n})
                , Pointer.to(new double[]{p}),
                Pointer.to(randomNumbers.pointer())
                , Pointer.to(out)
                , Pointer.to(random.generator())

        };

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                functionName,"double"
                , kernelParams);
        //we don't need this buffer anymore this was purely for storing the output
        randomNumbers.destroy();

    }

    @Override
    public double sample() {
        return inverseCumulativeProbability(random.nextDouble());
    }


    protected void doSampleUniformDouble(Pointer out, double min, double max, int n) {
        KernelLauncher.syncContext();

        JCurand.curandGenerateUniformDouble(random.generator(), out, n);
        String functionName = "uniform";
        int blocks = PointerUtil.getNumBlocks(n, 128, 64);
        int threads = PointerUtil.getNumThreads(n, 64);

        Object[] kernelParams = new Object[]{
                Pointer.to(new int[]{n})
                , Pointer.to(new double[]{min})
                , Pointer.to(new double[]{max})
                , Pointer.to(out)
                , Pointer.to(random.generator())

        };
        KernelLauncher.syncContext();

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                functionName,"double"
                , kernelParams);
    }

    protected void doSampleNormal(Pointer out, INDArray means, float std) {
        float[] means2 = means.data().asFloat();
        for (int i = 0; i < means.length(); i++) {
            JCudaBuffer dummy = KernelFunctions.alloc(new float[2]);

            JCurand.curandGenerateNormal(
                    random.generator()
                    , dummy.pointer()
                    , 2
                    , means2[i]
                    , std);
            JCuda.cudaMemcpy(
                    out.withByteOffset(Sizeof.FLOAT * i)
                    , dummy.pointer()
                    , Sizeof.FLOAT
                    , cudaMemcpyKind.cudaMemcpyDeviceToDevice);
            dummy.destroy();

        }


    }

    protected void doSampleNormalDouble(Pointer out, INDArray means, double std) {
        double[] means2 = means.data().asDouble();
        for (int i = 0; i < means.length(); i++) {
            JCudaBuffer dummy = KernelFunctions.alloc(new double[2]);

            JCurand.curandGenerateNormalDouble(
                    random.generator()
                    , dummy.pointer()
                    , 2
                    , means2[i]
                    , std);
            JCuda.cudaMemcpy(
                    out.withByteOffset(Sizeof.DOUBLE * i)
                    , dummy.pointer(), Sizeof.DOUBLE
                    , cudaMemcpyKind.cudaMemcpyDeviceToDevice);
            dummy.destroy();
        }

    }

    protected void doSampleUniform(Pointer out, float min, float max, int n) {
        JCurand.curandGenerateUniform(random.generator(), out, n);
        String functionName = "uniform";
        int blocks = PointerUtil.getNumBlocks(n, KernelFunctions.BLOCKS, KernelFunctions.THREADS);
        int threads = PointerUtil.getNumThreads(n, KernelFunctions.THREADS);
        Object[] kernelParams = new Object[]{
                Pointer.to(new int[]{n})
                , Pointer.to(new float[]{min})
                , Pointer.to(new float[]{max})
                , Pointer.to(out)
                , Pointer.to(random.generator())

        };

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                functionName,"float"
                , kernelParams);

    }

    protected void doSampleNormal(float mean, float std, Pointer out, int n) {
        JCurand.curandGenerateNormal(random.generator(), out, n, mean, std);
    }

    protected void doSampleNormal(double mean, double std, Pointer out, int n) {
        JCurand.curandGenerateNormalDouble(random.generator(), out, n, mean, std);
    }


    public abstract double probability(double x0,
                                       double x1)
            throws NumberIsTooLargeException;
}
