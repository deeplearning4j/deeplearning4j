package org.nd4j.linalg.jcublas.rng.distribution;


import jcuda.Pointer;
import jcuda.driver.CUfunction;
import jcuda.jcublas.JCublas;
import jcuda.jcurand.JCurand;
import org.apache.commons.math3.exception.NumberIsTooLargeException;
import org.nd4j.linalg.api.buffer.DataBuffer;
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




    protected void doBinomial(INDArray p,Pointer out,int n,int len) {
        String functionName = "binomial";
        CUfunction func =  KernelFunctionLoader.getInstance().getFunction(functionName, "float");
        if(func == null)
            throw new IllegalArgumentException("Function " + functionName + " with data type float does not exist");
        int blocks = PointerUtil.getNumBlocks(len, 128, 64);
        int threads = PointerUtil.getNumThreads(len,64);
        JCudaBuffer randomNumbers = new CudaFloatDataBuffer(len * n);
        JCudaBuffer probBuffer = (JCudaBuffer) p.data();

        Pointer kernelParams = Pointer.to(
                Pointer.to(new int[]{len})
                ,Pointer.to(new int[]{n})
                ,Pointer.to(probBuffer.pointer())
                ,Pointer.to(randomNumbers.pointer())
                ,Pointer.to(out)
                ,random.generator()

        );

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                func
                , kernelParams);
        //we don't need this buffer anymore this was purely for storing the output
        randomNumbers.destroy();
    }


    protected void doBinomialDouble(INDArray p,Pointer out,int n,int len) {
        String functionName = "binomial";
        CUfunction func =  KernelFunctionLoader.getInstance().getFunction(functionName, "double");
        if(func == null)
            throw new IllegalArgumentException("Function " + functionName + " with data type double does not exist");
        int blocks = PointerUtil.getNumBlocks(len, 128, 64);
        int threads = PointerUtil.getNumThreads(len, 64);
        JCudaBuffer randomNumbers = new CudaDoubleDataBuffer(len);
        JCudaBuffer probBuffer = (JCudaBuffer) p.data();

        Pointer kernelParams = Pointer.to(
                Pointer.to(new int[]{len})
                ,Pointer.to(new int[]{n})
                ,Pointer.to(probBuffer.pointer()),
                Pointer.to(randomNumbers.pointer())
                ,Pointer.to(out)
                ,random.generator()

        );


        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                func
                , kernelParams);
        //we don't need this buffer anymore this was purely for storing the output
        randomNumbers.destroy();

    }

    protected void doBinomial(float p,Pointer out,int n,int len) {
        String functionName = "binomial_scalar";
        CUfunction func =  KernelFunctionLoader.getInstance().getFunction(functionName, "float");
        if(func == null)
            throw new IllegalArgumentException("Function " + functionName + " with data type float does not exist");
        int blocks = PointerUtil.getNumBlocks(len, 128, 64);
        int threads = PointerUtil.getNumThreads(len,64);
        JCudaBuffer randomNumbers = new CudaFloatDataBuffer(len * n);

        Pointer kernelParams = Pointer.to(
                Pointer.to(new int[]{len})
                ,Pointer.to(new int[]{n})
                ,Pointer.to(new float[]{p}),
                Pointer.to(randomNumbers.pointer())
                ,Pointer.to(out)
                ,random.generator()

        );

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                func
                , kernelParams);
        //we don't need this buffer anymore this was purely for storing the output
        randomNumbers.destroy();
    }


    protected void doBinomialDouble(double p,Pointer out,int n,int len) {
        String functionName = "binomial_scalar";
        CUfunction func =  KernelFunctionLoader.getInstance().getFunction(functionName, "double");
        if(func == null)
            throw new IllegalArgumentException("Function " + functionName + " with data type double does not exist");
        int blocks = PointerUtil.getNumBlocks(len, 128, 64);
        int threads = PointerUtil.getNumThreads(len,64);
        JCudaBuffer randomNumbers = new CudaDoubleDataBuffer(len);

        Pointer kernelParams = Pointer.to(
                Pointer.to(new int[]{len})
                ,Pointer.to(new int[]{n})
                ,Pointer.to(new double[]{p}),
                Pointer.to(randomNumbers.pointer())
                ,Pointer.to(out)
                ,Pointer.to(random.generator())

        );

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                func
                , kernelParams);
        //we don't need this buffer anymore this was purely for storing the output
        randomNumbers.destroy();

    }

    @Override
    public double sample() {
        return inverseCumulativeProbability(random.nextDouble());
    }




    protected void doSampleUniformDouble(Pointer out,double min,double max,int n) {
        JCurand.curandGenerateUniformDouble(random.generator(), out, n);
        String functionName = "uniform";
        CUfunction func =  KernelFunctionLoader.getInstance().getFunction(functionName, "double");
        if(func == null)
            throw new IllegalArgumentException("Function " + functionName + " with data type double does not exist");
        int blocks = PointerUtil.getNumBlocks(n, 128, 64);
        int threads = PointerUtil.getNumThreads(n,64);
        Pointer kernelParams = Pointer.to(
                Pointer.to(new int[]{n})
                ,Pointer.to(new double[]{min})
                ,Pointer.to(new double[]{max})
                ,Pointer.to(out)
                ,Pointer.to(random.generator())

        );

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                func
                , kernelParams);
    }

    protected void doSampleNormal(Pointer out,INDArray means,float std) {
        Pointer dummy = KernelFunctions.alloc(new float[2]);
        for(int i = 0; i < means.length(); i++) {
            JCurand.curandGenerateNormal(random.generator(),dummy,2,means.linearView().getFloat(i),std);
            JCublas.cublasScopy(
                    1
                    ,
                    dummy
                    ,1
                    ,out
                    ,means.majorStride());
        }

        JCublas.cublasFree(dummy);


    }

    protected void doSampleNormalDouble(Pointer out,INDArray means,double std) {
        Pointer dummy = KernelFunctions.alloc(new double[2]);
        for(int i = 0; i < means.length(); i++) {
            JCurand.curandGenerateNormalDouble(random.generator(), dummy, 2, means.linearView().getDouble(i), std);
            JCublas.cublasDcopy(
                    1
                    ,
                    dummy
                    , 1
                    , out
                    , means.majorStride());
        }

        JCublas.cublasFree(dummy);
    }

    protected void doSampleUniform(Pointer out,float min,float max,int n) {
        JCurand.curandGenerateUniform(random.generator(),out,n);
        String functionName = "uniform";
        CUfunction func =  KernelFunctionLoader.getInstance().getFunction(functionName, "double");
        if(func == null)
            throw new IllegalArgumentException("Function " + functionName + " with data type double does not exist");
        int blocks = PointerUtil.getNumBlocks(n, 128, 64);
        int threads = PointerUtil.getNumThreads(n,64);
        Pointer kernelParams = Pointer.to(
                Pointer.to(new int[]{n})
                ,Pointer.to(new double[]{min})
                ,Pointer.to(new double[]{max})
                ,Pointer.to(out)
                ,Pointer.to(random.generator())

        );

        //int len,int n,double *ps,double *result, curandState *s
        KernelFunctions.invoke(
                blocks,
                threads,
                func
                , kernelParams);

    }

    protected void doSampleNormal(float mean,float std, Pointer out, int n) {
        JCurand.curandGenerateNormal(random.generator(), out, n, mean, std);
    }

    protected void doSampleNormal(double mean,double std, Pointer out, int n) {
        JCurand.curandGenerateNormalDouble(random.generator(), out, n, mean, std);
    }


    public abstract double probability(double x0,
                                       double x1)
            throws NumberIsTooLargeException;
}
