package org.nd4j.linalg.jcublas.fft.ops;

import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.VectorFFT;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.context.CudaContext;
import org.nd4j.linalg.jcublas.fft.JcudaFft;
import org.nd4j.linalg.jcublas.util.FFTUtils;

/**
 *
 * Uses jcuda's vector fft rather than
 * the normal cpu based method
 *
 * @author Adam Gibson
 */
public class JCudaVectorFFT extends VectorFFT {
    public JCudaVectorFFT(INDArray x, INDArray z, int fftLength) {
        super(x, z, fftLength);
    }


    public JCudaVectorFFT() {
    }

    public JCudaVectorFFT(INDArray x, INDArray z, int n, int fftLength) {
        super(x, z, n, fftLength);
    }

    public JCudaVectorFFT(INDArray x, INDArray y, INDArray z, int n, int fftLength) {
        super(x, y, z, n, fftLength);
    }

    public JCudaVectorFFT(INDArray x, int fftLength) {
        super(x, fftLength);
    }

    public JCudaVectorFFT(INDArray x) {
        super(x);
    }

    /**
     * Get the handle for the current thread
     * @return the handle for the current thread
     */
    public cufftHandle getHandle() {
        JcudaFft fft = (JcudaFft) Nd4j.getFFt();
        return fft.getHandle();

    }


    @Override
    public void exec() {
        if(!x.isVector() || executed)
            return;
        JcudaFft fft = (JcudaFft) Nd4j.getFFt();
        CudaContext ctx = new CudaContext();
        ctx.initOldStream();
        JCufft.cufftSetStream(fft.getHandle(), ctx.getOldStream());

        INDArray workerArea = Nd4j.create(x.length() * 2);
        try(CublasPointer inputPointer = new CublasPointer(x,ctx);
            CublasPointer workerPointer = new CublasPointer(workerArea,ctx)) {
            JCufft.cufftSetWorkArea(
                    getHandle()
                    , workerPointer.getDevicePointer());

            JCufft.cufftSetStream(
                    getHandle()
                    , ContextHolder.getInstance().getCudaStream());

            JCufft.cufftSetAutoAllocation(
                    getHandle()
                    , JCufft.cufftSetAutoAllocation(getHandle(), 1));


            JCufft.cufftPlan1d(
                    getHandle()
                    , fftLength
                    , FFTUtils.getPlanFor(x.data()), 1);


            if(x.data().dataType() == DataBuffer.Type.FLOAT)
                JCufft.cufftExecC2C(
                        getHandle()
                        , inputPointer.getDevicePointer()
                        , inputPointer.getDevicePointer()
                        , JCufft.CUFFT_FORWARD);
            else
                JCufft.cufftExecZ2Z(
                        getHandle()
                        , inputPointer.getDevicePointer()
                        , inputPointer.getDevicePointer()
                        , JCufft.CUFFT_FORWARD);

            inputPointer.copyToHost();
            ctx.destroy();

        }
        catch(Exception e) {
            throw new RuntimeException(e);
        }

        executed = true;
        this.z = x;
        JCufft.cufftSetStream(fft.getHandle(), ContextHolder.getInstance().getCudaStream());


    }
}

