package org.nd4j.linalg.jcublas.fft.ops;

import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.impl.transforms.VectorIFFT;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.fft.JcudaFft;

/**
 *
 * Uses jcuda's vector ifft rather
 * than normal cpu computation
 *
 * @author Adam Gibson
 */
public class JCudaVectorIFFT extends VectorIFFT {
    public JCudaVectorIFFT(INDArray x, INDArray z, int fftLength) {
        super(x, z, fftLength);
    }

    public JCudaVectorIFFT(INDArray x, INDArray z, int n, int fftLength) {
        super(x, z, n, fftLength);
    }

    public JCudaVectorIFFT(INDArray x, INDArray y, INDArray z, int n, int fftLength) {
        super(x, y, z, n, fftLength);
    }

    public JCudaVectorIFFT(INDArray x, int fftLength) {
        super(x, fftLength);
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
        INDArray workerArea = Nd4j.create(x.length() * 2);
        CublasPointer inputPointer = new CublasPointer(x);
        CublasPointer workerPointer = new CublasPointer(workerArea);

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
                , n
                , cufftType.CUFFT_C2C, 1);


        JCufft.cufftExecC2C(
                getHandle()
                , inputPointer.getDevicePointer()
                , inputPointer.getDevicePointer()
                , JCufft.CUFFT_INVERSE);

        try {
            inputPointer.copyToHost();
            inputPointer.close();
            workerPointer.close();

        } catch (Exception e) {
            e.printStackTrace();
        }

        this.z = x;

    }
}
