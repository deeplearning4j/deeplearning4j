package org.nd4j.linalg.jcublas.fft;

import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.DefaultFFTInstance;
import org.nd4j.linalg.jcublas.CublasPointer;
import org.nd4j.linalg.jcublas.context.ContextHolder;
import org.nd4j.linalg.jcublas.fft.ops.JCudaVectorFFT;
import org.nd4j.linalg.jcublas.fft.ops.JCudaVectorIFFT;
import org.nd4j.linalg.util.ArrayUtil;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 *
 * Uses JCufft rather than
 * the built in FFTs
 *
 * @author Adam Gibson
 */
public class JcudaFft extends DefaultFFTInstance {
    //map of thread names to handles (one fft handle per thread)
    private final Map<String,cufftHandle> handles;


    public JcudaFft() {
        this.handles = new ConcurrentHashMap<>();
        Runtime.getRuntime().addShutdownHook(new Thread(new Runnable() {
            @Override
            public void run() {
                for (cufftHandle handle : handles.values()) {
                    JCufft.cufftDestroy(handle);
                }
            }
        }));

        JCufft.setExceptionsEnabled(true);

    }


    /**
     * Get the handle for the current thread
     * @return the handle for the current thread
     */
    public cufftHandle getHandle() {
        cufftHandle handle =  handles.get(Thread.currentThread().getName());
        if(handle == null) {
            handle = new cufftHandle();
            JCufft.cufftCreate(handle);
            handles.put(Thread.currentThread().getName(), handle);
        }

        return handle;
    }





    private IComplexNDArray doFftn(IComplexNDArray transform,int[] shape,int[] axes,int fftType) {
        IComplexNDArray result = transform.dup();
        if(axes == null)
            axes = ArrayUtil.range(0,transform.shape().length);
        if(shape == null)
            shape = transform.shape();

        INDArray workerArea = Nd4j.create(result.length() * 2);
        try(CublasPointer workerPointer = new CublasPointer(workerArea); CublasPointer pointer = new CublasPointer(result)) {

            JCufft.cufftSetWorkArea(getHandle(),workerPointer.getDevicePointer());

            JCufft.cufftSetStream(
                    getHandle()
                    , ContextHolder.getInstance().getCudaStream());

            JCufft.cufftSetAutoAllocation(
                    getHandle()
                    , JCufft.cufftSetAutoAllocation(getHandle(), 1));



            long[] workAreaLength = new long[]{workerArea.length()  * workerArea.data().getElementSize()};
            if(transform.isVector()) {
                JCufft.cufftMakePlan1d(getHandle()
                        ,shape[0]
                        ,fftType
                        ,1
                        ,workAreaLength);
            }
            else if(transform.isMatrix()) {
                JCufft.cufftMakePlan2d(getHandle()
                        ,shape[0]
                        ,shape[1]
                        ,fftType
                        ,workAreaLength);
            }
            else if(transform.shape().length == 3) {
                JCufft.cufftMakePlan3d(
                        getHandle()
                        ,shape[0]
                        ,shape[1]
                        ,shape[2]
                        ,fftType
                        ,workAreaLength);

            }
            else {
                if(shape == null)
                    shape = transform.shape();

                for(int i = 0; i < result.slices(); i++) {
                    IComplexNDArray slice = transform.slice(i);
                    IComplexNDArray fftedSlice = doFftn(slice, ArrayUtil.removeIndex(shape, 0), ArrayUtil.removeIndex(shape, 0), fftType);
                    result.putSlice(i,fftedSlice);
                }



            }

            if(transform.data().dataType() == DataBuffer.Type.FLOAT)
                JCufft.cufftExecC2C(
                        getHandle()
                        , pointer.getDevicePointer()
                        , pointer.getDevicePointer()
                        , fftType);
            else
                JCufft.cufftExecZ2Z(
                        getHandle()
                        , pointer.getDevicePointer()
                        , pointer.getDevicePointer()
                        , fftType);


            pointer.copyToHost();

        }

        catch(Exception e) {
            throw new RuntimeException(e);
        }


        return result;

    }





    @Override
    public IComplexNDArray rawifftn(IComplexNDArray transform, int[] shape, int[] axes) {
        return doFftn(transform, shape, axes, JCufft.CUFFT_INVERSE);
    }




    @Override
    public IComplexNDArray rawfftn(IComplexNDArray transform, int[] shape, int[] axes) {
        return doFftn(transform, shape, axes, JCufft.CUFFT_FORWARD);
    }



    @Override
    protected Op getFftOp(INDArray arr, int n) {
        return new JCudaVectorFFT(arr,n);
    }
}
