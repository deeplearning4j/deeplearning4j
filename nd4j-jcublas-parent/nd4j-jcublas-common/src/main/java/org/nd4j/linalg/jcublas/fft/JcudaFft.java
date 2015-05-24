package org.nd4j.linalg.jcublas.fft;

import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import jcuda.jcufft.cufftType;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.complex.IComplexNDArray;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;
import org.nd4j.linalg.fft.DefaultFFTInstance;
import org.nd4j.linalg.jcublas.CublasPointer;

import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;

/**
 * @author Adam Gibson
 */
public class JcudaFft extends DefaultFFTInstance {
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


    public cufftHandle getHandle() {
        cufftHandle handle =  handles.get(Thread.currentThread().getName());
        if(handle == null) {
            handle = new cufftHandle();
            JCufft.cufftCreate(handle);
            handles.put(Thread.currentThread().getName(), handle);
        }

        return handle;
    }




    @Override
    public IComplexNDArray fft(INDArray transform, int numElements, int dimension) {
        JCufft.cufftPlan1d(getHandle(), numElements, cufftType.CUFFT_D2Z, 1);
        IComplexNDArray arr = Nd4j.createComplex(transform.shape());
        try(CublasPointer p = new CublasPointer(transform)) {
            CublasPointer p2 = new CublasPointer(arr);
            if(transform.data().dataType() == DataBuffer.Type.DOUBLE)
                JCufft.cufftExecD2Z(getHandle(), p.getDevicePointer(), p2.getDevicePointer());
            else
                JCufft.cufftExecC2R(getHandle(),p.getDevicePointer(),p2.getDevicePointer());


        } catch (Exception e) {
            e.printStackTrace();
        }

        return arr;
    }



    @Override
    public IComplexNDArray fft(IComplexNDArray inputC, int numElements, int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray ifft(INDArray transform, int numElements, int dimension) {
        return null;
    }



    @Override
    public IComplexNDArray ifft(IComplexNDArray inputC, int numElements, int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray ifftn(INDArray transform, int dimension, int numElements) {
        return null;
    }





    @Override
    public IComplexNDArray ifftn(IComplexNDArray transform, int dimension, int numElements) {
        return null;
    }

    @Override
    public IComplexNDArray fftn(IComplexNDArray transform, int dimension, int numElements) {
        return null;
    }

    @Override
    public IComplexNDArray fftn(INDArray transform, int dimension, int numElements) {
        return null;
    }




    @Override
    public IComplexNDArray rawifftn(IComplexNDArray transform, int[] shape, int[] axes) {
        return null;
    }

    @Override
    public IComplexNDArray rawfftn(IComplexNDArray transform, int[] shape, int[] axes) {
        return null;
    }

    @Override
    public IComplexNDArray rawfft(IComplexNDArray transform, int n, int dimension) {
        return null;
    }

    @Override
    public IComplexNDArray rawifft(IComplexNDArray transform, int n, int dimension) {
        return null;
    }


}
