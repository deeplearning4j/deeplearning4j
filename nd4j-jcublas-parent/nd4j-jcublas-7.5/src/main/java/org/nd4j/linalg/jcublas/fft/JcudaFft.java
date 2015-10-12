package org.nd4j.linalg.jcublas.fft;

import jcuda.jcufft.JCufft;
import jcuda.jcufft.cufftHandle;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.api.ops.Op;
import org.nd4j.linalg.fft.DefaultFFTInstance;
import org.nd4j.linalg.jcublas.fft.ops.JCudaVectorFFT;

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







    @Override
    protected Op getFftOp(INDArray arr, int n) {
        return new JCudaVectorFFT(arr,n);
    }
}
